from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass, field
from datetime import date, datetime
from statistics import median


LAG_FEATURES = (
    "lag_1_collection_t",
    "rolling_7d_sum_t",
    "rolling_30d_sum_t",
    "rolling_90d_median_t",
    "rolling_90d_count",
    "days_since_last_nonzero_collection",
)


def day_number(value: object) -> int:
    if isinstance(value, datetime):
        value = value.date()
    elif not isinstance(value, date):
        value = date.fromisoformat(str(value).split("T", 1)[0])
    return value.toordinal()


@dataclass
class _SeriesHistory:
    days: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    prefix_sums: list[float] = field(default_factory=lambda: [0.0])
    sorted_90d: list[float] = field(default_factory=list)
    start_90d: int = 0
    last_nonzero_day: int | None = None

    def _advance(self, day: int) -> None:
        cutoff = day - 90
        while self.start_90d < len(self.days) and self.days[self.start_90d] < cutoff:
            value = self.values[self.start_90d]
            self.sorted_90d.pop(bisect_left(self.sorted_90d, value))
            self.start_90d += 1

    def add(self, day: int, value: float) -> None:
        if self.days and day < self.days[-1]:
            raise ValueError("Lag history must be added in chronological order.")
        self._advance(day)
        numeric = max(float(value), 0.0)
        self.days.append(day)
        self.values.append(numeric)
        self.prefix_sums.append(self.prefix_sums[-1] + numeric)
        insort(self.sorted_90d, numeric)
        if numeric > 0:
            self.last_nonzero_day = day

    def _sum_since(self, day: int, window_days: int) -> float:
        start = bisect_left(self.days, day - window_days)
        end = bisect_left(self.days, day)
        return self.prefix_sums[end] - self.prefix_sums[start]

    def features(self, day: int) -> dict[str, float]:
        self._advance(day)
        end = bisect_left(self.days, day)
        recent_90d = end - self.start_90d
        median_values = (
            self.sorted_90d
            if end == len(self.days)
            else self.values[self.start_90d : end]
        )
        return {
            "lag_1_collection_t": self.values[end - 1] if end else 0.0,
            "rolling_7d_sum_t": self._sum_since(day, 7),
            "rolling_30d_sum_t": self._sum_since(day, 30),
            "rolling_90d_median_t": (
                float(median(median_values)) if recent_90d else 0.0
            ),
            "rolling_90d_count": float(recent_90d),
            "days_since_last_nonzero_collection": (
                float(day - self.last_nonzero_day)
                if self.last_nonzero_day is not None
                and self.last_nonzero_day < day
                else 0.0
            ),
        }


class LagState:
    def __init__(self) -> None:
        self._series: dict[tuple[str, str], _SeriesHistory] = {}

    def add(self, civic: object, fraction: object, day: object, value: float) -> None:
        key = (str(civic), str(fraction))
        self._series.setdefault(key, _SeriesHistory()).add(day_number(day), value)

    def features(
        self, civic: object, fraction: object, day: object
    ) -> dict[str, float]:
        history = self._series.get((str(civic), str(fraction)))
        if history is None:
            return {name: 0.0 for name in LAG_FEATURES}
        return history.features(day_number(day))
