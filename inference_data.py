"""Dependency-free preparation of QuantumLeap rows for ONNX inference."""

from __future__ import annotations

from bisect import bisect_left
from datetime import date, timedelta
from pathlib import Path
import csv
import math


FRACTIONS = (
    "altro_rd",
    "carta_cartone",
    "frazione_organica",
    "ingombranti_misti",
    "legno",
    "metallo",
    "plastica",
    "pulizia_stradale_recupero",
    "raee",
    "rifiuti_cd",
    "selettiva",
    "tessili",
    "vetro",
)
DAILY_ATTRS = (
    "id_civico",
    "frazione",
    "anno_produzione",
    "coord_x",
    "coord_y",
    "densita_locale",
    "zona_raccolta",
    "zona_costruzione",
    "giorni_accumulati",
    "conferimento_t",
)
STATIC_FIELDS = (
    "coord_x",
    "coord_y",
    "densita_locale",
    "zona_raccolta",
    "zona_costruzione",
)


def normalize_zone(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def annual_rows(rows: list[dict], civic_ids: list[str] | None = None) -> list[dict]:
    selected = set(map(str, civic_ids)) if civic_ids else None
    groups: dict[tuple[str, int], dict] = {}
    entity_fractions: dict[tuple[str, int, str], str] = {}
    fraction_entities: dict[tuple[str, int, str], str] = {}
    for row in rows:
        missing = {"entity_id", "timestamp", *DAILY_ATTRS} - row.keys()
        if missing:
            raise ValueError(f"Missing daily fields: {', '.join(sorted(missing))}.")
        civic = str(row["id_civico"])
        if selected is not None and civic not in selected:
            continue
        fraction = str(row["frazione"])
        if fraction not in FRACTIONS:
            raise ValueError(f"Unknown fraction: {fraction}.")
        try:
            year_value = float(row["anno_produzione"])
            if not year_value.is_integer():
                raise ValueError
            year = int(year_value)
            numeric = {
                name: float(row[name])
                for name in (*STATIC_FIELDS[:-1], "giorni_accumulati", "conferimento_t")
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid numeric QuantumLeap value.") from exc
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError("Non-finite numeric QuantumLeap value.")
        static = {
            "coord_x": numeric["coord_x"],
            "coord_y": numeric["coord_y"],
            "densita_locale": numeric["densita_locale"],
            "zona_raccolta": normalize_zone(row["zona_raccolta"]),
            "zona_costruzione": str(row["zona_costruzione"]).strip(),
        }
        key = (civic, year)
        if key not in groups:
            groups[key] = {"id_civico": civic, "anno_produzione": year, **static}
            groups[key].update({f"{name}_t": 0.0 for name in FRACTIONS})
        elif any(groups[key][name] != value for name, value in static.items()):
            raise ValueError(f"Conflicting static fields within civic/year {key}.")
        entity_key = (civic, year, str(row["entity_id"]))
        if entity_key in entity_fractions and entity_fractions[entity_key] != fraction:
            raise ValueError("Conflicting fraction within civic/year/entity ID.")
        entity_fractions[entity_key] = fraction
        fraction_key = (civic, year, fraction)
        if fraction_key in fraction_entities and fraction_entities[fraction_key] != str(row["entity_id"]):
            raise ValueError("Conflicting entity ID within civic/year/fraction.")
        fraction_entities[fraction_key] = str(row["entity_id"])
        groups[key][f"{fraction}_t"] += numeric["conferimento_t"]
    result = sorted(groups.values(), key=lambda item: (item["id_civico"], item["anno_produzione"]))
    for item in result:
        item["totale_t"] = sum(item[f"{name}_t"] for name in FRACTIONS)
    return result


def split_years(start: date, end: date):
    for year in range(start.year, end.year + 1):
        yield year, max(start, date(year, 1, 1)), min(end, date(year, 12, 31))


def latest_annual(annual: list[dict], forecast_year: int) -> list[dict]:
    latest: dict[str, dict] = {}
    for row in annual:
        if row["anno_produzione"] <= forecast_year:
            current = latest.get(row["id_civico"])
            if current is None or row["anno_produzione"] > current["anno_produzione"]:
                latest[row["id_civico"]] = row
    return [latest[key] for key in sorted(latest)]


WEEKDAYS = {
    "lunedi": 0,
    "martedi": 1,
    "mercoledi": 2,
    "giovedi": 3,
    "venerdi": 4,
    "sabato": 5,
    "domenica": 6,
}


def calendar_rules(path: Path) -> dict[tuple[str, str], set[int]]:
    rules: dict[tuple[str, str], set[int]] = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            raw_fraction = row["frazione"].strip()
            fractions = ("plastica", "metallo") if raw_fraction == "plastica_e_metalli" else (raw_fraction,)
            weekday = WEEKDAYS[row["weekday"].strip().lower()]
            for fraction in fractions:
                rules.setdefault((normalize_zone(row["zona_raccolta"]), fraction), set()).add(weekday)
    return rules


def forecast_events(
    calendar_path: Path,
    zones: list[str],
    fractions: list[str],
    start: date,
    end: date,
) -> list[dict]:
    rules = calendar_rules(calendar_path)
    output = []
    for year, year_start, year_end in split_years(start, end):
        first, last = date(year, 1, 1), date(year, 12, 31)
        production_dates = [first + timedelta(days=i) for i in range((last - first).days + 1)]
        extended = [first + timedelta(days=i) for i in range((last - first).days + 15)]
        for zone in sorted(set(map(normalize_zone, zones))):
            for fraction in fractions:
                weekdays = rules.get((zone, fraction))
                collections = extended if not weekdays else [day for day in extended if day.weekday() in weekdays]
                counts: dict[date, int] = {}
                for day in production_dates:
                    position = bisect_left(collections, day)
                    collection = collections[position] if position < len(collections) else last
                    counts[collection] = counts.get(collection, 0) + 1
                for event_date, accumulated in counts.items():
                    if year_start <= event_date <= year_end and event_date.year == year:
                        output.append({
                            "zona_raccolta": zone,
                            "frazione": fraction,
                            "data": event_date,
                            "giorni_accumulati": accumulated,
                            "annual_collection_count": len(counts),
                        })
    return sorted(output, key=lambda item: (item["data"], item["zona_raccolta"], item["frazione"]))


def twin_id(entity_type: str, civic: str, year: int) -> str:
    return f"urn:ngsi-v2:{entity_type}:{civic}:{year}"


def attribute(value: object, ngsi_type: str) -> dict:
    return {"type": ngsi_type, "value": value}
