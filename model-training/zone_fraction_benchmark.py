#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import subprocess
import time
import warnings

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX


CANONICAL_FRACTIONS = (
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
EPISODIC_FRACTIONS = (
    "ingombranti_misti",
    "legno",
    "raee",
    "rifiuti_cd",
    "selettiva",
    "tessili",
)
ANNUAL_QUANTITY_COLS = tuple(f"{name}_t" for name in CANONICAL_FRACTIONS)
WEEKDAYS = {
    "lunedi": 0,
    "martedi": 1,
    "mercoledi": 2,
    "giovedi": 3,
    "venerdi": 4,
    "sabato": 5,
    "domenica": 6,
}
NUMERIC_FEATURES = [
    "giorni_dal_ritiro_precedente",
    "month",
    "day_of_week",
    "week_of_year",
    "day_of_year_sin",
    "day_of_year_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_holiday",
    "is_after_holiday",
    "is_summer",
    "is_christmas_period",
    "lag_1_collection_t",
    "lag_7d_collection_t",
    "rolling_4_collection_mean_t",
    "rolling_4_collection_median_t",
    "rolling_7d_mean_t",
    "rolling_7d_sum_t",
    "rolling_7d_count",
    "lag_1_missing",
    "lag_7d_missing",
    "rolling_4_missing",
    "rolling_7d_missing",
]
CATEGORICAL_FEATURES = ["zona_raccolta", "frazione", "cluster_label"]
SERIES_COLUMNS = ["scenario_seed", "zona_raccolta", "frazione", "cluster_label"]
ARIMA_SERIES_COLUMNS = ["scenario_seed", "zona_raccolta", "frazione"]
ARIMA_ORDERS = ((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1))
ARIMA_SEASONAL_ORDERS = ((0, 0, 0, 0), (1, 0, 0, 7))
SARIMAX_EXOG_COLUMNS = (
    "giorni_dal_ritiro_precedente",
    "is_holiday",
    "is_after_holiday",
)
CIVIC_BAYES_NUMERIC_FEATURES = (
    "log_historical_share",
    "log_last_share",
    "log_rolling_4_share",
    "rolling_4_share_std",
    "log_history_count",
    "coord_x",
    "coord_y",
    "densita_locale",
)
CIVIC_BAYES_CATEGORICAL_FEATURES = ("zona_raccolta", "frazione")
CIVIC_INTERVAL_LEVELS = (0.80, 0.85, 0.90, 0.95, 0.975)
CIVIC_BAYES_STD_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
CIVIC_DIRICHLET_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
CIVIC_POSTERIOR_DRAWS = 200
CIVIC_BAYES_MAX_TRAIN_ROWS = 500_000


@dataclass(frozen=True)
class PreparedOutputs:
    withdrawals: Path
    static_civics: Path
    manifest: Path


def sql_lit(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def table_schema(con: duckdb.DuckDBPyConnection, view: str) -> dict[str, str]:
    return {
        row[0]: row[1]
        for row in con.execute(f"DESCRIBE SELECT * FROM {sql_ident(view)}").fetchall()
    }


def require_columns(schema: dict[str, str], required: set[str], label: str) -> None:
    if missing := required - schema.keys():
        raise ValueError(f"{label}: missing columns: {', '.join(sorted(missing))}")


def scalar(con: duckdb.DuckDBPyConnection, query: str) -> object:
    return con.execute(query).fetchone()[0]


def check_zero(con: duckdb.DuckDBPyConnection, query: str, message: str) -> None:
    count = int(scalar(con, query))
    if count:
        raise ValueError(f"{message}: {count} invalid rows")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_version(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def normalize_token(value: object) -> str:
    text = str(value).strip().lower()
    replacements = str.maketrans("àèéìòù", "aeeiou")
    return text.translate(replacements).replace(" ", "_").replace("-", "_")


def expand_fraction(value: object) -> tuple[str, ...]:
    token = normalize_token(value)
    if token == "plastica_e_metalli":
        return ("plastica", "metallo")
    return (token,)


def parse_active(value: object) -> bool:
    if value is None or str(value).strip() == "":
        return True
    return normalize_token(value) in {"1", "true", "si", "yes", "y"}


def read_calendar(
    path: Path,
    zones: list[str],
    fractions: list[str],
    year: int,
    spillover_days: int,
    missing_mode: str,
) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    if "frazione" not in fields or "zona_raccolta" not in fields:
        raise ValueError("calendar: zona_raccolta and frazione are required")
    if "weekday" not in fields and "data" not in fields:
        raise ValueError("calendar: either weekday or data is required")

    start = date(year, 1, 1)
    end = date(year, 12, 31) + timedelta(days=spillover_days)
    extended_start = start - timedelta(days=14)
    events: set[tuple[str, str, date]] = set()
    specifications: set[tuple[str, str, str]] = set()

    for row in rows:
        if "raccolta" in fields and not parse_active(row.get("raccolta")):
            continue
        zone = str(row["zona_raccolta"]).strip()
        if zone not in zones:
            raise ValueError(f"calendar: unknown zone {zone!r}")
        for fraction in expand_fraction(row["frazione"]):
            if fraction not in CANONICAL_FRACTIONS:
                raise ValueError(f"calendar: unknown fraction {fraction!r}")
            if "data" in fields and str(row.get("data", "")).strip():
                event_date = date.fromisoformat(str(row["data"]).strip())
                specification = (zone, fraction, event_date.isoformat())
                if specification in specifications:
                    raise ValueError(f"calendar: duplicate event {specification}")
                specifications.add(specification)
                if extended_start <= event_date <= end:
                    events.add((zone, fraction, event_date))
            else:
                weekday_token = normalize_token(row.get("weekday", ""))
                if weekday_token not in WEEKDAYS:
                    raise ValueError(f"calendar: invalid weekday {row.get('weekday')!r}")
                specification = (zone, fraction, weekday_token)
                if specification in specifications:
                    raise ValueError(f"calendar: duplicate rule {specification}")
                specifications.add(specification)
                current = extended_start
                while current <= end:
                    if current.weekday() == WEEKDAYS[weekday_token]:
                        events.add((zone, fraction, current))
                    current += timedelta(days=1)

    all_dates = pd.date_range(extended_start, end, freq="D").date
    for zone in zones:
        for fraction in fractions:
            if not any(z == zone and f == fraction for z, f, _ in events):
                if missing_mode == "error":
                    raise ValueError(
                        f"calendar missing for zone={zone!r}, fraction={fraction!r}"
                    )
                events.update((zone, fraction, day) for day in all_dates)

    frame = pd.DataFrame(
        sorted(events, key=lambda item: (item[0], item[1], item[2])),
        columns=["zona_raccolta", "frazione", "data"],
    )
    frame["data"] = pd.to_datetime(frame["data"])
    frame["giorni_dal_ritiro_precedente"] = (
        frame.groupby(["zona_raccolta", "frazione"], sort=False)["data"]
        .diff()
        .dt.days
        .astype("Int64")
    )
    return frame[frame["data"].dt.date >= start].reset_index(drop=True)


def easter_date(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def italian_holidays(year: int) -> set[date]:
    easter = easter_date(year)
    return {
        date(year, 1, 1),
        date(year, 1, 6),
        easter,
        easter + timedelta(days=1),
        date(year, 4, 25),
        date(year, 5, 1),
        date(year, 6, 2),
        date(year, 8, 15),
        date(year, 11, 1),
        date(year, 12, 8),
        date(year, 12, 25),
        date(year, 12, 26),
    }


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dates = result["data"].dt
    result["month"] = dates.month.astype("int16")
    result["day_of_week"] = dates.dayofweek.astype("int16")
    result["week_of_year"] = dates.isocalendar().week.astype("int16")
    day_of_year = dates.dayofyear.to_numpy(dtype=float)
    result["day_of_year_sin"] = np.sin(2 * math.pi * day_of_year / 366.0)
    result["day_of_year_cos"] = np.cos(2 * math.pi * day_of_year / 366.0)
    result["day_of_week_sin"] = np.sin(
        2 * math.pi * result["day_of_week"].to_numpy(dtype=float) / 7.0
    )
    result["day_of_week_cos"] = np.cos(
        2 * math.pi * result["day_of_week"].to_numpy(dtype=float) / 7.0
    )
    all_holidays = set()
    for current_year in sorted(result["data"].dt.year.unique()):
        all_holidays.update(italian_holidays(int(current_year)))
    day_values = result["data"].dt.date
    result["is_holiday"] = day_values.map(lambda day: int(day in all_holidays))
    result["is_after_holiday"] = day_values.map(
        lambda day: int(day - timedelta(days=1) in all_holidays)
    )
    result["is_summer"] = dates.month.isin([6, 7, 8]).astype("int8")
    result["is_christmas_period"] = (
        ((dates.month == 12) & (dates.day >= 15))
        | ((dates.month == 1) & (dates.day <= 6))
    ).astype("int8")

    grouped = result.groupby(SERIES_COLUMNS, sort=False)["conferimento_t"]
    result["lag_1_collection_t"] = grouped.shift(1)
    result["rolling_4_collection_mean_t"] = grouped.transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).mean()
    )
    result["rolling_4_collection_median_t"] = grouped.transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).median()
    )
    result["lag_7d_collection_t"] = np.nan
    result["rolling_7d_mean_t"] = np.nan
    result["rolling_7d_sum_t"] = np.nan
    result["rolling_7d_count"] = 0
    for _, series in result.groupby(SERIES_COLUMNS, sort=False):
        series = series.sort_values("data", kind="stable")
        indices = series.index.to_list()
        dates_for_group = series["data"].to_list()
        values = series["conferimento_t"].to_numpy(dtype=float)
        values_by_date = dict(zip(dates_for_group, values, strict=True))
        window_start = 0
        for position, (row_index, current_date) in enumerate(
            zip(indices, dates_for_group, strict=True)
        ):
            cutoff = current_date - pd.Timedelta(days=7)
            while dates_for_group[window_start] < cutoff:
                window_start += 1
            history = values[window_start:position]
            result.at[row_index, "lag_7d_collection_t"] = values_by_date.get(cutoff)
            if len(history):
                result.at[row_index, "rolling_7d_mean_t"] = float(history.mean())
                result.at[row_index, "rolling_7d_sum_t"] = float(history.sum())
                result.at[row_index, "rolling_7d_count"] = len(history)
    result["lag_1_missing"] = result["lag_1_collection_t"].isna().astype("int8")
    result["lag_7d_missing"] = result["lag_7d_collection_t"].isna().astype("int8")
    result["rolling_4_missing"] = (
        result["rolling_4_collection_mean_t"].isna().astype("int8")
    )
    result["rolling_7d_missing"] = (
        result["rolling_7d_mean_t"].isna().astype("int8")
    )
    return result


def write_parquet(
    con: duckdb.DuckDBPyConnection, frame: pd.DataFrame, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con.register("output_frame", frame)
    con.execute(
        f"COPY output_frame TO {sql_lit(path)} "
        "(FORMAT parquet, COMPRESSION zstd)"
    )
    con.unregister("output_frame")


def mismatch_condition(left: str, right: str, abs_tol: float, rel_tol: float) -> str:
    return (
        f"abs(coalesce({left}, 0.0) - coalesce({right}, 0.0)) > "
        f"greatest({abs_tol}, {rel_tol} * "
        f"greatest(abs(coalesce({left}, 0.0)), abs(coalesce({right}, 0.0))))"
    )


def classify_collections(
    collections_path: Path,
    assignments_path: Path,
    output_path: Path,
) -> Path:
    collections_path = collections_path.resolve()
    assignments_path = assignments_path.resolve()
    if not collections_path.is_file():
        raise FileNotFoundError(f"collections input does not exist: {collections_path}")
    if not assignments_path.is_file():
        raise FileNotFoundError(f"K-means assignments do not exist: {assignments_path}")

    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW collections AS "
        f"SELECT * FROM read_parquet({sql_lit(collections_path)})"
    )
    con.execute(
        f"CREATE VIEW assignments AS "
        f"SELECT * FROM read_parquet({sql_lit(assignments_path)})"
    )
    collection_schema = table_schema(con, "collections")
    assignment_schema = table_schema(con, "assignments")
    year_col = (
        "anno_from_col"
        if "anno_from_col" in assignment_schema
        else "anno_produzione"
    )
    require_columns(
        collection_schema, {"id_civico", "anno_produzione"}, "collections"
    )
    require_columns(
        assignment_schema, {"id_civico", year_col, "cluster_label"}, "assignments"
    )
    check_zero(
        con,
        f"""
        SELECT count(*)
        FROM (
            SELECT cast(id_civico AS VARCHAR), try_cast({sql_ident(year_col)} AS INTEGER)
            FROM assignments
            GROUP BY 1, 2
            HAVING count(*) <> 1
        )
        """,
        "K-means civic/year assignments must be unique",
    )
    check_zero(
        con,
        f"""
        SELECT count(*)
        FROM collections c
        LEFT JOIN assignments a
          ON cast(c.id_civico AS VARCHAR) = cast(a.id_civico AS VARCHAR)
         AND try_cast(c.anno_produzione AS INTEGER)
             = try_cast(a.{sql_ident(year_col)} AS INTEGER)
        WHERE a.cluster_label IS NULL
        """,
        "every collection row must have a cluster assignment",
    )

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collection_select = (
        "c.* EXCLUDE (cluster_label)"
        if "cluster_label" in collection_schema
        else "c.*"
    )
    con.execute(
        f"""
        COPY (
            SELECT
                {collection_select},
                try_cast(a.cluster_label AS INTEGER) AS cluster_label
            FROM collections c
            JOIN assignments a
              ON cast(c.id_civico AS VARCHAR) = cast(a.id_civico AS VARCHAR)
             AND try_cast(c.anno_produzione AS INTEGER)
                 = try_cast(a.{sql_ident(year_col)} AS INTEGER)
        )
        TO {sql_lit(output_path)}
        (FORMAT parquet, COMPRESSION zstd)
        """
    )
    return output_path


def prepare_complete_dataset(
    *,
    annual_path: Path,
    production_path: Path | None,
    collections_path: Path,
    calendar_path: Path,
    output_dir: Path,
    year: int,
    annual_seed: int,
    scenario_seed: int,
    spillover_days: int,
    missing_calendar_mode: str,
    generator_version: str,
    abs_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-9,
) -> PreparedOutputs:
    input_paths = {
        "annual": annual_path.resolve(),
        "collections": collections_path.resolve(),
        "calendar": calendar_path.resolve(),
    }
    if production_path is not None:
        input_paths["production"] = production_path.resolve()
    for label, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} input does not exist: {path}")
    with input_paths["calendar"].open(encoding="utf-8-sig", newline="") as stream:
        calendar_rows = sum(1 for _ in csv.DictReader(stream))
    if year < 1 or spillover_days < 0:
        raise ValueError("year must be positive and spillover_days non-negative")

    output_dir = output_dir.resolve()
    data_dir = output_dir / "data"
    static_path = data_dir / "static_civics.parquet"
    withdrawals_path = (
        data_dir
        / "withdrawals"
        / f"scenario_seed={scenario_seed}"
        / "part-000.parquet"
    )
    manifest_path = output_dir / "manifest.json"

    con = duckdb.connect()
    data_labels = ("annual", "collections") + (
        ("production",) if production_path is not None else ()
    )
    for label in data_labels:
        con.execute(
            f"CREATE VIEW {sql_ident(label)} AS "
            f"SELECT * FROM read_parquet({sql_lit(input_paths[label])})"
        )

    schemas = {label: table_schema(con, label) for label in data_labels}
    annual_required = {
        "anno",
        "id_civico",
        "zona_raccolta",
        "zona_costruzione",
        "totale_t",
        *ANNUAL_QUANTITY_COLS,
    }
    if production_path is not None:
        annual_required.update(
            {
                "prg_error",
                "zona_raccolta_inferita",
                "zona_costruzione_riassegnata",
                "distanza_inferenza_zona_raccolta_gradi",
                "distanza_riassegnazione_prg",
            }
        )
    require_columns(
        schemas["annual"],
        annual_required,
        "annual",
    )
    if production_path is not None:
        require_columns(
            schemas["production"],
            {
                "anno_produzione",
                "data",
                "id_civico",
                "zona_raccolta",
                "frazione",
                "produzione_teorica_t",
                "conferimento_data",
                "raccolta_prevista",
            },
            "production",
        )
    require_columns(
        schemas["collections"],
        {
            "anno_produzione",
            "data",
            "id_civico",
            "zona_raccolta",
            "frazione",
            "conferimento_t",
            "giorni_accumulati",
            "cluster_label",
        },
        "collections",
    )

    annual_rows = int(scalar(con, "SELECT count(*) FROM annual"))
    production_rows = (
        int(scalar(con, "SELECT count(*) FROM production"))
        if production_path is not None
        else None
    )
    collection_rows = int(scalar(con, "SELECT count(*) FROM collections"))
    check_zero(
        con,
        "SELECT count(*) - count(DISTINCT id_civico) FROM annual",
        "annual id_civico must be unique",
    )
    check_zero(
        con,
        f"SELECT count(*) FROM annual WHERE try_cast(anno AS INTEGER) <> {year} "
        "OR anno IS NULL",
        "annual year mismatch",
    )
    check_zero(
        con,
        "SELECT count(*) FROM annual WHERE zona_raccolta IS NULL "
        "OR trim(cast(zona_raccolta AS VARCHAR)) = ''",
        "annual zona_raccolta",
    )
    annual_value_checks = " OR ".join(
        f"{sql_ident(column)} IS NULL OR NOT isfinite(try_cast({sql_ident(column)} AS DOUBLE)) "
        f"OR try_cast({sql_ident(column)} AS DOUBLE) < 0"
        for column in (*ANNUAL_QUANTITY_COLS, "totale_t")
    )
    check_zero(
        con,
        f"SELECT count(*) FROM annual WHERE {annual_value_checks}",
        "annual quantities",
    )
    annual_sum = " + ".join(sql_ident(column) for column in ANNUAL_QUANTITY_COLS)
    check_zero(
        con,
        f"SELECT count(*) FROM annual WHERE "
        f"abs(totale_t - ({annual_sum})) > greatest({abs_tolerance}, "
        f"{relative_tolerance} * greatest(abs(totale_t), abs({annual_sum})))",
        "annual totale_t reconciliation",
    )

    start = date(year, 1, 1)
    production_end = date(year, 12, 31)
    spillover_end = production_end + timedelta(days=spillover_days)
    dated_inputs = [("collections", "data", "conferimento_t", spillover_end)]
    if production_path is not None:
        dated_inputs.insert(
            0, ("production", "data", "produzione_teorica_t", production_end)
        )
    for view, date_col, amount_col, allowed_end in dated_inputs:
        check_zero(
            con,
            f"SELECT count(*) FROM {view} WHERE try_cast({date_col} AS DATE) IS NULL",
            f"{view} invalid {date_col}",
        )
        check_zero(
            con,
            f"SELECT count(*) FROM {view} WHERE try_cast({date_col} AS DATE) "
            f"NOT BETWEEN DATE {sql_lit(start)} AND DATE {sql_lit(allowed_end)}",
            f"{view} date range",
        )
        check_zero(
            con,
            f"SELECT count(*) FROM {view} WHERE {amount_col} IS NULL "
            f"OR NOT isfinite(try_cast({amount_col} AS DOUBLE)) "
            f"OR try_cast({amount_col} AS DOUBLE) < 0",
            f"{view} quantities",
        )
        check_zero(
            con,
            f"SELECT count(*) FROM {view} WHERE try_cast(anno_produzione AS INTEGER) "
            f"<> {year} OR anno_produzione IS NULL",
            f"{view} year mismatch",
        )
        check_zero(
            con,
            f"SELECT count(*) FROM {view} WHERE zona_raccolta IS NULL "
            "OR trim(cast(zona_raccolta AS VARCHAR)) = '' "
            "OR frazione IS NULL OR trim(cast(frazione AS VARCHAR)) = ''",
            f"{view} keys",
        )
    check_zero(
        con,
        "SELECT count(*) FROM collections WHERE try_cast(cluster_label AS INTEGER) IS NULL",
        "collections cluster_label",
    )

    if production_path is not None:
        check_zero(
            con,
            "SELECT count(*) FROM production "
            "WHERE try_cast(conferimento_data AS DATE) IS NULL",
            "production invalid conferimento_data",
        )
        check_zero(
            con,
            f"SELECT count(*) FROM production WHERE try_cast(conferimento_data AS DATE) "
            f"NOT BETWEEN DATE {sql_lit(start)} AND DATE {sql_lit(spillover_end)} "
            "OR try_cast(conferimento_data AS DATE) < try_cast(data AS DATE)",
            "production conferimento_data range",
        )
        check_zero(
            con,
            "SELECT count(*) FROM production "
            "WHERE try_cast(raccolta_prevista AS INTEGER) "
            "NOT IN (0, 1) OR raccolta_prevista IS NULL",
            "production raccolta_prevista",
        )

    canonical_sql = ", ".join(sql_lit(value) for value in CANONICAL_FRACTIONS)
    validated_views = (
        ("production", "collections")
        if production_path is not None
        else ("collections",)
    )
    for view in validated_views:
        check_zero(
            con,
            f"SELECT count(*) FROM {view} WHERE trim(cast(frazione AS VARCHAR)) "
            f"NOT IN ({canonical_sql})",
            f"{view} unknown fractions",
        )

    annual_totals = {
        fraction: float(
            scalar(con, f"SELECT sum({sql_ident(fraction + '_t')}) FROM annual")
        )
        for fraction in CANONICAL_FRACTIONS
    }
    active_fractions = sorted(
        fraction
        for fraction, total in annual_totals.items()
        if total > abs_tolerance
    )
    for view in validated_views:
        present = {
            str(row[0])
            for row in con.execute(
                f"SELECT DISTINCT trim(cast(frazione AS VARCHAR)) FROM {view}"
            ).fetchall()
        }
        if present != set(active_fractions):
            raise ValueError(
                f"{view} fractions differ from positive annual totals: "
                f"present={sorted(present)}, expected={active_fractions}"
            )

    zones = [
        str(row[0])
        for row in con.execute(
            "SELECT DISTINCT trim(cast(zona_raccolta AS VARCHAR)) AS zone "
            "FROM annual ORDER BY zone"
        ).fetchall()
    ]
    schedule = read_calendar(
        input_paths["calendar"],
        zones,
        active_fractions,
        year,
        spillover_days,
        missing_calendar_mode,
    )
    schedule = schedule[schedule["frazione"].isin(active_fractions)].copy()
    zone_clusters = con.execute(
        """
        SELECT DISTINCT
            trim(cast(zona_raccolta AS VARCHAR)) AS zona_raccolta,
            try_cast(cluster_label AS INTEGER) AS cluster_label
        FROM collections
        ORDER BY 1, 2
        """
    ).df()
    schedule = schedule.merge(
        zone_clusters,
        on="zona_raccolta",
        how="inner",
        validate="many_to_many",
    )
    schedule["scenario_seed"] = scenario_seed
    schedule["anno_produzione"] = year

    con.execute(
        """
        CREATE TEMP VIEW collection_cluster_agg AS
        SELECT
            try_cast(data AS DATE) AS data,
            trim(cast(zona_raccolta AS VARCHAR)) AS zona_raccolta,
            trim(cast(frazione AS VARCHAR)) AS frazione,
            try_cast(cluster_label AS INTEGER) AS cluster_label,
            sum(try_cast(conferimento_t AS DOUBLE)) AS conferimento_t
        FROM collections
        GROUP BY 1, 2, 3, 4
        """
    )
    con.execute(
        """
        CREATE TEMP VIEW collection_agg AS
        SELECT data, zona_raccolta, frazione, sum(conferimento_t) AS conferimento_t
        FROM collection_cluster_agg
        GROUP BY 1, 2, 3
        """
    )
    if production_path is not None:
        con.execute(
            """
            CREATE TEMP VIEW production_agg AS
            SELECT
                try_cast(conferimento_data AS DATE) AS data,
                trim(cast(zona_raccolta AS VARCHAR)) AS zona_raccolta,
                trim(cast(frazione AS VARCHAR)) AS frazione,
                sum(try_cast(produzione_teorica_t AS DOUBLE)) AS produzione_teorica_t
            FROM production
            GROUP BY 1, 2, 3
            """
        )
        mismatch = mismatch_condition(
            "p.produzione_teorica_t",
            "c.conferimento_t",
            abs_tolerance,
            relative_tolerance,
        )
        check_zero(
            con,
            "SELECT count(*) FROM production_agg p FULL OUTER JOIN collection_agg c "
            "USING (data, zona_raccolta, frazione) "
            f"WHERE {mismatch}",
            "production/collections reconciliation",
        )

    collection_totals = {
        str(fraction): float(total)
        for fraction, total in con.execute(
            "SELECT frazione, sum(conferimento_t) FROM collection_agg GROUP BY 1"
        ).fetchall()
    }
    for fraction, expected in annual_totals.items():
        actual = collection_totals.get(fraction, 0.0)
        tolerance = max(
            abs_tolerance, relative_tolerance * max(abs(expected), abs(actual))
        )
        if abs(expected - actual) > tolerance:
            raise ValueError(
                f"annual/collections reconciliation failed for {fraction}: "
                f"annual={expected}, collections={actual}"
            )

    targets = con.execute(
        "SELECT data, zona_raccolta, frazione, cluster_label, conferimento_t "
        "FROM collection_cluster_agg "
        "ORDER BY zona_raccolta, frazione, cluster_label, data"
    ).df()
    targets["data"] = pd.to_datetime(targets["data"])
    target_keys = ["data", "zona_raccolta", "frazione", "cluster_label"]
    schedule_keys = schedule[target_keys]
    off_schedule = targets.merge(
        schedule_keys,
        on=target_keys,
        how="left",
        indicator=True,
    )
    off_schedule = off_schedule[off_schedule["_merge"] == "left_only"]
    if not off_schedule.empty:
        raise ValueError(f"collections contain {len(off_schedule)} off-schedule targets")

    withdrawals = schedule.merge(
        targets,
        on=target_keys,
        how="left",
        validate="one_to_one",
    )
    withdrawals["conferimento_t"] = withdrawals["conferimento_t"].fillna(0.0)
    withdrawals = withdrawals[
        [
            "scenario_seed",
            "anno_produzione",
            "data",
            "zona_raccolta",
            "frazione",
            "cluster_label",
            "conferimento_t",
            "giorni_dal_ritiro_precedente",
        ]
    ].sort_values(
        [*SERIES_COLUMNS, "data"],
        kind="stable",
    )
    if withdrawals.duplicated(
        ["scenario_seed", "data", "zona_raccolta", "frazione", "cluster_label"]
    ).any():
        raise ValueError("withdrawal grid contains duplicate keys")
    withdrawals = add_features(withdrawals.reset_index(drop=True))

    data_dir.mkdir(parents=True, exist_ok=True)
    static_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT
                * EXCLUDE (id_civico, zona_raccolta),
                cast(id_civico AS VARCHAR) AS id_civico,
                trim(cast(zona_raccolta AS VARCHAR)) AS zona_raccolta
            FROM annual
            ORDER BY cast(id_civico AS VARCHAR)
        )
        TO {sql_lit(static_path)}
        (FORMAT parquet, COMPRESSION zstd)
        """
    )
    write_parquet(con, withdrawals, withdrawals_path)

    root = Path(__file__).resolve().parents[1]
    date_range = {"withdrawals_to": spillover_end.isoformat()}
    if production_path is not None:
        date_range.update(
            {
                "production_from": start.isoformat(),
                "production_to": production_end.isoformat(),
            }
        )
    else:
        date_range.update(
            {
                "history_from": start.isoformat(),
                "history_to": production_end.isoformat(),
            }
        )
    manifest = {
        "contract": "zone-fraction-scheduled-withdrawal-v1",
        "year": year,
        "annual_seed": annual_seed,
        "scenario_seed": scenario_seed,
        "generator_version": generator_version,
        "code_version": git_version(root),
        "units": "tonnes",
        "date_range": date_range,
        "spillover_days": spillover_days,
        "missing_calendar_mode": missing_calendar_mode,
        "zones": zones,
        "fractions": active_fractions,
        "inputs": {
            label: {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": (
                    annual_rows
                    if label == "annual"
                    else production_rows
                    if label == "production"
                    else collection_rows
                    if label == "collections"
                    else calendar_rows
                ),
                "schema": schemas.get(label),
            }
            for label, path in input_paths.items()
        },
        "validation": {
            "unique_civics": annual_rows,
            "missing_prg": int(
                scalar(con, "SELECT count(*) FROM annual WHERE zona_costruzione IS NULL")
            ),
            "scheduled_withdrawals": len(withdrawals),
            "zero_withdrawals": int((withdrawals["conferimento_t"] == 0).sum()),
            "positive_target_groups": len(targets),
            "off_schedule_targets": 0,
            "total_t": float(withdrawals["conferimento_t"].sum()),
            "annual_totals_t": annual_totals,
        },
        "outputs": {
            "static_civics": str(static_path),
            "withdrawals": str(withdrawals_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return PreparedOutputs(withdrawals_path, static_path, manifest_path)


def chronological_split(frame: pd.DataFrame, year: int) -> pd.Series:
    dates = frame["data"]
    return pd.Series(
        np.select(
            [
                (dates.dt.year == year) & (dates.dt.month <= 9),
                (dates.dt.year == year) & dates.dt.month.isin([10, 11]),
            ],
            ["train", "validation"],
            default="test",
        ),
        index=frame.index,
        dtype="string",
    )


def expanding_validation_folds(
    frame: pd.DataFrame, folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    dates = np.sort(frame["data"].unique())
    if folds < 2:
        raise ValueError("validation_folds must be at least 2")
    if len(dates) <= folds:
        raise ValueError(
            f"need more than {folds} distinct dates for k-fold validation"
        )
    return [
        (
            frame.index[frame["data"].isin(dates[train_dates])].to_numpy(),
            frame.index[frame["data"].isin(dates[validation_dates])].to_numpy(),
        )
        for train_dates, validation_dates in TimeSeriesSplit(
            n_splits=folds
        ).split(dates)
    ]


def build_ridge() -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numeric", numeric, NUMERIC_FEATURES),
                        ("categorical", categorical, CATEGORICAL_FEATURES),
                    ],
                    sparse_threshold=0.0,
                ),
            ),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def build_sgd(seed: int) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numeric", numeric, NUMERIC_FEATURES),
                        ("categorical", categorical, CATEGORICAL_FEATURES),
                    ],
                    sparse_threshold=0.0,
                ),
            ),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=SGDRegressor(
                        loss="squared_error",
                        penalty="l2",
                        alpha=0.0001,
                        learning_rate="invscaling",
                        eta0=0.01,
                        power_t=0.25,
                        max_iter=3,
                        tol=None,
                        random_state=seed,
                    ),
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )


def build_histogram_model(seed: int, loss: str, quantile: float | None = None) -> Pipeline:
    categorical = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                NUMERIC_FEATURES,
            ),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )
    model = HistGradientBoostingRegressor(
        loss=loss,
        quantile=quantile,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def training_means(
    train: pd.DataFrame,
) -> tuple[dict[tuple[str, str, int], float], float]:
    by_group = (
        train.groupby(["zona_raccolta", "frazione", "cluster_label"])[
            "conferimento_t"
        ]
        .mean()
        .to_dict()
    )
    return by_group, float(train["conferimento_t"].mean())


def fill_from_training_mean(
    frame: pd.DataFrame,
    values: pd.Series,
    means: dict[tuple[str, str, int], float],
    global_mean: float,
) -> np.ndarray:
    result = pd.to_numeric(values, errors="coerce").copy()
    missing = result.isna()
    if missing.any():
        result.loc[missing] = [
            means.get((str(zone), str(fraction), int(cluster)), global_mean)
            for zone, fraction, cluster in frame.loc[
                missing, ["zona_raccolta", "frazione", "cluster_label"]
            ].itertuples(index=False, name=None)
        ]
    return np.maximum(result.to_numpy(dtype=float), 0.0)


def fit_benchmark_models(
    train: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    seed: int,
    hgb_loss: str,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, float]]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    result = prediction_frame.copy()
    x_train = train[features]
    y_train = train["conferimento_t"].to_numpy(dtype=float)
    means, global_mean = training_means(train)
    result["pred_last_t"] = fill_from_training_mean(
        result, result["lag_1_collection_t"], means, global_mean
    )
    result["pred_rolling4_t"] = fill_from_training_mean(
        result, result["rolling_4_collection_mean_t"], means, global_mean
    )
    result["pred_train_mean_t"] = [
        means.get((str(zone), str(fraction), int(cluster)), global_mean)
        for zone, fraction, cluster in result[
            ["zona_raccolta", "frazione", "cluster_label"]
        ].itertuples(index=False, name=None)
    ]

    fit_times = {}
    fit_started_at = time.perf_counter()
    ridge = build_ridge()
    ridge.fit(x_train, y_train)
    fit_times["ridge"] = time.perf_counter() - fit_started_at
    result["pred_ridge_t"] = np.maximum(ridge.predict(result[features]), 0.0)

    fit_started_at = time.perf_counter()
    sgd = build_sgd(seed)
    sgd.fit(x_train, y_train)
    fit_times["sgd"] = time.perf_counter() - fit_started_at
    result["pred_sgd_t"] = np.maximum(sgd.predict(result[features]), 0.0)

    fit_started_at = time.perf_counter()
    histogram = build_histogram_model(seed, hgb_loss)
    histogram.fit(x_train, y_train)
    fit_times["hist_gradient_boosting"] = time.perf_counter() - fit_started_at
    result["pred_hist_t"] = np.maximum(histogram.predict(result[features]), 0.0)

    models = {
        "ridge": ridge,
        "sgd": sgd,
        "hist_gradient_boosting": histogram,
    }
    episodic_train = train[train["frazione"].isin(EPISODIC_FRACTIONS)]
    result["pred_q50_t"] = np.nan
    result["pred_q90_t"] = np.nan
    if not episodic_train.empty:
        fit_started_at = time.perf_counter()
        mask = result["frazione"].isin(EPISODIC_FRACTIONS)
        for name, quantile in (("q50", 0.5), ("q90", 0.9)):
            model = build_histogram_model(seed, "quantile", quantile)
            model.fit(
                episodic_train[features],
                episodic_train["conferimento_t"].to_numpy(dtype=float),
            )
            if mask.any():
                result.loc[mask, f"pred_{name}_t"] = np.maximum(
                    model.predict(result.loc[mask, features]), 0.0
                )
            models[name] = model
        if mask.any():
            result.loc[mask, "pred_q90_t"] = np.maximum(
                result.loc[mask, "pred_q90_t"],
                result.loc[mask, "pred_q50_t"],
            )
        fit_times["hist_quantiles"] = time.perf_counter() - fit_started_at
    return result, models, fit_times


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    denominator = float(np.abs(actual).sum())
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "wape": float(np.abs(actual - predicted).sum() / denominator)
        if denominator
        else math.nan,
        "r2": float(r2_score(actual, predicted, force_finite=False))
        if len(actual) >= 2
        else math.nan,
    }


def metric_rows(
    frame: pd.DataFrame, model: str, prediction_col: str
) -> list[dict[str, object]]:
    rows = []
    for split in ("train", "validation", "test"):
        split_frame = frame[frame["split"] == split]
        if split_frame.empty:
            continue
        for fraction, group in split_frame.groupby("frazione", sort=True):
            values = regression_metrics(
                group["conferimento_t"].to_numpy(dtype=float),
                group[prediction_col].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "frazione": fraction,
                    **values,
                    "pinball_q50": math.nan,
                    "pinball_q90": math.nan,
                    "coverage_q90": math.nan,
                }
            )
    return rows


def fraction_average_metrics(
    frame: pd.DataFrame, split: str, prediction_col: str
) -> dict[str, float]:
    values = [
        regression_metrics(
            group["conferimento_t"].to_numpy(dtype=float),
            group[prediction_col].to_numpy(dtype=float),
        )
        for _, group in frame[frame["split"] == split].groupby(
            "frazione", sort=True
        )
    ]
    return {
        name: float(np.nanmean([item[name] for item in values]))
        for name in ("mae", "rmse", "wape", "r2")
    }


def quantile_metric_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    episodic = frame[frame["frazione"].isin(EPISODIC_FRACTIONS)]
    for split in ("train", "validation", "test"):
        split_frame = episodic[episodic["split"] == split]
        for fraction, group in split_frame.groupby("frazione", sort=True):
            actual = group["conferimento_t"].to_numpy(dtype=float)
            q50 = group["pred_q50_t"].to_numpy(dtype=float)
            q90 = group["pred_q90_t"].to_numpy(dtype=float)
            values = {
                "pinball_q50": float(mean_pinball_loss(actual, q50, alpha=0.5)),
                "pinball_q90": float(mean_pinball_loss(actual, q90, alpha=0.9)),
                "coverage_q90": float(np.mean(actual <= q90)),
            }
            rows.append(
                {
                    "model": "hist_quantiles",
                    "split": split,
                    "frazione": fraction,
                    "mae": math.nan,
                    "rmse": math.nan,
                    "wape": math.nan,
                    "r2": math.nan,
                    **values,
                }
            )
    return rows


def print_training_summary(
    *,
    selected: str,
    validation: dict[str, float],
    test: dict[str, float],
    hyperparameters: dict[str, object],
    fit_times: dict[str, float],
    elapsed_seconds: float,
    candidate_metrics: dict[str, dict[str, dict[str, float]]],
) -> None:
    print("\nTraining summary")
    print(f"  selected model: {selected}")
    print(f"  elapsed: {elapsed_seconds:.2f} s")
    print(
        "  validation fraction-average: "
        f"MAE={validation['mae']:.6f}, RMSE={validation['rmse']:.6f}, "
        f"WAPE={validation['wape']:.2%}, R2={validation['r2']:.6f}"
    )
    print(
        "  test fraction-average: "
        f"MAE={test['mae']:.6f}, RMSE={test['rmse']:.6f}, "
        f"WAPE={test['wape']:.2%}, R2={test['r2']:.6f}"
    )
    print("  candidate metrics:")
    for name, splits in candidate_metrics.items():
        for split, values in splits.items():
            print(
                f"    {name} {split}: "
                f"MAE={values['mae']:.6f}, RMSE={values['rmse']:.6f}, "
                f"WAPE={values['wape']:.2%}, R2={values['r2']:.6f}"
            )
    print(
        "  fit times: "
        + ", ".join(f"{name}={seconds:.2f} s" for name, seconds in fit_times.items())
    )
    print(
        "  hyperparameters: "
        + json.dumps(hyperparameters, sort_keys=True, default=str)
    )


def train_benchmark(
    input_path: Path,
    output_dir: Path,
    seed: int,
    exclude_episodic_fractions: bool = False,
    hgb_loss: str = "poisson",
    validation_strategy: str = "fixed",
    validation_folds: int = 5,
) -> dict[str, object]:
    started_at = time.perf_counter()
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"prepared withdrawals do not exist: {input_path}")
    output_dir = output_dir.resolve()
    con = duckdb.connect()
    frame = con.execute(
        "SELECT * FROM read_parquet(?) "
        "ORDER BY scenario_seed, zona_raccolta, frazione, data",
        [str(input_path)],
    ).df()
    required = {
        "scenario_seed",
        "anno_produzione",
        "data",
        "zona_raccolta",
        "frazione",
        "conferimento_t",
        *NUMERIC_FEATURES,
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"prepared withdrawals missing: {', '.join(sorted(missing))}")
    if exclude_episodic_fractions:
        frame = frame[~frame["frazione"].isin(EPISODIC_FRACTIONS)].reset_index(
            drop=True
        )
        if frame.empty:
            raise ValueError("excluding episodic fractions left no training rows")
    frame["data"] = pd.to_datetime(frame["data"])
    years = frame["anno_produzione"].unique()
    if len(years) != 1:
        raise ValueError("single-scenario benchmark requires one anno_produzione")
    year = int(years[0])
    frame = frame[frame["data"].dt.year == year].reset_index(drop=True)
    frame["split"] = chronological_split(frame, year)
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError("chronological split produced an empty partition")
    if validation_strategy not in {"fixed", "k-fold"}:
        raise ValueError(f"unsupported validation strategy: {validation_strategy}")

    validation_fold_details = []
    if validation_strategy == "fixed":
        fitted_frame, models, fit_times = fit_benchmark_models(
            frame[frame["split"] == "train"], frame, seed, hgb_loss
        )
        evaluation_frame = fitted_frame
        validation_frame = fitted_frame[fitted_frame["split"] == "validation"]
        split_strategy = "Jan-Sep train, Oct-Nov validation, Dec test"
    else:
        development = frame[frame["split"] != "test"].copy()
        validation_parts = []
        for fold, (train_index, validation_index) in enumerate(
            expanding_validation_folds(development, validation_folds), start=1
        ):
            fold_train = development.loc[train_index]
            fold_validation = development.loc[validation_index].copy()
            fold_validation["split"] = "validation"
            fold_predictions, _, _ = fit_benchmark_models(
                fold_train, fold_validation, seed, hgb_loss
            )
            fold_predictions["validation_fold"] = fold
            validation_parts.append(fold_predictions)
            validation_fold_details.append(
                {
                    "fold": fold,
                    "train_from": fold_train["data"].min().date().isoformat(),
                    "train_to": fold_train["data"].max().date().isoformat(),
                    "validation_from": (
                        fold_validation["data"].min().date().isoformat()
                    ),
                    "validation_to": (
                        fold_validation["data"].max().date().isoformat()
                    ),
                }
            )
        validation_frame = pd.concat(validation_parts, ignore_index=True)
        final_frame = frame.copy()
        final_frame.loc[final_frame["split"] != "test", "split"] = "train"
        fitted_frame, models, fit_times = fit_benchmark_models(
            development, final_frame, seed, hgb_loss
        )
        evaluation_frame = pd.concat(
            [fitted_frame, validation_frame], ignore_index=True
        )
        split_strategy = (
            f"{validation_folds}-fold expanding-window validation, Dec test"
        )

    metric_definitions = (
        ("baseline_last", "pred_last_t"),
        ("baseline_rolling4", "pred_rolling4_t"),
        ("baseline_train_mean", "pred_train_mean_t"),
        ("ridge", "pred_ridge_t"),
        ("sgd", "pred_sgd_t"),
        ("hist_gradient_boosting", "pred_hist_t"),
    )
    metrics = []
    for model, column in metric_definitions:
        metrics.extend(metric_rows(evaluation_frame, model, column))
    if "q50" in models:
        metrics.extend(quantile_metric_rows(evaluation_frame))
    metrics_frame = pd.DataFrame(metrics)

    candidate_columns = {
        "ridge": "pred_ridge_t",
        "sgd": "pred_sgd_t",
        "hist_gradient_boosting": "pred_hist_t",
    }
    candidate_metrics = {
        name: {
            "validation": fraction_average_metrics(
                validation_frame, "validation", column
            ),
            "test": fraction_average_metrics(fitted_frame, "test", column),
        }
        for name, column in candidate_columns.items()
    }
    candidate_wape = {
        name: values["validation"]["wape"]
        for name, values in candidate_metrics.items()
    }
    selected = min(candidate_wape, key=candidate_wape.get)
    selected_column = candidate_columns[selected]
    fitted_frame["selected_model"] = selected
    fitted_frame["selected_prediction_t"] = fitted_frame[selected_column]
    validation_summary = fraction_average_metrics(
        validation_frame, "validation", selected_column
    )
    test_summary = fraction_average_metrics(fitted_frame, "test", selected_column)
    selected_pipeline = models[selected]
    selected_hyperparameters = selected_pipeline.named_steps["model"].get_params(
        deep=False
    )

    model_path = output_dir / "models" / "zone_fraction_benchmark.joblib"
    metrics_path = output_dir / "metrics" / "zone_fraction_benchmark_metrics.csv"
    predictions_path = (
        output_dir / "predictions" / "zone_fraction_test_predictions.parquet"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "contract": "zone-fraction-scheduled-withdrawal-v1",
            "selected_model": selected,
            "models": models,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": "conferimento_t",
            "units": "tonnes",
            "split_strategy": split_strategy,
            "validation_strategy": validation_strategy,
            "validation_folds": validation_fold_details,
            "scenario_seed_is_feature": False,
            "excluded_episodic_fractions": exclude_episodic_fractions,
            "hgb_loss": hgb_loss,
            "seed": seed,
            "validation_macro_wape": candidate_wape,
        },
        model_path,
    )
    metrics_frame.to_csv(metrics_path, index=False)

    test_columns = [
        "scenario_seed",
        "anno_produzione",
        "data",
        "zona_raccolta",
        "frazione",
        "conferimento_t",
        "pred_last_t",
        "pred_rolling4_t",
        "pred_train_mean_t",
        "pred_ridge_t",
        "pred_sgd_t",
        "pred_hist_t",
        "pred_q50_t",
        "pred_q90_t",
        "selected_model",
        "selected_prediction_t",
    ]
    write_parquet(
        con,
        fitted_frame.loc[
            fitted_frame["split"] == "test", test_columns
        ].reset_index(drop=True),
        predictions_path,
    )
    elapsed_seconds = time.perf_counter() - started_at
    print_training_summary(
        selected=selected,
        validation=validation_summary,
        test=test_summary,
        hyperparameters=selected_hyperparameters,
        fit_times=fit_times,
        elapsed_seconds=elapsed_seconds,
        candidate_metrics=candidate_metrics,
    )
    return {
        "model": model_path,
        "metrics": metrics_path,
        "predictions": predictions_path,
        "selected_model": selected,
        "elapsed_seconds": elapsed_seconds,
        "validation_metrics": validation_summary,
        "test_metrics": test_summary,
    }


def fit_sarima(
    values: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    exog: pd.DataFrame | None = None,
):
    total_difference = order[1] + seasonal_order[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(
            np.log1p(values),
            order=order,
            seasonal_order=seasonal_order,
            exog=exog,
            trend="c" if total_difference == 0 else "n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=100)


def sarima_forecast(
    model: object,
    start: date,
    end: date,
    exog: pd.DataFrame | None = None,
) -> pd.Series:
    forecast_dates = pd.date_range(start, end, freq="D")
    forecast_args: dict[str, object] = {"steps": len(forecast_dates)}
    if exog is not None:
        forecast_args["exog"] = exog
    predicted_log = np.asarray(
        model.get_forecast(**forecast_args).predicted_mean,
        dtype=float,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        predicted = np.expm1(predicted_log)
    predicted[~np.isfinite(predicted)] = np.nan
    return pd.Series(np.maximum(predicted, 0.0), index=forecast_dates)


def apply_forecast_fallback(
    predicted: np.ndarray,
    fallback: pd.Series,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    invalid = ~np.isfinite(predicted)
    values = np.where(invalid, fallback.to_numpy(dtype=float), predicted)
    methods = np.where(invalid, "seasonal_naive_7d", model_name)
    return np.maximum(values, 0.0), methods


def daily_missing_series(
    group: pd.DataFrame, start: date, end: date
) -> pd.Series:
    daily = pd.Series(np.nan, index=pd.date_range(start, end, freq="D"))
    daily.loc[group["data"]] = group["conferimento_t"].to_numpy(dtype=float)
    return daily


def daily_sarimax_exog(
    group: pd.DataFrame, start: date, end: date
) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="D")
    intervals = (
        group.set_index("data")["giorni_dal_ritiro_precedente"]
        .reindex(index)
        .fillna(0)
        .astype(float)
    )
    holidays = set()
    for current_year in index.year.unique():
        holidays.update(italian_holidays(int(current_year)))
    dates = index.date
    return pd.DataFrame(
        {
            "giorni_dal_ritiro_precedente": intervals.to_numpy(),
            "is_holiday": [int(day in holidays) for day in dates],
            "is_after_holiday": [
                int(day - timedelta(days=1) in holidays) for day in dates
            ],
        },
        index=index,
    )


def select_sarima(
    train_daily: pd.Series,
    validation: pd.DataFrame,
    validation_start: date,
    validation_end: date,
) -> tuple[object, tuple[int, int, int], tuple[int, int, int, int], float]:
    return select_sarima_folds(
        [(train_daily, validation, validation_start, validation_end)]
    )


def select_sarima_folds(
    folds: list[tuple[pd.Series, pd.DataFrame, date, date]],
    exog: pd.DataFrame | None = None,
) -> tuple[object, tuple[int, int, int], tuple[int, int, int, int], float]:
    best = None
    for order in ARIMA_ORDERS:
        for seasonal_order in ARIMA_SEASONAL_ORDERS:
            scores = []
            model = None
            for train_daily, validation, validation_start, validation_end in folds:
                try:
                    train_exog = (
                        exog.reindex(train_daily.index) if exog is not None else None
                    )
                    forecast_exog = (
                        exog.reindex(
                            pd.date_range(validation_start, validation_end, freq="D")
                        )
                        if exog is not None
                        else None
                    )
                    model = fit_sarima(
                        train_daily, order, seasonal_order, train_exog
                    )
                    forecast = sarima_forecast(
                        model, validation_start, validation_end, forecast_exog
                    ).reindex(validation["data"])
                except (ValueError, np.linalg.LinAlgError):
                    break
                predicted = forecast.to_numpy(dtype=float)
                if not np.isfinite(predicted).all():
                    break
                scores.append(
                    regression_metrics(
                        validation["conferimento_t"].to_numpy(dtype=float),
                        predicted,
                    )["wape"]
                )
            if len(scores) != len(folds):
                continue
            score = float(np.mean(scores))
            candidate = (score, order, seasonal_order, model)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        raise ValueError("no ARIMA candidate produced finite validation forecasts")
    score, order, seasonal_order, model = best
    return model, order, seasonal_order, float(score)


def aggregate_arima_series(frame: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        frame.groupby(
            [*ARIMA_SERIES_COLUMNS, "anno_produzione", "data"],
            as_index=False,
            sort=True,
        )
        .agg(
            conferimento_t=("conferimento_t", "sum"),
            giorni_dal_ritiro_precedente=(
                "giorni_dal_ritiro_precedente",
                "first",
            ),
        )
        .sort_values([*ARIMA_SERIES_COLUMNS, "data"], kind="stable")
        .reset_index(drop=True)
    )
    aggregated["lag_7d_collection_t"] = np.nan
    for _, group in aggregated.groupby(ARIMA_SERIES_COLUMNS, sort=False):
        values_by_date = group.set_index("data")["conferimento_t"]
        aggregated.loc[group.index, "lag_7d_collection_t"] = [
            values_by_date.get(day - pd.Timedelta(days=7), np.nan)
            for day in group["data"]
        ]
    return aggregated


def relation_metrics(
    con: duckdb.DuckDBPyConnection,
    relation: str,
    actual_col: str,
    prediction_col: str,
) -> pd.DataFrame:
    rows = con.execute(f"""
        SELECT
            frazione,
            count(*)::BIGINT AS rows,
            avg(abs({sql_ident(prediction_col)} - {sql_ident(actual_col)})) AS mae,
            sqrt(avg(pow({sql_ident(prediction_col)} - {sql_ident(actual_col)}, 2)))
                AS rmse,
            CASE
                WHEN sum(abs({sql_ident(actual_col)})) > 0
                THEN sum(abs({sql_ident(prediction_col)} - {sql_ident(actual_col)}))
                     / sum(abs({sql_ident(actual_col)}))
                ELSE NULL
            END AS wape,
            CASE
                WHEN regr_syy({sql_ident(actual_col)}, {sql_ident(prediction_col)}) > 0
                THEN 1
                     - sum(pow({sql_ident(prediction_col)} - {sql_ident(actual_col)}, 2))
                       / regr_syy({sql_ident(actual_col)}, {sql_ident(prediction_col)})
                ELSE NULL
            END AS r2
        FROM {relation}
        GROUP BY frazione
        ORDER BY frazione
    """).df()
    macro = {
        "frazione": "__macro__",
        "rows": int(rows["rows"].sum()),
        **{
            name: float(rows[name].mean())
            for name in ("mae", "rmse", "wape", "r2")
        },
    }
    return pd.concat([rows, pd.DataFrame([macro])], ignore_index=True)


def civic_share_statistics(
    con: duckdb.DuckDBPyConnection,
    history_end: date,
) -> pd.DataFrame:
    quantiles = sorted(
        {
            (1.0 - level) / 2.0
            for level in CIVIC_INTERVAL_LEVELS
        }
        | {
            1.0 - (1.0 - level) / 2.0
            for level in CIVIC_INTERVAL_LEVELS
        }
    )
    quantile_sql = ",\n".join(
        f"quantile_cont(event_share, {value}) AS q_{str(value).replace('.', '_')}"
        for value in quantiles
    )
    return con.execute(f"""
        WITH events AS (
            SELECT
                id_civico,
                zona_raccolta,
                frazione,
                data,
                sum(conferimento_t) AS conferimento_t
            FROM civic_rows
            WHERE data <= DATE {sql_lit(history_end)}
            GROUP BY ALL
        ),
        shares AS (
            SELECT
                *,
                conferimento_t
                / nullif(
                    sum(conferimento_t) OVER (
                        PARTITION BY data, zona_raccolta, frazione
                    ),
                    0
                ) AS event_share
            FROM events
        ),
        weights AS (
            SELECT
                id_civico,
                zona_raccolta,
                frazione,
                sum(conferimento_t) AS allocation_weight_t
            FROM events
            GROUP BY ALL
        ),
        points AS (
            SELECT
                *,
                allocation_weight_t
                / nullif(
                    sum(allocation_weight_t) OVER (
                        PARTITION BY zona_raccolta, frazione
                    ),
                    0
                ) AS point_share
            FROM weights
        ),
        stats AS (
            SELECT
                id_civico,
                zona_raccolta,
                frazione,
                {quantile_sql},
                count(event_share) AS history_count
            FROM shares
            WHERE event_share IS NOT NULL
            GROUP BY ALL
        ),
        event_dispersion AS (
            SELECT
                s.data,
                s.zona_raccolta,
                s.frazione,
                sum(pow(s.event_share - p.point_share, 2)) AS squared_distance
            FROM shares s
            JOIN points p USING (id_civico, zona_raccolta, frazione)
            WHERE s.event_share IS NOT NULL
            GROUP BY ALL
        ),
        point_norm AS (
            SELECT
                zona_raccolta,
                frazione,
                1.0 - sum(pow(point_share, 2)) AS variance_numerator
            FROM points
            GROUP BY zona_raccolta, frazione
        ),
        mean_dispersion AS (
            SELECT
                zona_raccolta,
                frazione,
                avg(squared_distance) AS squared_distance
            FROM event_dispersion
            GROUP BY zona_raccolta, frazione
        ),
        concentration AS (
            SELECT
                n.zona_raccolta,
                n.frazione,
                greatest(
                    n.variance_numerator
                    / nullif(d.squared_distance, 0) - 1.0,
                    1.0
                ) AS dirichlet_concentration
            FROM point_norm n
            JOIN mean_dispersion d USING (zona_raccolta, frazione)
        )
        SELECT
            p.*,
            s.* EXCLUDE (id_civico, zona_raccolta, frazione),
            coalesce(c.dirichlet_concentration, 1.0)
                AS dirichlet_concentration
        FROM points p
        JOIN stats s USING (id_civico, zona_raccolta, frazione)
        JOIN concentration c USING (zona_raccolta, frazione)
        ORDER BY zona_raccolta, frazione, id_civico
    """).df()


def civic_bayesian_features(
    con: duckdb.DuckDBPyConnection,
    history_end: date,
    target_start: date | None = None,
    target_end: date | None = None,
) -> pd.DataFrame:
    history = con.execute(f"""
        WITH events AS (
            SELECT
                id_civico,
                zona_raccolta,
                frazione,
                data,
                sum(conferimento_t) AS conferimento_t
            FROM civic_rows
            WHERE data <= DATE {sql_lit(history_end)}
            GROUP BY ALL
        ),
        totals AS (
            SELECT
                data,
                zona_raccolta,
                frazione,
                sum(conferimento_t) AS group_total_t
            FROM events
            GROUP BY ALL
        ),
        shares AS (
            SELECT
                e.*,
                e.conferimento_t / nullif(t.group_total_t, 0) AS event_share
            FROM events e
            JOIN totals t USING (data, zona_raccolta, frazione)
        )
        SELECT
            id_civico,
            zona_raccolta,
            frazione,
            sum(conferimento_t) AS historical_weight_t,
            arg_max(event_share, data) AS last_share,
            count(event_share) AS history_count
        FROM shares
        WHERE event_share IS NOT NULL
        GROUP BY ALL
        ORDER BY zona_raccolta, frazione, id_civico
    """).df()
    recent = con.execute(f"""
        WITH events AS (
            SELECT
                id_civico,
                zona_raccolta,
                frazione,
                data,
                sum(conferimento_t) AS conferimento_t
            FROM civic_rows
            WHERE data <= DATE {sql_lit(history_end)}
            GROUP BY ALL
        ),
        totals AS (
            SELECT
                data,
                zona_raccolta,
                frazione,
                sum(conferimento_t) AS group_total_t
            FROM events
            GROUP BY ALL
        ),
        ranked AS (
            SELECT
                e.*,
                e.conferimento_t / nullif(t.group_total_t, 0) AS event_share,
                row_number() OVER (
                    PARTITION BY e.id_civico, e.frazione
                    ORDER BY e.data DESC
                ) AS recent_rank
            FROM events e
            JOIN totals t USING (data, zona_raccolta, frazione)
        )
        SELECT
            id_civico,
            zona_raccolta,
            frazione,
            avg(event_share) AS rolling_4_share,
            stddev_pop(event_share) AS rolling_4_share_std
        FROM ranked
        WHERE recent_rank <= 4 AND event_share IS NOT NULL
        GROUP BY ALL
    """).df()
    dimension = con.execute("""
        SELECT
            id_civico,
            any_value(coord_x) AS coord_x,
            any_value(coord_y) AS coord_y,
            any_value(densita_locale) AS densita_locale
        FROM civic_rows
        GROUP BY id_civico
    """).df()
    keys = ["id_civico", "zona_raccolta", "frazione"]
    frame = history.merge(recent, on=keys, validate="one_to_one").merge(
        dimension, on="id_civico", validate="many_to_one"
    )
    totals = frame.groupby(["zona_raccolta", "frazione"])[
        "historical_weight_t"
    ].transform("sum")
    frame["historical_share"] = frame["historical_weight_t"] / totals
    frame["log_historical_share"] = np.log(
        frame["historical_share"].clip(lower=1e-12)
    )
    frame["log_last_share"] = np.log(frame["last_share"].clip(lower=1e-12))
    frame["log_rolling_4_share"] = np.log(
        frame["rolling_4_share"].clip(lower=1e-12)
    )
    frame["rolling_4_share_std"] = frame["rolling_4_share_std"].fillna(0.0)
    frame["log_history_count"] = np.log1p(frame["history_count"])

    if target_start is not None and target_end is not None:
        target = con.execute(f"""
            SELECT
                id_civico,
                zona_raccolta,
                frazione,
                sum(conferimento_t) AS target_weight_t
            FROM civic_rows
            WHERE data BETWEEN DATE {sql_lit(target_start)}
                           AND DATE {sql_lit(target_end)}
            GROUP BY ALL
        """).df()
        target_totals = target.groupby(["zona_raccolta", "frazione"])[
            "target_weight_t"
        ].transform("sum")
        target["target_share"] = target["target_weight_t"] / target_totals
        frame = frame.merge(
            target[keys + ["target_share"]],
            on=keys,
            how="left",
            validate="one_to_one",
        )
        frame["target_share"] = frame["target_share"].fillna(0.0)
    else:
        frame["target_share"] = np.nan
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)


def build_civic_bayesian_regressor() -> Pipeline:
    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            numeric,
                            list(CIVIC_BAYES_NUMERIC_FEATURES),
                        ),
                        (
                            "categorical",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                            list(CIVIC_BAYES_CATEGORICAL_FEATURES),
                        ),
                    ],
                    sparse_threshold=0.0,
                ),
            ),
            ("model", BayesianRidge()),
        ]
    )


def fit_civic_bayesian_regressor(frame: pd.DataFrame) -> Pipeline:
    if frame.empty:
        raise ValueError("Bayesian civic allocator has no training rows")
    if len(frame) > CIVIC_BAYES_MAX_TRAIN_ROWS:
        group_columns = ["zona_raccolta", "frazione"]
        groups = frame.groupby(group_columns, sort=False).ngroups
        rows_per_group = max(1, CIVIC_BAYES_MAX_TRAIN_ROWS // groups)
        frame = (
            frame.assign(
                _sample_hash=pd.util.hash_pandas_object(
                    frame[["id_civico", *group_columns]], index=False
                )
            )
            .sort_values([*group_columns, "_sample_hash"], kind="stable")
            .groupby(group_columns, sort=False)
            .head(rows_per_group)
        )
    model = build_civic_bayesian_regressor()
    features = [
        *CIVIC_BAYES_NUMERIC_FEATURES,
        *CIVIC_BAYES_CATEGORICAL_FEATURES,
    ]
    model.fit(
        frame[features],
        np.log(np.clip(frame["target_share"].to_numpy(dtype=float), 1e-12, None)),
    )
    return model


def normalized_bayesian_shares(
    model: Pipeline,
    frame: pd.DataFrame,
    std_scale: float,
    seed: int,
    draws: int = CIVIC_POSTERIOR_DRAWS,
) -> pd.DataFrame:
    features = [
        *CIVIC_BAYES_NUMERIC_FEATURES,
        *CIVIC_BAYES_CATEGORICAL_FEATURES,
    ]
    transformed = model.named_steps["preprocess"].transform(frame[features])
    mean, std = model.named_steps["model"].predict(
        transformed, return_std=True
    )
    result_parts = []
    rng = np.random.default_rng(seed)
    group_keys = ["zona_raccolta", "frazione"]
    for _, group in frame.assign(_mean=mean, _std=std).groupby(
        group_keys, sort=False
    ):
        raw = np.exp(
            np.clip(
                rng.normal(
                    group["_mean"].to_numpy()[:, None],
                    group["_std"].to_numpy()[:, None] * std_scale,
                    size=(len(group), draws),
                ),
                -40,
                20,
            )
        )
        samples = raw / raw.sum(axis=0, keepdims=True)
        part = group[["id_civico", *group_keys]].copy()
        part["point_share"] = samples.mean(axis=1)
        part["lower_share"] = np.quantile(samples, 0.05, axis=1)
        part["upper_share"] = np.quantile(samples, 0.95, axis=1)
        result_parts.append(part)
    return pd.concat(result_parts, ignore_index=True)


def normalized_dirichlet_shares(
    statistics: pd.DataFrame,
    concentration_scale: float,
    seed: int,
    draws: int = CIVIC_POSTERIOR_DRAWS,
) -> pd.DataFrame:
    result_parts = []
    rng = np.random.default_rng(seed)
    group_keys = ["zona_raccolta", "frazione"]
    for _, group in statistics.groupby(group_keys, sort=False):
        point = group["point_share"].to_numpy(dtype=float)
        point = point / point.sum()
        concentration = float(group["dirichlet_concentration"].iloc[0])
        alpha = np.clip(point * concentration * concentration_scale, 1e-9, None)
        samples = rng.dirichlet(alpha, size=draws).T
        part = group[["id_civico", *group_keys]].copy()
        part["point_share"] = point
        part["lower_share"] = np.quantile(samples, 0.05, axis=1)
        part["upper_share"] = np.quantile(samples, 0.95, axis=1)
        result_parts.append(part)
    return pd.concat(result_parts, ignore_index=True)


def historical_share_ranges(
    statistics: pd.DataFrame,
    level: float,
) -> pd.DataFrame:
    lower = (1.0 - level) / 2.0
    upper = 1.0 - lower
    result = statistics[
        ["id_civico", "zona_raccolta", "frazione", "point_share"]
    ].copy()
    result["lower_share"] = statistics[f"q_{str(lower).replace('.', '_')}"]
    result["upper_share"] = statistics[f"q_{str(upper).replace('.', '_')}"]
    return result


def civic_allocator_metrics(
    con: duckdb.DuckDBPyConnection,
    group_predictions: pd.DataFrame,
    shares: pd.DataFrame,
    allocator: str,
    setting: float,
    split: str,
) -> pd.DataFrame:
    groups = group_predictions.copy()
    groups["data"] = pd.to_datetime(groups["data"]).dt.date
    con.register("allocator_groups", groups)
    con.register("allocator_shares", shares)
    rows = con.execute("""
        WITH actuals AS (
            SELECT
                id_civico,
                data,
                zona_raccolta,
                frazione,
                sum(conferimento_t) AS actual_t
            FROM civic_rows
            GROUP BY ALL
        ),
        allocated AS (
            SELECT
                g.frazione,
                coalesce(a.actual_t, 0.0) AS actual_t,
                g.pred_sarima_t * s.point_share AS sarima_point,
                g.pred_sarima_t * s.lower_share AS sarima_lower,
                g.pred_sarima_t * s.upper_share AS sarima_upper,
                g.pred_sarimax_t * s.point_share AS sarimax_point,
                g.pred_sarimax_t * s.lower_share AS sarimax_lower,
                g.pred_sarimax_t * s.upper_share AS sarimax_upper
            FROM allocator_groups g
            JOIN allocator_shares s USING (zona_raccolta, frazione)
            LEFT JOIN actuals a
              ON a.id_civico = s.id_civico
             AND a.data = g.data
             AND a.zona_raccolta = g.zona_raccolta
             AND a.frazione = g.frazione
        ),
        long_rows AS (
            SELECT
                frazione,
                actual_t,
                unnest(['sarima', 'sarimax']) AS forecast_model,
                unnest([sarima_point, sarimax_point]) AS point_t,
                unnest([sarima_lower, sarimax_lower]) AS lower_t,
                unnest([sarima_upper, sarimax_upper]) AS upper_t
            FROM allocated
        )
        SELECT
            forecast_model,
            frazione,
            sum(abs(point_t - actual_t)) / nullif(sum(abs(actual_t)), 0) AS wape,
            avg((actual_t BETWEEN lower_t AND upper_t)::INTEGER) AS coverage,
            avg(upper_t - lower_t) AS mean_interval_width_t
        FROM long_rows
        GROUP BY forecast_model, frazione
        ORDER BY forecast_model, frazione
    """).df()
    rows.insert(0, "split", split)
    rows.insert(0, "setting", setting)
    rows.insert(0, "allocator", allocator)
    return rows


def select_civic_allocators(metrics: pd.DataFrame) -> dict[str, dict[str, object]]:
    validation = metrics[metrics["split"] == "validation"]
    aggregate = (
        validation.groupby(["allocator", "setting", "frazione"], as_index=False)
        .agg(
            coverage=("coverage", "min"),
            wape=("wape", "mean"),
            mean_interval_width_t=("mean_interval_width_t", "mean"),
        )
    )
    selected: dict[str, dict[str, object]] = {}
    for fraction, rows in aggregate.groupby("frazione", sort=True):
        eligible = rows[rows["coverage"] >= 0.90]
        if eligible.empty:
            chosen = rows.sort_values(
                ["coverage", "wape", "mean_interval_width_t"],
                ascending=[False, True, True],
            ).iloc[0]
        else:
            chosen = eligible.sort_values(
                ["wape", "mean_interval_width_t", "coverage"],
                ascending=[True, True, False],
            ).iloc[0]
        selected[str(fraction)] = {
            "allocator": str(chosen["allocator"]),
            "setting": float(chosen["setting"]),
            "validation_coverage": float(chosen["coverage"]),
            "validation_wape": float(chosen["wape"]),
        }
    return selected


def create_civic_arima_evaluation(
    civic_input_path: Path,
    group_predictions: pd.DataFrame,
    test_start: date,
    test_end: date,
    output_dir: Path,
    year: int,
) -> tuple[Path, Path, pd.DataFrame, Path, dict[str, object]]:
    civic_input_path = civic_input_path.resolve()
    if not civic_input_path.is_file():
        raise FileNotFoundError(
            f"classified civic collections do not exist: {civic_input_path}"
        )

    required_group_columns = {
        "scenario_seed",
        "anno_produzione",
        "data",
        "zona_raccolta",
        "frazione",
        "conferimento_t",
        "pred_sarima_t",
        "pred_sarimax_t",
        "pred_seasonal_naive_7d_t",
    }
    if missing := required_group_columns - set(group_predictions.columns):
        raise ValueError(
            "group predictions missing: " + ", ".join(sorted(missing))
        )

    con = duckdb.connect()
    con.execute("SET threads = 2")
    con.execute("SET memory_limit = '1GB'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(
        f"CREATE VIEW civic_raw AS "
        f"SELECT * FROM read_parquet({sql_lit(civic_input_path)})"
    )
    civic_columns = table_schema(con, "civic_raw")
    require_columns(
        civic_columns,
        {
            "anno_produzione",
            "data",
            "id_civico",
            "zona_raccolta",
            "frazione",
            "cluster_label",
            "conferimento_t",
        },
        "classified collections",
    )
    optional_numeric = {
        column: (
            f"try_cast({sql_ident(column)} AS DOUBLE)"
            if column in civic_columns
            else "NULL::DOUBLE"
        )
        for column in ("coord_x", "coord_y", "densita_locale")
    }
    con.execute(f"""
        CREATE TEMP VIEW civic_rows AS
        SELECT
            cast(id_civico AS VARCHAR) AS id_civico,
            try_cast(anno_produzione AS INTEGER) AS anno_produzione,
            try_cast(data AS DATE) AS data,
            trim(cast(zona_raccolta AS VARCHAR)) AS zona_raccolta,
            trim(cast(frazione AS VARCHAR)) AS frazione,
            try_cast(cluster_label AS INTEGER) AS cluster_label,
            try_cast(conferimento_t AS DOUBLE) AS conferimento_t,
            {optional_numeric["coord_x"]} AS coord_x,
            {optional_numeric["coord_y"]} AS coord_y,
            {optional_numeric["densita_locale"]} AS densita_locale
        FROM civic_raw
        WHERE try_cast(anno_produzione AS INTEGER) = {year}
    """)
    check_zero(
        con,
        """
        SELECT count(*)
        FROM (
            SELECT id_civico
            FROM civic_rows
            GROUP BY id_civico
            HAVING count(DISTINCT struct_pack(
                zone := zona_raccolta,
                cluster := cluster_label
            )) <> 1
        )
        """,
        "each civic must have one zone/cluster mapping",
    )
    check_zero(
        con,
        """
        SELECT count(*) FROM civic_rows
        WHERE id_civico IS NULL
           OR zona_raccolta IS NULL
           OR cluster_label IS NULL
           OR data IS NULL
           OR conferimento_t IS NULL
           OR conferimento_t < 0
           OR NOT isfinite(conferimento_t)
        """,
        "classified civic collections",
    )

    all_groups = group_predictions.copy()
    all_groups["data"] = pd.to_datetime(all_groups["data"]).dt.date
    if "split" in all_groups:
        validation_groups = all_groups[
            (all_groups["split"] == "validation")
            & (pd.to_datetime(all_groups["data"]).dt.month.isin([10, 11]))
        ].drop(columns=["split"])
        test_groups = all_groups[all_groups["split"] == "test"].drop(
            columns=["split"]
        )
    else:
        validation_groups = all_groups.iloc[0:0].copy()
        test_groups = all_groups
    if test_groups.empty:
        raise ValueError("civic evaluation has no test group predictions")

    active_fractions = sorted(test_groups["frazione"].unique())
    allocator_metric_rows: list[pd.DataFrame] = []
    selected: dict[str, dict[str, object]] = {
        fraction: {
            "allocator": "historical",
            "setting": 0.90,
            "validation_coverage": float("nan"),
            "validation_wape": float("nan"),
        }
        for fraction in active_fractions
    }

    if not validation_groups.empty:
        validation_statistics = civic_share_statistics(
            con, date(year, 9, 30)
        )
        validation_statistics = validation_statistics[
            validation_statistics["frazione"].isin(active_fractions)
        ]
        for level in CIVIC_INTERVAL_LEVELS:
            shares = historical_share_ranges(validation_statistics, level)
            allocator_metric_rows.append(
                civic_allocator_metrics(
                    con,
                    validation_groups,
                    shares,
                    "historical",
                    level,
                    "validation",
                )
            )
        for scale in CIVIC_DIRICHLET_SCALES:
            shares = normalized_dirichlet_shares(
                validation_statistics, scale, seed=year
            )
            allocator_metric_rows.append(
                civic_allocator_metrics(
                    con,
                    validation_groups,
                    shares,
                    "dirichlet",
                    scale,
                    "validation",
                )
            )

        bayes_training = civic_bayesian_features(
            con,
            date(year, 6, 30),
            date(year, 7, 1),
            date(year, 9, 30),
        )
        bayes_validation = civic_bayesian_features(
            con, date(year, 9, 30)
        )
        bayes_training = bayes_training[
            bayes_training["frazione"].isin(active_fractions)
        ]
        bayes_validation = bayes_validation[
            bayes_validation["frazione"].isin(active_fractions)
        ]
        if not bayes_training.empty and not bayes_validation.empty:
            validation_model = fit_civic_bayesian_regressor(bayes_training)
            for scale in CIVIC_BAYES_STD_SCALES:
                shares = normalized_bayesian_shares(
                    validation_model,
                    bayes_validation,
                    scale,
                    seed=year,
                )
                allocator_metric_rows.append(
                    civic_allocator_metrics(
                        con,
                        validation_groups,
                        shares,
                        "bayesian_ridge",
                        scale,
                        "validation",
                    )
                )
        if allocator_metric_rows:
            selected = select_civic_allocators(
                pd.concat(allocator_metric_rows, ignore_index=True)
            )

    final_statistics = civic_share_statistics(con, date(year, 11, 30))
    final_statistics = final_statistics[
        final_statistics["frazione"].isin(active_fractions)
    ]
    selected_parts = []
    final_models: dict[str, object] = {}
    for allocator in ("historical", "dirichlet", "bayesian_ridge"):
        choices = {
            float(choice["setting"])
            for choice in selected.values()
            if choice["allocator"] == allocator
        }
        if not choices:
            continue
        if allocator == "bayesian_ridge":
            bayes_final_training = civic_bayesian_features(
                con,
                date(year, 9, 30),
                date(year, 10, 1),
                date(year, 11, 30),
            )
            bayes_test = civic_bayesian_features(con, date(year, 11, 30))
            bayes_final_training = bayes_final_training[
                bayes_final_training["frazione"].isin(active_fractions)
            ]
            bayes_test = bayes_test[
                bayes_test["frazione"].isin(active_fractions)
            ]
            final_model = fit_civic_bayesian_regressor(bayes_final_training)
            final_models["bayesian_ridge"] = final_model
        for setting in choices:
            if allocator == "historical":
                shares = historical_share_ranges(final_statistics, setting)
            elif allocator == "dirichlet":
                shares = normalized_dirichlet_shares(
                    final_statistics, setting, seed=year + 1
                )
            else:
                shares = normalized_bayesian_shares(
                    final_model,
                    bayes_test,
                    setting,
                    seed=year + 1,
                )
            fractions = [
                fraction
                for fraction, choice in selected.items()
                if choice["allocator"] == allocator
                and float(choice["setting"]) == setting
            ]
            shares = shares[shares["frazione"].isin(fractions)].copy()
            shares["allocation_method"] = allocator
            shares["allocation_setting"] = setting
            selected_parts.append(shares)
    selected_shares = pd.concat(selected_parts, ignore_index=True)
    selected_shares["lower_share"] = np.minimum(
        selected_shares["lower_share"], selected_shares["point_share"]
    )
    selected_shares["upper_share"] = np.maximum(
        selected_shares["upper_share"], selected_shares["point_share"]
    )
    selected_shares = selected_shares.merge(
        final_statistics[
            [
                "id_civico",
                "zona_raccolta",
                "frazione",
                "allocation_weight_t",
            ]
        ],
        on=["id_civico", "zona_raccolta", "frazione"],
        how="left",
        validate="one_to_one",
    )

    test_groups["data"] = pd.to_datetime(test_groups["data"]).dt.date
    con.register("test_group_predictions", test_groups)
    con.register("selected_allocator_shares", selected_shares)
    con.execute(f"""
        CREATE TEMP TABLE civic_test_predictions AS
        WITH civic_dimension AS (
            SELECT
                id_civico,
                any_value(zona_raccolta) AS zona_raccolta,
                any_value(cluster_label) AS cluster_label
            FROM civic_rows
            GROUP BY id_civico
        ),
        test_actuals AS (
            SELECT
                id_civico,
                data,
                frazione,
                sum(conferimento_t) AS actual_conferimento_t
            FROM civic_rows
            WHERE data BETWEEN DATE {sql_lit(test_start)} AND DATE {sql_lit(test_end)}
            GROUP BY id_civico, data, frazione
        ),
        eligible AS (
            SELECT
                g.*,
                d.id_civico,
                d.cluster_label AS civic_cluster_label,
                greatest(coalesce(s.allocation_weight_t, 0.0), 0.0)
                    AS allocation_weight_t,
                s.allocation_method,
                s.allocation_setting,
                coalesce(s.point_share, 0.0) AS raw_point_share,
                coalesce(s.lower_share, 0.0) AS raw_lower_share,
                coalesce(s.upper_share, 0.0) AS raw_upper_share,
                coalesce(a.actual_conferimento_t, 0.0) AS actual_conferimento_t
            FROM test_group_predictions g
            JOIN civic_dimension d
              ON g.zona_raccolta = d.zona_raccolta
            LEFT JOIN selected_allocator_shares s
              ON d.id_civico = s.id_civico
             AND g.zona_raccolta = s.zona_raccolta
             AND g.frazione = s.frazione
            LEFT JOIN test_actuals a
              ON d.id_civico = a.id_civico
             AND g.data = a.data
             AND g.frazione = a.frazione
        ),
        weighted AS (
            SELECT
                *,
                sum(raw_point_share) OVER group_day AS total_point_share,
                count(*) OVER group_day AS eligible_civics
            FROM eligible
            WINDOW group_day AS (
                PARTITION BY
                    scenario_seed,
                    data,
                    zona_raccolta,
                    frazione
            )
        ),
        allocated AS (
            SELECT
                *,
                CASE
                    WHEN total_point_share > 0
                    THEN raw_point_share / total_point_share
                    ELSE 1.0 / eligible_civics
                END AS allocation_share,
                CASE
                    WHEN total_point_share > 0 THEN raw_lower_share
                    ELSE 1.0 / eligible_civics
                END AS allocation_share_lower,
                CASE
                    WHEN total_point_share > 0 THEN raw_upper_share
                    ELSE 1.0 / eligible_civics
                END AS allocation_share_upper
            FROM weighted
        )
        SELECT
            scenario_seed,
            anno_produzione,
            data,
            zona_raccolta,
            frazione,
            civic_cluster_label AS cluster_label,
            id_civico,
            allocation_weight_t,
            coalesce(allocation_method, 'equal') AS allocation_method,
            allocation_setting,
            allocation_share,
            allocation_share_lower,
            allocation_share_upper,
            actual_conferimento_t,
            conferimento_t AS actual_group_total_t,
            pred_sarima_t AS pred_group_sarima_t,
            pred_sarimax_t AS pred_group_sarimax_t,
            pred_seasonal_naive_7d_t AS pred_group_seasonal_naive_7d_t,
            pred_sarima_t * allocation_share AS pred_civic_sarima_t,
            pred_sarima_t * allocation_share_lower
                AS pred_civic_sarima_lower_t,
            pred_sarima_t * allocation_share_upper
                AS pred_civic_sarima_upper_t,
            pred_sarimax_t * allocation_share AS pred_civic_sarimax_t,
            pred_sarimax_t * allocation_share_lower
                AS pred_civic_sarimax_lower_t,
            pred_sarimax_t * allocation_share_upper
                AS pred_civic_sarimax_upper_t,
            pred_seasonal_naive_7d_t * allocation_share
                AS pred_civic_seasonal_naive_7d_t,
            pred_seasonal_naive_7d_t * allocation_share_lower
                AS pred_civic_seasonal_naive_7d_lower_t,
            pred_seasonal_naive_7d_t * allocation_share_upper
                AS pred_civic_seasonal_naive_7d_upper_t
        FROM allocated
    """)
    check_zero(
        con,
        """
        SELECT count(*)
        FROM test_group_predictions g
        LEFT JOIN (
            SELECT DISTINCT
                scenario_seed, data, zona_raccolta, frazione
            FROM civic_test_predictions
        ) c
        USING (scenario_seed, data, zona_raccolta, frazione)
        WHERE c.data IS NULL
        """,
        "every ARIMA group must contain known civics",
    )
    for civic_col, group_col in (
        ("actual_conferimento_t", "actual_group_total_t"),
        ("pred_civic_sarima_t", "pred_group_sarima_t"),
        ("pred_civic_sarimax_t", "pred_group_sarimax_t"),
        (
            "pred_civic_seasonal_naive_7d_t",
            "pred_group_seasonal_naive_7d_t",
        ),
    ):
        check_zero(
            con,
            f"""
            SELECT count(*)
            FROM (
                SELECT
                    sum({sql_ident(civic_col)}) AS civic_total,
                    any_value({sql_ident(group_col)}) AS group_total
                FROM civic_test_predictions
                GROUP BY
                    scenario_seed, data, zona_raccolta, frazione
            )
            WHERE abs(civic_total - group_total)
                  > greatest(
                      1e-10,
                      1e-9 * greatest(abs(civic_total), abs(group_total))
                  )
            """,
            f"civic disaggregation must preserve {group_col}",
        )

    predictions_path = (
        output_dir
        / "predictions"
        / "zone_fraction_arima_test_predictions_per_civic.parquet"
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT
                *,
                pred_civic_sarima_t - actual_conferimento_t AS sarima_error,
                abs(pred_civic_sarima_t - actual_conferimento_t)
                    AS sarima_absolute_error,
                pred_civic_sarimax_t - actual_conferimento_t AS sarimax_error,
                abs(pred_civic_sarimax_t - actual_conferimento_t)
                    AS sarimax_absolute_error,
                pred_civic_seasonal_naive_7d_t - actual_conferimento_t
                    AS seasonal_naive_7d_error,
                abs(pred_civic_seasonal_naive_7d_t - actual_conferimento_t)
                    AS seasonal_naive_7d_absolute_error
            FROM civic_test_predictions
            ORDER BY data, zona_raccolta, frazione, cluster_label, id_civico
        )
        TO {sql_lit(predictions_path)}
        (FORMAT parquet, COMPRESSION zstd)
    """)

    con.register("group_metric_rows", test_groups)
    model_columns = {
        "sarima": (
            "pred_sarima_t",
            "pred_civic_sarima_t",
            "pred_civic_sarima_lower_t",
            "pred_civic_sarima_upper_t",
        ),
        "sarimax": (
            "pred_sarimax_t",
            "pred_civic_sarimax_t",
            "pred_civic_sarimax_lower_t",
            "pred_civic_sarimax_upper_t",
        ),
        "seasonal_naive_7d": (
            "pred_seasonal_naive_7d_t",
            "pred_civic_seasonal_naive_7d_t",
            "pred_civic_seasonal_naive_7d_lower_t",
            "pred_civic_seasonal_naive_7d_upper_t",
        ),
    }
    comparison_rows = []
    for model, (group_col, civic_col, lower_col, upper_col) in model_columns.items():
        group_metrics = relation_metrics(
            con, "group_metric_rows", "conferimento_t", group_col
        )
        civic_metrics = relation_metrics(
            con, "civic_test_predictions", "actual_conferimento_t", civic_col
        )
        merged = group_metrics.merge(
            civic_metrics,
            on="frazione",
            suffixes=("_group", "_civic"),
            validate="one_to_one",
        )
        merged.insert(0, "model", model)
        merged["wape_delta_pp"] = (
            merged["wape_civic"] - merged["wape_group"]
        ) * 100.0
        merged["wape_relative_change"] = np.where(
            merged["wape_group"] > 0,
            merged["wape_civic"] / merged["wape_group"] - 1.0,
            np.nan,
        )
        interval_rows = con.execute(f"""
            SELECT
                frazione,
                avg((
                    actual_conferimento_t BETWEEN
                    {sql_ident(lower_col)} AND {sql_ident(upper_col)}
                )::INTEGER) AS interval_coverage,
                avg({sql_ident(upper_col)} - {sql_ident(lower_col)})
                    AS mean_interval_width_t
            FROM civic_test_predictions
            GROUP BY frazione
            ORDER BY frazione
        """).df()
        interval_macro = pd.DataFrame(
            [
                {
                    "frazione": "__macro__",
                    "interval_coverage": float(
                        interval_rows["interval_coverage"].mean()
                    ),
                    "mean_interval_width_t": float(
                        interval_rows["mean_interval_width_t"].mean()
                    ),
                }
            ]
        )
        merged = merged.merge(
            pd.concat([interval_rows, interval_macro], ignore_index=True),
            on="frazione",
            validate="one_to_one",
        )
        comparison_rows.append(merged)
    comparison = pd.concat(comparison_rows, ignore_index=True)
    metrics_path = (
        output_dir / "metrics" / "zone_fraction_arima_civic_metrics.csv"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(metrics_path, index=False)
    allocator_metric_rows.append(
        civic_allocator_metrics(
            con,
            test_groups,
            selected_shares[
                [
                    "id_civico",
                    "zona_raccolta",
                    "frazione",
                    "point_share",
                    "lower_share",
                    "upper_share",
                ]
            ],
            "selected",
            0.90,
            "test",
        )
    )
    allocator_metrics = pd.concat(allocator_metric_rows, ignore_index=True)
    allocator_metrics_path = (
        output_dir / "metrics" / "zone_fraction_arima_allocator_benchmark.csv"
    )
    allocator_metrics.to_csv(allocator_metrics_path, index=False)
    allocator_bundle = {
        "selection_rule": "coverage>=90%, then WAPE, then interval width",
        "target_coverage": 0.90,
        "posterior_draws": CIVIC_POSTERIOR_DRAWS,
        "selected_by_fraction": selected,
        "models": final_models,
    }
    return (
        predictions_path,
        metrics_path,
        comparison,
        allocator_metrics_path,
        allocator_bundle,
    )


def train_arima_benchmark(
    input_path: Path,
    output_dir: Path,
    exclude_episodic_fractions: bool = False,
    validation_strategy: str = "fixed",
    validation_folds: int = 5,
    civic_input_path: Path | None = None,
) -> dict[str, object]:
    started_at = time.perf_counter()
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"prepared withdrawals do not exist: {input_path}")
    output_dir = output_dir.resolve()
    con = duckdb.connect()
    frame = con.execute(
        "SELECT * FROM read_parquet(?) "
        "ORDER BY scenario_seed, zona_raccolta, frazione, data",
        [str(input_path)],
    ).df()
    required = {
        "scenario_seed",
        "anno_produzione",
        "data",
        "zona_raccolta",
        "frazione",
        "conferimento_t",
        "giorni_dal_ritiro_precedente",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"prepared withdrawals missing: {', '.join(sorted(missing))}")
    if exclude_episodic_fractions:
        frame = frame[~frame["frazione"].isin(EPISODIC_FRACTIONS)].reset_index(
            drop=True
        )
    if frame.empty:
        raise ValueError("ARIMA benchmark has no rows")
    frame["data"] = pd.to_datetime(frame["data"])
    years = frame["anno_produzione"].unique()
    if len(years) != 1:
        raise ValueError("ARIMA benchmark requires one anno_produzione")
    year = int(years[0])
    frame = frame[frame["data"].dt.year == year].reset_index(drop=True)
    frame = aggregate_arima_series(frame)
    frame["split"] = chronological_split(frame, year)
    if validation_strategy not in {"fixed", "k-fold"}:
        raise ValueError(f"unsupported validation strategy: {validation_strategy}")
    if validation_strategy == "k-fold" and validation_folds < 2:
        raise ValueError("validation_folds must be at least 2")

    train_end = date(year, 9, 30)
    validation_start = date(year, 10, 1)
    validation_end = date(year, 11, 30)
    test_start = date(year, 12, 1)
    test_end = frame["data"].max().date()
    series_start = date(year, 1, 1)
    train_end_timestamp = pd.Timestamp(train_end)
    validation_end_timestamp = pd.Timestamp(validation_end)
    split_strategy = (
        "Jan-Sep train, Oct-Nov validation, Dec test"
        if validation_strategy == "fixed"
        else f"{validation_folds}-fold expanding-window validation, Dec test"
    )
    result_rows = []
    saved_models = {}
    saved_sarimax_models = {}
    selected_orders = Counter()
    selected_sarimax_orders = Counter()
    fallback_rows = Counter()
    groups = list(frame.groupby(ARIMA_SERIES_COLUMNS, sort=True))
    for index, (group_key, group) in enumerate(groups, start=1):
        group = group.sort_values("data", kind="stable").copy()
        daily = daily_missing_series(group, series_start, test_end)
        daily_exog = daily_sarimax_exog(group, series_start, test_end)
        test = group[group["split"] == "test"]
        if test.empty:
            raise ValueError(f"ARIMA group has an empty split: {group_key}")
        fold_details = []
        if validation_strategy == "fixed":
            validation = group[group["split"] == "validation"]
            if validation.empty:
                raise ValueError(f"ARIMA group has an empty split: {group_key}")
            fold_specs = [
                (
                    daily.loc[:train_end_timestamp],
                    validation,
                    validation_start,
                    validation_end,
                    float(
                        group.loc[
                            group["split"] == "train", "conferimento_t"
                        ].mean()
                    ),
                )
            ]
        else:
            development = group[group["split"] != "test"]
            fold_specs = []
            for fold, (train_index, validation_index) in enumerate(
                expanding_validation_folds(development, validation_folds), start=1
            ):
                fold_train = development.loc[train_index]
                fold_validation = development.loc[validation_index].copy()
                fold_validation["split"] = "validation"
                fold_start = fold_validation["data"].min().date()
                fold_end = fold_validation["data"].max().date()
                fold_specs.append(
                    (
                        daily.loc[
                            : pd.Timestamp(fold_start) - pd.Timedelta(days=1)
                        ],
                        fold_validation,
                        fold_start,
                        fold_end,
                        float(fold_train["conferimento_t"].mean()),
                    )
                )
                fold_details.append(
                    {
                        "fold": fold,
                        "train_from": (
                            fold_train["data"].min().date().isoformat()
                        ),
                        "train_to": fold_train["data"].max().date().isoformat(),
                        "validation_from": fold_start.isoformat(),
                        "validation_to": fold_end.isoformat(),
                    }
                )

        _, order, seasonal_order, validation_wape = select_sarima_folds(
            [spec[:4] for spec in fold_specs]
        )
        _, sarimax_order, sarimax_seasonal_order, sarimax_validation_wape = (
            select_sarima_folds([spec[:4] for spec in fold_specs], daily_exog)
        )
        evaluations = []
        sarimax_evaluations = []
        for train_daily, validation, fold_start, fold_end, train_mean in fold_specs:
            validation_model = fit_sarima(
                train_daily, order, seasonal_order
            )
            validation_prediction = sarima_forecast(
                validation_model, fold_start, fold_end
            ).reindex(validation["data"]).to_numpy(dtype=float)
            evaluations.append((validation, validation_prediction, train_mean))
            sarimax_validation_model = fit_sarima(
                train_daily,
                sarimax_order,
                sarimax_seasonal_order,
                daily_exog.reindex(train_daily.index),
            )
            sarimax_validation_prediction = sarima_forecast(
                sarimax_validation_model,
                fold_start,
                fold_end,
                daily_exog.reindex(pd.date_range(fold_start, fold_end, freq="D")),
            ).reindex(validation["data"]).to_numpy(dtype=float)
            sarimax_evaluations.append(
                (validation, sarimax_validation_prediction, train_mean)
            )

        test_model = fit_sarima(
            daily.loc[:validation_end_timestamp], order, seasonal_order
        )
        test_prediction = sarima_forecast(
            test_model, test_start, test_end
        ).reindex(test["data"]).to_numpy(dtype=float)
        sarimax_test_model = fit_sarima(
            daily.loc[:validation_end_timestamp],
            sarimax_order,
            sarimax_seasonal_order,
            daily_exog.loc[:validation_end_timestamp],
        )
        sarimax_test_prediction = sarima_forecast(
            sarimax_test_model,
            test_start,
            test_end,
            daily_exog.reindex(pd.date_range(test_start, test_end, freq="D")),
        ).reindex(test["data"]).to_numpy(dtype=float)
        evaluations.append(
            (
                test,
                test_prediction,
                float(
                    group.loc[
                        group["split"] != "test", "conferimento_t"
                    ].mean()
                ),
            )
        )
        sarimax_evaluations.append(
            (
                test,
                sarimax_test_prediction,
                float(
                    group.loc[
                        group["split"] != "test", "conferimento_t"
                    ].mean()
                ),
            )
        )
        final_model = fit_sarima(daily, order, seasonal_order)
        final_sarimax_model = fit_sarima(
            daily, sarimax_order, sarimax_seasonal_order, daily_exog
        )
        model_key = (
            f"scenario_seed={group_key[0]}|zona_raccolta={group_key[1]}|"
            f"frazione={group_key[2]}"
        )
        saved_models[model_key] = {
            "model": final_model,
            "order": order,
            "seasonal_order": seasonal_order,
            "validation_wape": validation_wape,
            "validation_folds": fold_details,
        }
        selected_orders[f"{order}x{seasonal_order}"] += 1
        saved_sarimax_models[model_key] = {
            "model": final_sarimax_model,
            "order": sarimax_order,
            "seasonal_order": sarimax_seasonal_order,
            "validation_wape": sarimax_validation_wape,
            "validation_folds": fold_details,
        }
        selected_sarimax_orders[
            f"{sarimax_order}x{sarimax_seasonal_order}"
        ] += 1

        for (split_frame, prediction, train_mean), (_, sarimax_prediction, _) in zip(
            evaluations, sarimax_evaluations, strict=True
        ):
            split_rows = split_frame[
                [
                    "scenario_seed",
                    "anno_produzione",
                    "data",
                    "zona_raccolta",
                    "frazione",
                    "conferimento_t",
                    "lag_7d_collection_t",
                    "split",
                ]
            ].copy()
            split_rows["pred_seasonal_naive_7d_t"] = (
                split_rows["lag_7d_collection_t"].fillna(train_mean).clip(lower=0)
            )
            split_rows["pred_sarima_t"], split_rows["sarima_forecast_method"] = (
                apply_forecast_fallback(
                    prediction,
                    split_rows["pred_seasonal_naive_7d_t"],
                    "sarima",
                )
            )
            (
                split_rows["pred_sarimax_t"],
                split_rows["sarimax_forecast_method"],
            ) = apply_forecast_fallback(
                sarimax_prediction,
                split_rows["pred_seasonal_naive_7d_t"],
                "sarimax",
            )
            fallback_rows["sarima"] += int(
                (split_rows["sarima_forecast_method"] == "seasonal_naive_7d").sum()
            )
            fallback_rows["sarimax"] += int(
                (split_rows["sarimax_forecast_method"] == "seasonal_naive_7d").sum()
            )
            result_rows.append(split_rows)
        print(
            f"[{index}/{len(groups)}] {model_key}: "
            f"order={order}, seasonal_order={seasonal_order}, "
            f"validation_wape={validation_wape:.2%}; "
            f"sarimax_order={sarimax_order}, "
            f"sarimax_seasonal_order={sarimax_seasonal_order}, "
            f"sarimax_validation_wape={sarimax_validation_wape:.2%}"
        )

    predictions = pd.concat(result_rows, ignore_index=True)
    metrics = pd.DataFrame(
        [
            *metric_rows(predictions, "sarima", "pred_sarima_t"),
            *metric_rows(predictions, "sarimax", "pred_sarimax_t"),
            *metric_rows(
                predictions,
                "seasonal_naive_7d",
                "pred_seasonal_naive_7d_t",
            ),
        ]
    )
    validation_summary = fraction_average_metrics(
        predictions, "validation", "pred_sarima_t"
    )
    test_summary = fraction_average_metrics(predictions, "test", "pred_sarima_t")
    sarimax_validation_summary = fraction_average_metrics(
        predictions, "validation", "pred_sarimax_t"
    )
    sarimax_test_summary = fraction_average_metrics(
        predictions, "test", "pred_sarimax_t"
    )
    naive_test_summary = fraction_average_metrics(
        predictions, "test", "pred_seasonal_naive_7d_t"
    )

    model_path = output_dir / "models" / "zone_fraction_arima.joblib"
    metrics_path = output_dir / "metrics" / "zone_fraction_arima_metrics.csv"
    predictions_path = (
        output_dir / "predictions" / "zone_fraction_arima_test_predictions.parquet"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "contract": "zone-fraction-scheduled-withdrawal-v1",
        "model_name": "zone_fraction_sarima",
        "model_variants": ("sarima", "sarimax"),
        "group_columns": ARIMA_SERIES_COLUMNS[1:],
        "models": saved_models,
        "sarimax_models": saved_sarimax_models,
        "sarimax_exog_columns": SARIMAX_EXOG_COLUMNS,
        "validation_metrics": validation_summary,
        "test_metrics": test_summary,
        "sarimax_validation_metrics": sarimax_validation_summary,
        "sarimax_test_metrics": sarimax_test_summary,
        "target_transform": "log1p",
        "calendar_frequency": "D",
        "unscheduled_days": "missing",
        "selection_metric": (
            "validation WAPE"
            if validation_strategy == "fixed"
            else "mean validation-fold WAPE"
        ),
        "split_strategy": split_strategy,
        "validation_strategy": validation_strategy,
        "validation_fold_count": (
            validation_folds if validation_strategy == "k-fold" else 1
        ),
        "excluded_episodic_fractions": exclude_episodic_fractions,
        "candidate_orders": ARIMA_ORDERS,
        "candidate_seasonal_orders": ARIMA_SEASONAL_ORDERS,
        "fallback_rows": dict(fallback_rows),
        "civic_allocation": (
            "validation_selected_bayesian_share" if civic_input_path else None
        ),
    }
    metrics.to_csv(metrics_path, index=False)
    test_predictions = predictions[predictions["split"] == "test"].drop(
        columns=["split"]
    )
    write_parquet(con, test_predictions, predictions_path)

    civic_predictions_path = None
    civic_metrics_path = None
    civic_allocator_metrics_path = None
    civic_comparison = None
    if civic_input_path is not None:
        (
            civic_predictions_path,
            civic_metrics_path,
            civic_comparison,
            civic_allocator_metrics_path,
            civic_allocator_bundle,
        ) = create_civic_arima_evaluation(
            civic_input_path,
            predictions,
            test_start,
            test_end,
            output_dir,
            year,
        )
        model_bundle["civic_allocator"] = civic_allocator_bundle
    joblib.dump(model_bundle, model_path)

    elapsed_seconds = time.perf_counter() - started_at
    print("\nARIMA summary")
    print(f"  elapsed: {elapsed_seconds:.2f} s")
    print(
        "  validation fraction-average: "
        f"MAE={validation_summary['mae']:.6f}, "
        f"RMSE={validation_summary['rmse']:.6f}, "
        f"WAPE={validation_summary['wape']:.2%}, "
        f"R2={validation_summary['r2']:.6f}"
    )
    print(
        "  test fraction-average: "
        f"MAE={test_summary['mae']:.6f}, RMSE={test_summary['rmse']:.6f}, "
        f"WAPE={test_summary['wape']:.2%}, R2={test_summary['r2']:.6f}"
    )
    print(
        "  SARIMAX validation fraction-average: "
        f"MAE={sarimax_validation_summary['mae']:.6f}, "
        f"RMSE={sarimax_validation_summary['rmse']:.6f}, "
        f"WAPE={sarimax_validation_summary['wape']:.2%}, "
        f"R2={sarimax_validation_summary['r2']:.6f}"
    )
    print(
        "  SARIMAX test fraction-average: "
        f"MAE={sarimax_test_summary['mae']:.6f}, "
        f"RMSE={sarimax_test_summary['rmse']:.6f}, "
        f"WAPE={sarimax_test_summary['wape']:.2%}, "
        f"R2={sarimax_test_summary['r2']:.6f}"
    )
    print(
        "  seasonal-naive test fraction-average: "
        f"WAPE={naive_test_summary['wape']:.2%}, "
        f"R2={naive_test_summary['r2']:.6f}"
    )
    print(
        "  selected orders: "
        + json.dumps(dict(sorted(selected_orders.items())), sort_keys=True)
    )
    print(
        "  selected SARIMAX orders: "
        + json.dumps(dict(sorted(selected_sarimax_orders.items())), sort_keys=True)
    )
    print(f"  seasonal-naive fallback rows: {dict(fallback_rows)}")
    if civic_comparison is not None:
        print("  civic disaggregation (validation-selected allocator):")
        for row in civic_comparison[
            civic_comparison["frazione"] == "__macro__"
        ].itertuples(index=False):
            relative = (
                f"{row.wape_relative_change:+.2%}"
                if np.isfinite(row.wape_relative_change)
                else "undefined"
            )
            print(
                f"    {row.model}: group WAPE={row.wape_group:.2%}, "
                f"civic WAPE={row.wape_civic:.2%}, "
                f"change={row.wape_delta_pp:+.2f} pp ({relative})"
            )
    return {
        "model": model_path,
        "metrics": metrics_path,
        "predictions": predictions_path,
        "elapsed_seconds": elapsed_seconds,
        "validation_metrics": validation_summary,
        "test_metrics": test_summary,
        "sarimax_validation_metrics": sarimax_validation_summary,
        "sarimax_test_metrics": sarimax_test_summary,
        "civic_predictions": civic_predictions_path,
        "civic_metrics": civic_metrics_path,
        "civic_allocator_metrics": civic_allocator_metrics_path,
    }


def select_future_models(metrics: pd.DataFrame) -> dict[str, str]:
    required = {"model", "split", "frazione", "wape", "rmse"}
    if missing := required - set(metrics.columns):
        raise ValueError(
            "ARIMA metrics missing columns: " + ", ".join(sorted(missing))
        )
    candidates = metrics[
        (metrics["split"] == "validation")
        & metrics["model"].isin(
            [
                "baseline_last",
                "baseline_rolling4",
                "baseline_train_mean",
                "ridge",
                "sgd",
                "hist_gradient_boosting",
                "sarima",
                "sarimax",
                "seasonal_naive_7d",
            ]
        )
        & (metrics["frazione"] != "__macro__")
    ].copy()
    candidates["wape"] = pd.to_numeric(candidates["wape"], errors="coerce")
    candidates["rmse"] = pd.to_numeric(candidates["rmse"], errors="coerce")
    candidates = candidates[
        np.isfinite(candidates["wape"]) & np.isfinite(candidates["rmse"])
    ]
    if candidates.empty:
        raise ValueError("No finite validation metrics are available for forecasting.")
    selected = (
        candidates.sort_values(["frazione", "wape", "rmse", "model"], kind="stable")
        .drop_duplicates("frazione")
        .set_index("frazione")["model"]
    )
    return {str(fraction): str(model) for fraction, model in selected.items()}


def forecast_schedule(
    calendar_path: Path,
    zones: list[str],
    fractions: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    parts = [
        read_calendar(calendar_path, zones, fractions, year, 0, "error")
        for year in range(start.year, end.year + 1)
    ]
    schedule = pd.concat(parts, ignore_index=True)
    dates = pd.to_datetime(schedule["data"]).dt.date
    return schedule[(dates >= start) & (dates <= end)].reset_index(drop=True)


def _future_allocator_shares(
    civic_input_path: Path,
    bundle: dict[str, object],
    history_end: date,
    fractions: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW civic_raw AS "
        f"SELECT * FROM read_parquet({sql_lit(civic_input_path.resolve())})"
    )
    columns = table_schema(con, "civic_raw")
    require_columns(
        columns,
        {
            "id_civico",
            "data",
            "zona_raccolta",
            "frazione",
            "cluster_label",
            "conferimento_t",
        },
        "classified collections",
    )
    optional_numeric = {
        column: (
            f"try_cast({sql_ident(column)} AS DOUBLE)"
            if column in columns
            else "NULL::DOUBLE"
        )
        for column in ("coord_x", "coord_y", "densita_locale")
    }
    con.execute(f"""
        CREATE TEMP VIEW civic_rows AS
        SELECT
            cast(id_civico AS VARCHAR) AS id_civico,
            try_cast(data AS DATE) AS data,
            trim(cast(zona_raccolta AS VARCHAR)) AS zona_raccolta,
            trim(cast(frazione AS VARCHAR)) AS frazione,
            try_cast(cluster_label AS INTEGER) AS cluster_label,
            try_cast(conferimento_t AS DOUBLE) AS conferimento_t,
            {optional_numeric["coord_x"]} AS coord_x,
            {optional_numeric["coord_y"]} AS coord_y,
            {optional_numeric["densita_locale"]} AS densita_locale
        FROM civic_raw
        WHERE try_cast(data AS DATE) <= DATE {sql_lit(history_end)}
    """)
    check_zero(
        con,
        """
        SELECT count(*) FROM (
            SELECT id_civico
            FROM civic_rows
            GROUP BY id_civico
            HAVING count(DISTINCT struct_pack(
                zone := zona_raccolta, cluster := cluster_label
            )) <> 1
        )
        """,
        "each civic must have one zone/cluster mapping",
    )
    dimension = con.execute("""
        SELECT
            id_civico,
            any_value(zona_raccolta) AS zona_raccolta,
            any_value(cluster_label) AS cluster_label
        FROM civic_rows
        GROUP BY id_civico
        ORDER BY id_civico
    """).df()
    statistics = civic_share_statistics(con, history_end)
    statistics = statistics[statistics["frazione"].isin(fractions)]
    allocator_bundle = bundle.get("civic_allocator") or {}
    selected = allocator_bundle.get("selected_by_fraction") or {}
    saved_models = allocator_bundle.get("models") or {}
    parts = []
    for fraction in fractions:
        choice = selected.get(
            fraction,
            {"allocator": "historical", "setting": 0.90},
        )
        allocator = str(choice["allocator"])
        setting = float(choice["setting"])
        fraction_stats = statistics[statistics["frazione"] == fraction]
        if fraction_stats.empty:
            continue
        if allocator == "historical":
            shares = historical_share_ranges(fraction_stats, setting)
        elif allocator == "dirichlet":
            shares = normalized_dirichlet_shares(
                fraction_stats, setting, seed=history_end.year + 1
            )
        elif allocator == "bayesian_ridge":
            if "bayesian_ridge" not in saved_models:
                raise ValueError("Selected Bayesian allocator model is missing.")
            features = civic_bayesian_features(con, history_end)
            features = features[features["frazione"] == fraction]
            shares = normalized_bayesian_shares(
                saved_models["bayesian_ridge"],
                features,
                setting,
                seed=history_end.year + 1,
            )
        else:
            raise ValueError(f"Unsupported civic allocator: {allocator}")
        shares["allocation_method"] = allocator
        shares["allocation_setting"] = setting
        parts.append(shares)
    con.close()
    if not parts:
        raise ValueError("No civic allocation shares are available.")
    return pd.concat(parts, ignore_index=True), dimension


def forecast_tabular_groups(
    frame: pd.DataFrame,
    schedule: pd.DataFrame,
    selected: dict[str, str],
    model_bundle: dict[str, object],
) -> pd.DataFrame:
    tabular_methods = {
        "baseline_last",
        "baseline_rolling4",
        "baseline_train_mean",
        "ridge",
        "sgd",
        "hist_gradient_boosting",
    }
    fractions = [
        fraction for fraction, model in selected.items() if model in tabular_methods
    ]
    if not fractions:
        return pd.DataFrame()

    training = frame[frame["frazione"].isin(fractions)].copy()
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_train = training[features]
    y_train = training["conferimento_t"].to_numpy(dtype=float)
    seed = int(model_bundle.get("seed", 42))
    builders = {
        "ridge": build_ridge,
        "sgd": lambda: build_sgd(seed),
        "hist_gradient_boosting": lambda: build_histogram_model(
            seed, str(model_bundle.get("hgb_loss", "poisson"))
        ),
    }
    needed_models = set(selected.values()) & set(builders)
    models = {name: builders[name]() for name in needed_models}
    for model in models.values():
        model.fit(x_train, y_train)
    means, global_mean = training_means(training)

    groups = training[SERIES_COLUMNS].drop_duplicates()
    future = schedule[schedule["frazione"].isin(fractions)].merge(
        groups,
        on=["zona_raccolta", "frazione"],
        how="inner",
        validate="many_to_many",
    )
    if future.empty:
        return pd.DataFrame()
    future["anno_produzione"] = pd.to_datetime(future["data"]).dt.year
    future["conferimento_t"] = np.nan
    state = training.copy()
    result_parts = []
    for event_date in sorted(pd.to_datetime(future["data"]).unique()):
        current = future[pd.to_datetime(future["data"]) == event_date].copy()
        state = pd.concat([state, current], ignore_index=True, sort=False)
        state = add_features(
            state.sort_values([*SERIES_COLUMNS, "data"], kind="stable").reset_index(
                drop=True
            )
        )
        mask = (state["data"] == event_date) & state["conferimento_t"].isna()
        rows = state.loc[mask].copy()
        predictions = pd.Series(np.nan, index=rows.index, dtype=float)
        for fraction, fraction_rows in rows.groupby("frazione", sort=False):
            method = selected[str(fraction)]
            if method == "baseline_last":
                values = fill_from_training_mean(
                    fraction_rows,
                    fraction_rows["lag_1_collection_t"],
                    means,
                    global_mean,
                )
            elif method == "baseline_rolling4":
                values = fill_from_training_mean(
                    fraction_rows,
                    fraction_rows["rolling_4_collection_mean_t"],
                    means,
                    global_mean,
                )
            elif method == "baseline_train_mean":
                values = np.asarray(
                    [
                        means.get((str(zone), str(fraction), int(cluster)), global_mean)
                        for zone, fraction, cluster in fraction_rows[
                            ["zona_raccolta", "frazione", "cluster_label"]
                        ].itertuples(index=False, name=None)
                    ],
                    dtype=float,
                )
            else:
                values = models[method].predict(fraction_rows[features])
            predictions.loc[fraction_rows.index] = np.maximum(values, 0.0)
        state.loc[rows.index, "conferimento_t"] = predictions
        rows["group_prediction_t"] = predictions
        rows["model"] = rows["frazione"].map(selected)
        rows["forecast_method"] = rows["model"]
        result_parts.append(
            rows[
                [
                    "scenario_seed",
                    "data",
                    "zona_raccolta",
                    "frazione",
                    "group_prediction_t",
                    "model",
                    "forecast_method",
                ]
            ]
            .groupby(
                [
                    "scenario_seed",
                    "data",
                    "zona_raccolta",
                    "frazione",
                    "model",
                    "forecast_method",
                ],
                as_index=False,
            )["group_prediction_t"]
            .sum()
        )
    return pd.concat(result_parts, ignore_index=True)


def forecast_arima_benchmark(
    input_path: Path,
    civic_input_path: Path,
    model_path: Path,
    metrics_path: Path,
    tabular_model_path: Path,
    tabular_metrics_path: Path,
    calendar_path: Path,
    output_path: Path,
    forecast_start: date,
    forecast_end: date,
) -> Path:
    if forecast_start > forecast_end:
        raise ValueError("forecast start must not be after forecast end")
    for label, path in {
        "withdrawals": input_path,
        "civic input": civic_input_path,
        "model": model_path,
        "metrics": metrics_path,
        "tabular model": tabular_model_path,
        "tabular metrics": tabular_metrics_path,
        "calendar": calendar_path,
    }.items():
        if not path.resolve().is_file():
            raise FileNotFoundError(f"{label} does not exist: {path.resolve()}")

    con = duckdb.connect()
    frame = con.execute(
        "SELECT * FROM read_parquet(?)", [str(input_path.resolve())]
    ).df()
    con.close()
    frame["data"] = pd.to_datetime(frame["data"])
    history = aggregate_arima_series(frame)
    history_end = history["data"].max().date()
    if forecast_start <= history_end:
        raise ValueError("forecast dates must be after the training data")

    bundle = joblib.load(model_path)
    tabular_bundle = joblib.load(tabular_model_path)
    if bundle.get("model_name") != "zone_fraction_sarima":
        raise ValueError("Unsupported zone/fraction ARIMA bundle.")
    selected = select_future_models(
        pd.concat(
            [pd.read_csv(tabular_metrics_path), pd.read_csv(metrics_path)],
            ignore_index=True,
        )
    )
    model_entries = bundle["models"]
    sarimax_entries = bundle["sarimax_models"]
    parsed_keys = []
    for model_key in model_entries:
        values = dict(part.split("=", 1) for part in model_key.split("|"))
        parsed_keys.append(
            (
                int(values["scenario_seed"]),
                values["zona_raccolta"],
                values["frazione"],
                model_key,
            )
        )
    zones = sorted(history["zona_raccolta"].astype(str).unique())
    fractions = sorted(selected)
    schedule = forecast_schedule(
        calendar_path.resolve(), zones, fractions, forecast_start, forecast_end
    )
    full_start = history_end + timedelta(days=1)
    full_schedule = forecast_schedule(
        calendar_path.resolve(), zones, fractions, full_start, forecast_end
    )

    rows: list[dict[str, object]] = []
    for scenario_seed, zone, fraction, model_key in parsed_keys:
        method = selected.get(fraction)
        if method is None:
            raise ValueError(f"No selected forecast model for fraction {fraction}.")
        if method not in {"sarima", "sarimax", "seasonal_naive_7d"}:
            continue
        group_history = history[
            (history["scenario_seed"] == scenario_seed)
            & (history["zona_raccolta"] == zone)
            & (history["frazione"] == fraction)
        ].sort_values("data")
        group_schedule = full_schedule[
            (full_schedule["zona_raccolta"] == zone)
            & (full_schedule["frazione"] == fraction)
        ].sort_values("data")
        target_schedule = schedule[
            (schedule["zona_raccolta"] == zone)
            & (schedule["frazione"] == fraction)
        ].sort_values("data")
        if target_schedule.empty:
            continue
        index = pd.date_range(full_start, forecast_end, freq="D")
        if method == "sarima":
            raw = sarima_forecast(
                model_entries[model_key]["model"], full_start, forecast_end
            )
        elif method == "sarimax":
            exog = daily_sarimax_exog(
                group_schedule, full_start, forecast_end
            ).reindex(index)
            raw = sarima_forecast(
                sarimax_entries[model_key]["model"],
                full_start,
                forecast_end,
                exog,
            )
        else:
            raw = pd.Series(np.nan, index=index)

        known = {
            timestamp.date(): float(value)
            for timestamp, value in zip(
                group_history["data"],
                group_history["conferimento_t"],
                strict=True,
            )
        }
        train_mean = float(group_history["conferimento_t"].mean())
        scheduled_dates = set(pd.to_datetime(group_schedule["data"]).dt.date)
        for event_date in sorted(scheduled_dates):
            fallback = known.get(event_date - timedelta(days=7), train_mean)
            predicted = float(raw.get(pd.Timestamp(event_date), np.nan))
            actual_method = method
            if method == "seasonal_naive_7d" or not np.isfinite(predicted):
                predicted = fallback
                actual_method = "seasonal_naive_7d"
            known[event_date] = max(predicted, 0.0)
            if forecast_start <= event_date <= forecast_end:
                rows.append(
                    {
                        "scenario_seed": scenario_seed,
                        "data": event_date,
                        "zona_raccolta": zone,
                        "frazione": fraction,
                        "group_prediction_t": max(predicted, 0.0),
                        "model": method,
                        "forecast_method": actual_method,
                    }
                )
    tabular_groups = forecast_tabular_groups(
        frame,
        schedule,
        selected,
        tabular_bundle,
    )
    groups = pd.concat(
        [part for part in (pd.DataFrame(rows), tabular_groups) if not part.empty],
        ignore_index=True,
    )
    if groups.empty:
        raise ValueError("The forecast calendar produced no future withdrawals.")

    shares, dimension = _future_allocator_shares(
        civic_input_path.resolve(), bundle, history_end, fractions
    )
    allocated = groups.merge(
        dimension, on="zona_raccolta", how="inner", validate="many_to_many"
    ).merge(
        shares,
        on=["id_civico", "zona_raccolta", "frazione"],
        how="left",
        validate="many_to_one",
    )
    group_keys = ["scenario_seed", "data", "zona_raccolta", "frazione"]
    allocated["point_share"] = allocated["point_share"].fillna(0.0).clip(lower=0)
    totals = allocated.groupby(group_keys)["point_share"].transform("sum")
    counts = allocated.groupby(group_keys)["id_civico"].transform("count")
    allocated["point_share"] = np.where(
        totals > 0, allocated["point_share"] / totals, 1.0 / counts
    )
    allocated["lower_share"] = allocated["lower_share"].fillna(
        allocated["point_share"]
    )
    allocated["upper_share"] = allocated["upper_share"].fillna(
        allocated["point_share"]
    )
    allocated["prediction_t"] = (
        allocated["group_prediction_t"] * allocated["point_share"]
    )
    allocated["prediction_lower_t"] = np.minimum(
        allocated["group_prediction_t"] * allocated["lower_share"],
        allocated["prediction_t"],
    )
    allocated["prediction_upper_t"] = np.maximum(
        allocated["group_prediction_t"] * allocated["upper_share"],
        allocated["prediction_t"],
    )
    allocated["pipeline"] = "zone-fraction"
    canonical = allocated[
        [
            "id_civico",
            "data",
            "frazione",
            "prediction_t",
            "prediction_lower_t",
            "prediction_upper_t",
            "pipeline",
            "model",
            "forecast_method",
            "zona_raccolta",
            "cluster_label",
        ]
    ].sort_values(
        ["data", "zona_raccolta", "frazione", "cluster_label", "id_civico"],
        kind="stable",
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_con = duckdb.connect()
    write_parquet(output_con, canonical, output_path)
    output_con.close()
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and benchmark zone/fraction scheduled withdrawals."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify = subparsers.add_parser("classify")
    classify.add_argument("--collections-input", type=Path, required=True)
    classify.add_argument("--assignments-input", type=Path, required=True)
    classify.add_argument("--output", type=Path, required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--annual-input", type=Path, required=True)
    prepare.add_argument("--production-input", type=Path)
    prepare.add_argument("--collections-input", type=Path, required=True)
    prepare.add_argument("--calendar-input", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--year", type=int, required=True)
    prepare.add_argument("--annual-seed", type=int, default=42)
    prepare.add_argument("--scenario-seed", type=int, default=42)
    prepare.add_argument("--spillover-days", type=int, default=14)
    prepare.add_argument(
        "--missing-calendar-mode", choices=["error", "daily"], default="error"
    )
    prepare.add_argument("--generator-version", default="unversioned-current")

    train = subparsers.add_parser("train")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--exclude-episodic-fractions", action="store_true")
    train.add_argument(
        "--hgb-loss",
        choices=["poisson", "squared_error"],
        default="poisson",
    )
    train.add_argument(
        "--validation-strategy",
        choices=["fixed", "k-fold"],
        default="fixed",
    )
    train.add_argument("--validation-folds", type=int, default=5)
    arima = subparsers.add_parser("arima")
    arima.add_argument("--input", type=Path, required=True)
    arima.add_argument("--civic-input", type=Path)
    arima.add_argument("--output-dir", type=Path, required=True)
    arima.add_argument("--exclude-episodic-fractions", action="store_true")
    arima.add_argument(
        "--validation-strategy",
        choices=["fixed", "k-fold"],
        default="fixed",
    )
    arima.add_argument("--validation-folds", type=int, default=5)
    forecast = subparsers.add_parser("forecast")
    forecast.add_argument("--input", type=Path, required=True)
    forecast.add_argument("--civic-input", type=Path, required=True)
    forecast.add_argument("--model", type=Path, required=True)
    forecast.add_argument("--metrics", type=Path, required=True)
    forecast.add_argument("--tabular-model", type=Path, required=True)
    forecast.add_argument("--tabular-metrics", type=Path, required=True)
    forecast.add_argument("--calendar", type=Path, required=True)
    forecast.add_argument("--output", type=Path, required=True)
    forecast.add_argument("--forecast-from-date", type=date.fromisoformat, required=True)
    forecast.add_argument("--forecast-to-date", type=date.fromisoformat, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "classify":
        output = classify_collections(
            args.collections_input,
            args.assignments_input,
            args.output,
        )
        print(f"saved classified collections to {output}")
        return

    if args.command == "prepare":
        outputs = prepare_complete_dataset(
            annual_path=args.annual_input,
            production_path=args.production_input,
            collections_path=args.collections_input,
            calendar_path=args.calendar_input,
            output_dir=args.output_dir,
            year=args.year,
            annual_seed=args.annual_seed,
            scenario_seed=args.scenario_seed,
            spillover_days=args.spillover_days,
            missing_calendar_mode=args.missing_calendar_mode,
            generator_version=args.generator_version,
        )
        print(f"saved withdrawals to {outputs.withdrawals}")
        print(f"saved static civic dimension to {outputs.static_civics}")
        print(f"saved manifest to {outputs.manifest}")
        return

    if args.command == "arima":
        outputs = train_arima_benchmark(
            args.input,
            args.output_dir,
            args.exclude_episodic_fractions,
            args.validation_strategy,
            args.validation_folds,
            args.civic_input,
        )
        print(f"saved ARIMA model to {outputs['model']}")
        print(f"saved ARIMA metrics to {outputs['metrics']}")
        print(f"saved ARIMA test predictions to {outputs['predictions']}")
        if outputs["civic_predictions"]:
            print(
                "saved civic ARIMA test predictions to "
                f"{outputs['civic_predictions']}"
            )
            print(f"saved civic ARIMA metrics to {outputs['civic_metrics']}")
            print(
                "saved civic allocator benchmark to "
                f"{outputs['civic_allocator_metrics']}"
            )
        return

    if args.command == "forecast":
        output = forecast_arima_benchmark(
            args.input,
            args.civic_input,
            args.model,
            args.metrics,
            args.tabular_model,
            args.tabular_metrics,
            args.calendar,
            args.output,
            args.forecast_from_date,
            args.forecast_to_date,
        )
        print(f"saved future predictions to {output}")
        return

    outputs = train_benchmark(
        args.input,
        args.output_dir,
        args.seed,
        args.exclude_episodic_fractions,
        args.hgb_loss,
        args.validation_strategy,
        args.validation_folds,
    )
    print(f"saved model to {outputs['model']}")
    print(f"saved metrics to {outputs['metrics']}")
    print(f"saved test predictions to {outputs['predictions']}")


if __name__ == "__main__":
    main()
