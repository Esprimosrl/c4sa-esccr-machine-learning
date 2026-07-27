#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import argparse
import json
import sys
import warnings

import duckdb
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_data import forecast_events


GROUP_COLUMN_BY_MODE = {"cluster": "cluster_label", "zone": "zona_raccolta"}
ORDERS = ((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1))
SEASONAL_ORDERS = ((1, 0, 0, 7), (0, 1, 1, 7))


@dataclass(frozen=True)
class GroupSpec:
    key: str
    frazione: str
    group_col: str
    group_value: str


@dataclass(frozen=True)
class RollingFold:
    train_end: int
    validation_end: int


def sql_lit(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def source_sql(path: Path, input_format: str) -> str:
    selected = input_format
    if selected == "auto":
        if path.suffix.lower() == ".csv":
            selected = "csv"
        elif path.suffix.lower() in {".parquet", ".pq"}:
            selected = "parquet"
        else:
            raise ValueError("Could not detect input format; use --input-format.")
    if selected == "csv":
        return f"read_csv_auto({sql_lit(path)}, header=true, all_varchar=true)"
    if selected == "parquet":
        return f"read_parquet({sql_lit(path)})"
    raise ValueError(f"Unsupported input format: {selected}")


def table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {
        row[1]
        for row in con.execute(f"PRAGMA table_info({sql_lit(table)})").fetchall()
    }


def parse_split_ratios(value: str) -> tuple[float, float, float]:
    try:
        ratios = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Split ratios must be numeric.") from exc
    if len(ratios) != 3 or any(item <= 0 for item in ratios):
        raise argparse.ArgumentTypeError("Expected three positive split ratios.")
    if sum(ratios) > 1.5:
        ratios = tuple(item / 100 for item in ratios)
    if not np.isclose(sum(ratios), 1.0):
        raise argparse.ArgumentTypeError("Split ratios must sum to 1.0 or 100.")
    return ratios


def create_observations(
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    input_format: str,
    id_col: str,
    date_col: str,
    year_col: str,
    target_col: str,
    fraction_col: str,
    cluster_col: str,
    zone_col: str,
    annual_weight_col: str,
    group_by: str = "cluster",
) -> set[str]:
    con.execute(f"CREATE OR REPLACE TEMP VIEW raw_data AS SELECT * FROM {source_sql(input_path, input_format)}")
    columns = table_columns(con, "raw_data")
    required = {id_col, date_col, target_col, fraction_col}
    required.add(zone_col if group_by == "zone" else cluster_col)
    missing = required - columns
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}.")

    year_expr = (
        f"TRY_CAST({sql_ident(year_col)} AS INTEGER)"
        if year_col in columns
        else f"EXTRACT(YEAR FROM TRY_CAST({sql_ident(date_col)} AS DATE))::INTEGER"
    )
    cluster_expr = "NULL::VARCHAR"
    if cluster_col in columns:
        cluster_expr = (
            f"CAST(CAST(TRY_CAST({sql_ident(cluster_col)} AS DOUBLE) AS BIGINT) "
            "AS VARCHAR)"
        )
    zone_expr = "NULL::VARCHAR"
    if zone_col in columns:
        raw_zone = f"NULLIF(TRIM(CAST({sql_ident(zone_col)} AS VARCHAR)), '')"
        zone_expr = f"""
            CASE
                WHEN TRY_CAST({raw_zone} AS DOUBLE) IS NOT NULL THEN
                    CAST(CAST(TRY_CAST({raw_zone} AS DOUBLE) AS BIGINT) AS VARCHAR)
                ELSE {raw_zone}
            END
        """
    weight_expr = (
        f"TRY_CAST(REPLACE(CAST({sql_ident(annual_weight_col)} AS VARCHAR), ',', '.') AS DOUBLE)"
        if annual_weight_col in columns
        else "NULL::DOUBLE"
    )
    group_filter = (
        f"NULLIF(TRIM(CAST({sql_ident(zone_col)} AS VARCHAR)), '') IS NOT NULL"
        if group_by == "zone"
        else f"TRY_CAST({sql_ident(cluster_col)} AS DOUBLE) IS NOT NULL"
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE observations AS
        SELECT
            CAST({sql_ident(id_col)} AS VARCHAR) AS id_civico,
            TRY_CAST({sql_ident(date_col)} AS DATE) AS data,
            {year_expr} AS anno_produzione,
            TRIM(CAST({sql_ident(fraction_col)} AS VARCHAR)) AS frazione,
            {cluster_expr} AS cluster_label,
            {zone_expr} AS zona_raccolta,
            TRY_CAST(REPLACE(CAST({sql_ident(target_col)} AS VARCHAR), ',', '.') AS DOUBLE)
                AS target_value,
            {weight_expr} AS historical_annual_t
        FROM raw_data
        WHERE {sql_ident(id_col)} IS NOT NULL
          AND TRY_CAST({sql_ident(date_col)} AS DATE) IS NOT NULL
          AND {sql_ident(fraction_col)} IS NOT NULL
          AND {group_filter}
          AND TRY_CAST(REPLACE(CAST({sql_ident(target_col)} AS VARCHAR), ',', '.') AS DOUBLE) IS NOT NULL;
    """)
    invalid = con.execute("""
        SELECT count(*) FROM observations
        WHERE target_value < 0 OR NOT isfinite(target_value)
    """).fetchone()[0]
    if invalid:
        raise ValueError("Target values must be finite and non-negative.")
    return columns


def load_daily_series(
    con: duckdb.DuckDBPyConnection,
    group_by: str = "cluster",
) -> tuple[pd.DatetimeIndex, list[GroupSpec], dict[str, np.ndarray]]:
    first_day, last_day = con.execute(
        "SELECT min(data), max(data) FROM observations"
    ).fetchone()
    if first_day is None:
        raise ValueError("The input contains no usable observations.")
    calendar = pd.date_range(first_day, last_day, freq="D")
    fractions = [row[0] for row in con.execute(
        "SELECT DISTINCT frazione FROM observations ORDER BY frazione"
    ).fetchall()]
    group_col = GROUP_COLUMN_BY_MODE[group_by]
    group_values = [row[0] for row in con.execute(
        f"SELECT DISTINCT {group_col} FROM observations "
        f"WHERE {group_col} IS NOT NULL "
        f"ORDER BY TRY_CAST({group_col} AS BIGINT), {group_col}"
    ).fetchall()]
    groups = [
        GroupSpec(
            f"frazione={fraction}|{group_col}={group_value}",
            fraction,
            group_col,
            group_value,
        )
        for fraction in fractions
        for group_value in group_values
    ]
    daily = con.execute(f"""
        SELECT data, frazione, {group_col}, sum(target_value) AS target_value
        FROM observations
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
    """).df()
    daily["data"] = pd.to_datetime(daily["data"])
    series: dict[str, np.ndarray] = {}
    for group in groups:
        selected = daily[
            (daily["frazione"] == group.frazione)
            & (daily[group.group_col] == group.group_value)
        ]
        series[group.key] = (
            selected.set_index("data")["target_value"]
            .reindex(calendar, fill_value=0.0)
            .to_numpy(dtype=np.float64)
        )
    return calendar, groups, series


def split_positions(
    observations: int, ratios: tuple[float, float, float]
) -> tuple[int, int]:
    if observations < 21:
        raise ValueError("Weekly SARIMA needs at least 21 calendar days.")
    train_end = max(14, int(observations * ratios[0]))
    validation_end = max(train_end + 1, int(observations * sum(ratios[:2])))
    validation_end = min(validation_end, observations - 1)
    if train_end >= validation_end:
        raise ValueError("The date range is too short for all three splits.")
    return train_end, validation_end


def rolling_validation_folds(
    train_end: int, validation_end: int, count: int = 3
) -> list[RollingFold]:
    horizon = validation_end - train_end
    first_validation = validation_end - count * horizon
    if horizon < 1 or first_validation < 14:
        raise ValueError("The date range is too short for three rolling folds.")
    return [
        RollingFold(start, start + horizon)
        for start in range(first_validation, validation_end, horizon)
    ]


def seasonal_naive(history: np.ndarray, steps: int, period: int = 7) -> np.ndarray:
    if len(history) < period:
        raise ValueError(f"Seasonal-naive forecasting needs {period} observations.")
    return np.resize(np.asarray(history[-period:], dtype=np.float64), steps)


def rolling_naive_predictions(
    values: np.ndarray, folds: list[RollingFold]
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    actual_parts = []
    predicted_parts = []
    scores = []
    for fold in folds:
        actual = values[fold.train_end : fold.validation_end]
        predicted = seasonal_naive(values[: fold.train_end], len(actual))
        actual_parts.append(actual)
        predicted_parts.append(predicted)
        scores.append(rmse(actual, predicted))
    return np.concatenate(actual_parts), np.concatenate(predicted_parts), scores


def fit_model(values: np.ndarray, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]):
    total_difference = order[1] + seasonal_order[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(
            np.log1p(values),
            order=order,
            seasonal_order=seasonal_order,
            trend="c" if total_difference == 0 else "n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=100)


def forecast_raw(result: object, steps: int) -> np.ndarray:
    values = np.asarray(result.get_forecast(steps=steps).predicted_mean, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.expm1(values)
    values[~np.isfinite(values)] = np.nan
    return np.maximum(values, 0.0)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(predicted - actual))))


def select_order(
    values: np.ndarray, folds: list[RollingFold]
) -> tuple[
    tuple[int, int, int],
    tuple[int, int, int, int],
    float,
    np.ndarray,
    np.ndarray,
    list[float],
]:
    best = None
    for order in ORDERS:
        for seasonal_order in SEASONAL_ORDERS:
            actual_parts = []
            predicted_parts = []
            fold_scores = []
            for fold in folds:
                actual = values[fold.train_end : fold.validation_end]
                try:
                    result = fit_model(
                        values[: fold.train_end], order, seasonal_order
                    )
                    predicted = forecast_raw(result, len(actual))
                except (ValueError, np.linalg.LinAlgError):
                    break
                if not np.isfinite(predicted).all():
                    break
                actual_parts.append(actual)
                predicted_parts.append(predicted)
                fold_scores.append(rmse(actual, predicted))
            if len(fold_scores) != len(folds):
                continue
            score = float(np.mean(fold_scores))
            candidate = (
                score,
                order,
                seasonal_order,
                np.concatenate(actual_parts),
                np.concatenate(predicted_parts),
                fold_scores,
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        raise ValueError("No SARIMA candidate produced finite validation forecasts.")
    score, order, seasonal_order, actual, predicted, fold_scores = best
    return order, seasonal_order, score, actual, predicted, fold_scores


def metric_row(
    actual: np.ndarray,
    predicted: np.ndarray,
    level: str,
    split: str,
    group: GroupSpec | None,
    order: tuple[int, int, int] | None = None,
    seasonal_order: tuple[int, int, int, int] | None = None,
    converged: bool | None = None,
    forecast_method: str = "sarima",
) -> dict[str, object]:
    error = predicted - actual
    denominator = float(np.square(actual - actual.mean()).sum())
    return {
        "level": level,
        "model_key": group.key if group else "__all_trained_models__",
        "frazione": group.frazione if group else "__all__",
        "cluster_label": (
            group.group_value
            if group and group.group_col == "cluster_label"
            else "__all__" if group is None else ""
        ),
        "zona_raccolta": (
            group.group_value
            if group and group.group_col == "zona_raccolta"
            else "__all__" if group is None else ""
        ),
        "split": split,
        "rows": len(actual),
        "mae": float(np.abs(error).mean()),
        "rmse": rmse(actual, predicted),
        "log_rmse": rmse(np.log1p(actual), np.log1p(predicted)),
        "r2": float(1 - np.square(error).sum() / denominator)
        if denominator > 0
        else np.nan,
        "order": json.dumps(order) if order else "",
        "seasonal_order": json.dumps(seasonal_order) if seasonal_order else "",
        "converged": converged,
        "forecast_method": forecast_method,
    }


def write_query(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        options = "(FORMAT csv, HEADER)"
    else:
        options = "(FORMAT parquet)"
    con.execute(f"COPY ({query}) TO {sql_lit(path)} {options}")


def create_test_predictions(
    con: duckdb.DuckDBPyConnection,
    group_forecasts: pd.DataFrame,
    test_start: pd.Timestamp,
    output_path: Path,
) -> pd.DataFrame:
    con.register("test_group_forecasts", group_forecasts)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE civic_test_predictions AS
        WITH actual AS (
            SELECT
                id_civico,
                data,
                frazione,
                cluster_label,
                any_value(zona_raccolta) AS zona_raccolta,
                sum(target_value) AS actual_conferimento_t
            FROM observations
            WHERE data >= DATE {sql_lit(test_start.date())}
            GROUP BY 1, 2, 3, 4
        ),
        historical_weights AS (
            SELECT
                id_civico,
                frazione,
                sum(target_value) AS allocation_weight_t
            FROM observations
            WHERE data < DATE {sql_lit(test_start.date())}
            GROUP BY 1, 2
        ),
        eligible AS (
            SELECT
                a.*,
                g.pred_group_total_t,
                greatest(coalesce(w.allocation_weight_t, 0.0), 0.0) AS allocation_weight_t
            FROM actual a
            JOIN test_group_forecasts g USING (data, frazione, cluster_label)
            LEFT JOIN historical_weights w USING (id_civico, frazione)
        ),
        denominators AS (
            SELECT
                *,
                sum(allocation_weight_t) OVER group_day AS total_weight_t,
                count(*) OVER group_day AS eligible_civics
            FROM eligible
            WINDOW group_day AS (PARTITION BY data, frazione, cluster_label)
        )
        SELECT
            id_civico,
            data,
            frazione,
            TRY_CAST(cluster_label AS BIGINT) AS cluster_label,
            zona_raccolta,
            actual_conferimento_t,
            pred_group_total_t,
            allocation_weight_t,
            CASE
                WHEN total_weight_t > 0 THEN allocation_weight_t / total_weight_t
                ELSE 1.0 / eligible_civics
            END AS allocation_share,
            pred_group_total_t * CASE
                WHEN total_weight_t > 0 THEN allocation_weight_t / total_weight_t
                ELSE 1.0 / eligible_civics
            END AS pred_conferimento_t
        FROM denominators;
    """)
    write_query(
        con,
        "SELECT *, pred_conferimento_t - actual_conferimento_t AS prediction_error, abs(pred_conferimento_t - actual_conferimento_t) AS absolute_error FROM civic_test_predictions ORDER BY data, frazione, cluster_label, id_civico",
        output_path,
    )
    return con.execute("""
        SELECT
            frazione,
            CAST(cluster_label AS VARCHAR) AS cluster_label,
            list(actual_conferimento_t ORDER BY data, id_civico) AS actual,
            list(pred_conferimento_t ORDER BY data, id_civico) AS predicted
        FROM civic_test_predictions
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()


def write_group_test_predictions(
    con: duckdb.DuckDBPyConnection,
    group_forecasts: pd.DataFrame,
    output_path: Path,
    group_col: str = "cluster_label",
) -> None:
    con.register("group_test_predictions", group_forecasts)
    write_query(
        con,
        f"""
            SELECT
                *,
                pred_group_total_t - actual_group_total_t AS prediction_error,
                abs(pred_group_total_t - actual_group_total_t) AS absolute_error,
                seasonal_naive_group_total_t - actual_group_total_t
                    AS seasonal_naive_error,
                abs(seasonal_naive_group_total_t - actual_group_total_t)
                    AS seasonal_naive_absolute_error
            FROM group_test_predictions
            ORDER BY data, frazione, {group_col}
        """,
        output_path,
    )


def create_future_predictions(
    con: duckdb.DuckDBPyConnection,
    groups: list[GroupSpec],
    fitted_models: dict[str, dict[str, object]],
    last_observation: pd.Timestamp,
    forecast_start: date,
    forecast_end: date,
    calendar_path: Path | None,
    output_path: Path,
    group_level: bool = False,
    group_col: str = "cluster_label",
) -> None:
    if forecast_start <= last_observation.date():
        raise ValueError("Future forecast dates must follow the observation range.")
    steps = (forecast_end - last_observation.date()).days
    offset = (forecast_start - last_observation.date()).days - 1
    dates = pd.date_range(last_observation + pd.Timedelta(days=1), forecast_end, freq="D")
    group_rows = []
    for group in groups:
        model_info = fitted_models[group.key]
        predicted = forecast_raw(model_info["model"], steps)
        forecast_method = "sarima"
        if not np.isfinite(predicted).all():
            predicted = seasonal_naive(model_info["seasonal_naive_history"], steps)
            forecast_method = "seasonal_naive"
            print(
                f"warning: non-finite SARIMA future forecast for {group.key}; "
                "using 7-day seasonal naive",
                file=sys.stderr,
            )
        model_info["future_forecast_method"] = forecast_method
        predicted = predicted[offset:]
        group_rows.extend(
            (
                day.date(),
                group.frazione,
                group.group_value,
                float(value),
                forecast_method,
            )
            for day, value in zip(dates[offset:], predicted, strict=True)
        )
    future_groups = pd.DataFrame(
        group_rows,
        columns=[
            "data",
            "frazione",
            group_col,
            "pred_group_total_t",
            "forecast_method",
        ],
    )
    if group_level:
        con.register("future_group_forecasts", future_groups)
        write_query(
            con,
            f"SELECT * FROM future_group_forecasts ORDER BY data, frazione, {group_col}",
            output_path,
        )
        return

    zones = [row[0] for row in con.execute(
        "SELECT DISTINCT zona_raccolta FROM observations WHERE zona_raccolta IS NOT NULL"
    ).fetchall()]
    schedule = pd.DataFrame(
        forecast_events(
            calendar_path,
            zones,
            sorted({group.frazione for group in groups}),
            forecast_start,
            forecast_end,
        )
    )
    con.register("future_group_forecasts", future_groups)
    con.register("future_schedule", schedule)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE civic_future_predictions AS
        WITH latest AS (
            SELECT
                id_civico,
                frazione,
                cluster_label,
                zona_raccolta,
                greatest(coalesce(historical_annual_t, 0.0), 0.0) AS allocation_weight_t
            FROM observations
            WHERE zona_raccolta IS NOT NULL
            QUALIFY row_number() OVER (
                PARTITION BY id_civico, frazione
                ORDER BY anno_produzione DESC, data DESC
            ) = 1
        ),
        eligible AS (
            SELECT
                l.*,
                s.data,
                s.giorni_accumulati,
                g.pred_group_total_t,
                g.forecast_method
            FROM latest l
            JOIN future_schedule s USING (zona_raccolta, frazione)
            JOIN future_group_forecasts g USING (data, frazione, cluster_label)
        ),
        denominators AS (
            SELECT
                *,
                sum(allocation_weight_t) OVER group_day AS total_weight_t,
                count(*) OVER group_day AS eligible_civics
            FROM eligible
            WINDOW group_day AS (PARTITION BY data, frazione, cluster_label)
        )
        SELECT
            id_civico,
            data,
            frazione,
            TRY_CAST(cluster_label AS BIGINT) AS cluster_label,
            zona_raccolta,
            giorni_accumulati,
            pred_group_total_t,
            forecast_method,
            allocation_weight_t,
            CASE
                WHEN total_weight_t > 0 THEN allocation_weight_t / total_weight_t
                ELSE 1.0 / eligible_civics
            END AS allocation_share,
            pred_group_total_t * CASE
                WHEN total_weight_t > 0 THEN allocation_weight_t / total_weight_t
                ELSE 1.0 / eligible_civics
            END AS pred_conferimento_t
        FROM denominators;
    """)
    write_query(
        con,
        "SELECT * FROM civic_future_predictions ORDER BY data, frazione, cluster_label, id_civico",
        output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one weekly SARIMA model per fraction and selected group."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-format", choices=["auto", "csv", "parquet"], default="auto")
    parser.add_argument("--split-ratios", type=parse_split_ratios, default=(0.8, 0.1, 0.1))
    parser.add_argument("--id-col", default="id_civico")
    parser.add_argument("--date-col", default="data")
    parser.add_argument("--year-col", default="anno_produzione")
    parser.add_argument("--target-col", default="conferimento_t")
    parser.add_argument("--fraction-col", default="frazione")
    parser.add_argument("--cluster-col", default="cluster_label")
    parser.add_argument("--zone-col", default="zona_raccolta")
    parser.add_argument("--annual-weight-col", default="historical_annual_t")
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--forecast-from-date", type=date.fromisoformat)
    parser.add_argument("--forecast-to-date", type=date.fromisoformat)
    parser.add_argument("--forecast-calendar", type=Path)
    parser.add_argument("--future-predictions-output", type=Path)
    parser.add_argument("--group-level", action="store_true")
    parser.add_argument("--group-by", choices=GROUP_COLUMN_BY_MODE, default="cluster")
    parser.add_argument("--duckdb-memory-limit")
    parser.add_argument("--duckdb-threads", type=int)
    parser.add_argument("--duckdb-temp-directory", type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.group_by == "zone" and not args.group_level:
        parser.error("--group-by zone requires --group-level.")
    required_future_values = (
        args.forecast_from_date,
        args.forecast_to_date,
        args.future_predictions_output,
    )
    all_future_values = required_future_values + (args.forecast_calendar,)
    if any(all_future_values) and not all(required_future_values):
        parser.error("Future forecasts require both dates and an output path.")
    if (
        any(required_future_values)
        and not args.group_level
        and args.forecast_calendar is None
    ):
        parser.error("Future civic forecasts also require --forecast-calendar.")
    if args.forecast_from_date and args.forecast_from_date > args.forecast_to_date:
        parser.error("--forecast-from-date must not be after --forecast-to-date")

    con = duckdb.connect()
    if args.duckdb_memory_limit:
        con.execute(f"PRAGMA memory_limit={sql_lit(args.duckdb_memory_limit)}")
    if args.duckdb_threads:
        con.execute(f"PRAGMA threads={args.duckdb_threads}")
    if args.duckdb_temp_directory:
        args.duckdb_temp_directory.mkdir(parents=True, exist_ok=True)
        con.execute(f"PRAGMA temp_directory={sql_lit(args.duckdb_temp_directory)}")
    columns = create_observations(
        con,
        args.input,
        args.input_format,
        args.id_col,
        args.date_col,
        args.year_col,
        args.target_col,
        args.fraction_col,
        args.cluster_col,
        args.zone_col,
        args.annual_weight_col,
        args.group_by,
    )
    if args.forecast_from_date and not args.group_level:
        required = {args.zone_col, args.annual_weight_col}
        if missing := required - columns:
            raise ValueError(
                "Future civic forecasts require columns: " + ", ".join(sorted(missing))
            )
        if not args.forecast_calendar.is_file():
            raise ValueError(f"Forecast calendar does not exist: {args.forecast_calendar}")
        invalid_weights = con.execute("""
            SELECT count(*) FROM observations
            WHERE historical_annual_t IS NOT NULL
              AND (historical_annual_t < 0 OR NOT isfinite(historical_annual_t))
        """).fetchone()[0]
        if invalid_weights:
            raise ValueError("Annual allocation weights must be finite and non-negative.")

    group_col = GROUP_COLUMN_BY_MODE[args.group_by]
    calendar, groups, series = load_daily_series(con, args.group_by)
    train_end, validation_end = split_positions(len(calendar), args.split_ratios)
    rolling_folds = rolling_validation_folds(train_end, validation_end)
    validation_dates = calendar[train_end:validation_end]
    test_dates = calendar[validation_end:]
    print(
        f"training {len(groups)} daily frazione/{group_col} series; "
        f"range={calendar[0].date()}..{calendar[-1].date()}"
    )

    fitted_models: dict[str, dict[str, object]] = {}
    metrics: list[dict[str, object]] = []
    validation_all: list[tuple[np.ndarray, np.ndarray]] = []
    test_all: list[tuple[np.ndarray, np.ndarray]] = []
    naive_validation_all: list[tuple[np.ndarray, np.ndarray]] = []
    naive_test_all: list[tuple[np.ndarray, np.ndarray]] = []
    test_forecast_rows = []
    for index, group in enumerate(groups, start=1):
        values = series[group.key]
        test = values[validation_end:]
        try:
            (
                order,
                seasonal_order,
                validation_score,
                rolling_actual,
                rolling_prediction,
                rolling_scores,
            ) = select_order(
                values, rolling_folds
            )
        except ValueError as exc:
            raise ValueError(f"{group.key}: {exc}") from exc
        (
            naive_rolling_actual,
            naive_rolling_prediction,
            naive_rolling_scores,
        ) = rolling_naive_predictions(values, rolling_folds)
        test_model = fit_model(values[:validation_end], order, seasonal_order)
        test_prediction = forecast_raw(test_model, len(test))
        naive_test_prediction = seasonal_naive(values[:validation_end], len(test))
        final_model = fit_model(values, order, seasonal_order)
        converged = bool(final_model.mle_retvals.get("converged", False))
        metrics.extend(
            [
                metric_row(
                    rolling_actual,
                    rolling_prediction,
                    "group_total",
                    "validation",
                    group,
                    order,
                    seasonal_order,
                ),
                metric_row(
                    naive_rolling_actual,
                    naive_rolling_prediction,
                    "group_total",
                    "validation",
                    group,
                    forecast_method="seasonal_naive",
                ),
                metric_row(
                    test,
                    test_prediction,
                    "group_total",
                    "test",
                    group,
                    order,
                    seasonal_order,
                    bool(test_model.mle_retvals.get("converged", False)),
                ),
                metric_row(
                    test,
                    naive_test_prediction,
                    "group_total",
                    "test",
                    group,
                    forecast_method="seasonal_naive",
                ),
            ]
        )
        validation_all.append((rolling_actual, rolling_prediction))
        test_all.append((test, test_prediction))
        naive_validation_all.append(
            (naive_rolling_actual, naive_rolling_prediction)
        )
        naive_test_all.append((test, naive_test_prediction))
        test_forecast_rows.extend(
            (
                day.date(),
                group.frazione,
                group.group_value,
                float(actual),
                float(prediction),
                float(naive_prediction),
            )
            for day, actual, prediction, naive_prediction in zip(
                test_dates,
                test,
                test_prediction,
                naive_test_prediction,
                strict=True,
            )
        )
        fitted_models[group.key] = {
            "model": final_model,
            "group_values": {
                "frazione": group.frazione,
                group.group_col: group.group_value,
            },
            "order": order,
            "seasonal_order": seasonal_order,
            "validation_rmse": validation_score,
            "rolling_validation_rmse": rolling_scores,
            "seasonal_naive_validation_rmse": float(
                np.mean(naive_rolling_scores)
            ),
            "seasonal_naive_rolling_validation_rmse": naive_rolling_scores,
            "seasonal_naive_history": values[-7:],
            "converged": converged,
        }
        print(
            f"[{index}/{len(groups)}] {group.key}: "
            f"order={order} seasonal_order={seasonal_order} "
            f"rolling_rmse={validation_score:.6g} "
            f"naive_rolling_rmse={np.mean(naive_rolling_scores):.6g}"
        )

    for method, splits in (
        ("sarima", (("validation", validation_all), ("test", test_all))),
        (
            "seasonal_naive",
            (
                ("validation", naive_validation_all),
                ("test", naive_test_all),
            ),
        ),
    ):
        for split, values in splits:
            actual = np.concatenate([item[0] for item in values])
            predicted = np.concatenate([item[1] for item in values])
            metrics.append(
                metric_row(
                    actual,
                    predicted,
                    "group_total",
                    split,
                    None,
                    forecast_method=method,
                )
            )

    test_group_forecasts = pd.DataFrame(
        test_forecast_rows,
        columns=[
            "data",
            "frazione",
            group_col,
            "actual_group_total_t",
            "pred_group_total_t",
            "seasonal_naive_group_total_t",
        ],
    )
    if args.group_level:
        write_group_test_predictions(
            con, test_group_forecasts, args.predictions_output, group_col
        )
    else:
        civic_metrics = create_test_predictions(
            con,
            test_group_forecasts.drop(
                columns=["actual_group_total_t", "seasonal_naive_group_total_t"]
            ),
            test_dates[0],
            args.predictions_output,
        )
        civic_all: list[tuple[np.ndarray, np.ndarray]] = []
        group_by_key = {group.key: group for group in groups}
        for row in civic_metrics.itertuples(index=False):
            group = group_by_key[
                f"frazione={row.frazione}|cluster_label={row.cluster_label}"
            ]
            actual = np.asarray(row.actual, dtype=np.float64)
            predicted = np.asarray(row.predicted, dtype=np.float64)
            metrics.append(metric_row(actual, predicted, "per_civic", "test", group))
            civic_all.append((actual, predicted))
        if civic_all:
            metrics.append(
                metric_row(
                    np.concatenate([item[0] for item in civic_all]),
                    np.concatenate([item[1] for item in civic_all]),
                    "per_civic",
                    "test",
                    None,
                )
            )

    if args.forecast_from_date:
        create_future_predictions(
            con,
            groups,
            fitted_models,
            calendar[-1],
            args.forecast_from_date,
            args.forecast_to_date,
            args.forecast_calendar,
            args.future_predictions_output,
            args.group_level,
            group_col,
        )
        print(f"saved future civic forecasts to {args.future_predictions_output}")

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.metrics_output, index=False)
    bundle = {
        "model_name": "grouped_arima",
        "model_display_name": "Grouped weekly SARIMA",
        "group_cols": ["frazione", group_col],
        "frequency": "D",
        "seasonal_period": 7,
        "validation_strategy": "3-fold expanding window",
        "selection_metric": "mean rolling validation RMSE",
        "benchmark": "7-day seasonal naive",
        "target_transform": "log1p",
        "inverse_target_transform": "expm1_clipped_at_zero",
        "prediction_level": "group" if args.group_level else "civic",
        "models": fitted_models,
        "split_ratios": args.split_ratios,
        "date_ranges": {
            "train": (str(calendar[0].date()), str(calendar[train_end - 1].date())),
            "validation": (
                str(validation_dates[0].date()),
                str(validation_dates[-1].date()),
            ),
            "test": (str(test_dates[0].date()), str(test_dates[-1].date())),
        },
        "rolling_validation_folds": [
            {
                "train": (
                    str(calendar[0].date()),
                    str(calendar[fold.train_end - 1].date()),
                ),
                "validation": (
                    str(calendar[fold.train_end].date()),
                    str(calendar[fold.validation_end - 1].date()),
                ),
            }
            for fold in rolling_folds
        ],
        "metrics": metrics,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_output)
    print(f"saved grouped model bundle to {args.model_output}")
    print(f"saved metrics to {args.metrics_output}")
    print(
        f"saved test {'group' if args.group_level else 'civic'} predictions "
        f"to {args.predictions_output}"
    )


if __name__ == "__main__":
    main()
