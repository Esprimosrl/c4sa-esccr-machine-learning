#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import argparse
import json
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lag_features import LAG_FEATURES, LagState

from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


CATEGORICAL_LEVELS = {
    "zona_raccolta": ["1", "2", "3", "4"],
}

GROUP_COLS = ["frazione", "cluster_label"]
Regressor = SGDRegressor | GaussianProcessRegressor | HistGradientBoostingRegressor


@dataclass(frozen=True)
class GroupSpec:
    key: str
    frazione: str
    cluster_label: str
    train_rows: int


def sql_lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def detect_input_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(
        "Could not detect input format. Use --input-format csv or parquet."
    )


def source_sql(path: Path, input_format: str) -> str:
    if input_format == "auto":
        input_format = detect_input_format(path)

    if input_format == "csv":
        return f"""
            read_csv_auto(
                {sql_lit(str(path))},
                header = true,
                all_varchar = true
            )
        """

    if input_format == "parquet":
        return f"read_parquet({sql_lit(str(path))})"

    raise ValueError(f"Unsupported input format: {input_format}")


def export_options(output_format: str) -> str:
    if output_format == "parquet":
        return "(FORMAT parquet)"
    if output_format == "csv":
        return "(FORMAT csv, HEADER)"
    raise ValueError(f"Unsupported output format: {output_format}")


def table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info({sql_lit(table_name)})").fetchall()
    return {row[1] for row in rows}


def numeric_expr(col: str, cols: set[str]) -> str:
    if col not in cols:
        return f"NULL::DOUBLE AS {sql_ident(col)}"

    return f"""
        TRY_CAST(
            REPLACE(CAST({sql_ident(col)} AS VARCHAR), ',', '.')
            AS DOUBLE
        ) AS {sql_ident(col)}
    """


def categorical_expr(col: str, cols: set[str]) -> str:
    if col not in cols:
        return f"NULL::VARCHAR AS {sql_ident(col)}"

    return f"CAST({sql_ident(col)} AS VARCHAR) AS {sql_ident(col)}"


def normalized_category_expr(col: str, cols: set[str], alias: str) -> str:
    if col not in cols:
        return f"'__MISSING__'::VARCHAR AS {sql_ident(alias)}"

    raw = f"NULLIF(TRIM(CAST({sql_ident(col)} AS VARCHAR)), '')"
    if col in {"zona_raccolta", "cluster_label"}:
        normalized = f"""
            CASE
                WHEN TRY_CAST({raw} AS DOUBLE) IS NOT NULL THEN
                    CAST(CAST(TRY_CAST({raw} AS DOUBLE) AS BIGINT) AS VARCHAR)
                ELSE {raw}
            END
        """
    else:
        normalized = raw

    return f"COALESCE({normalized}, '__MISSING__') AS {sql_ident(alias)}"


def parse_split_ratios(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Expected three comma-separated ratios: train,validation,test."
        )

    try:
        ratios = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Split ratios must be numeric.") from exc

    if any(ratio <= 0 for ratio in ratios):
        raise argparse.ArgumentTypeError("All split ratios must be greater than zero.")

    total = sum(ratios)
    if total > 1.5:
        ratios = tuple(ratio / 100.0 for ratio in ratios)
        total = sum(ratios)

    if not np.isclose(total, 1.0, atol=1e-6):
        raise argparse.ArgumentTypeError("Split ratios must sum to 1.0 or 100.")

    return ratios


def parse_penalty(value: str) -> str | None:
    if value.lower() == "none":
        return None
    if value in {"l2", "l1", "elasticnet"}:
        return value
    raise argparse.ArgumentTypeError("Expected one of: l2, l1, elasticnet, none.")


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a date in YYYY-MM-DD format.") from exc


def get_feature_columns(
    exclude_giorni_accumulati: bool,
) -> tuple[list[str], list[str], list[str]]:
    numeric_cols = [
        "coord_x",
        "coord_y",
        "densita_locale",
        "giorni_accumulati",
        "historical_annual_t",
        "historical_collection_mean_t",
        *LAG_FEATURES,
        "anno_ref",
        "month",
        "day_of_month",
        "day_of_year",
        "day_of_week",
        "week_of_year",
        "quarter",
        "season_sin",
        "season_cos",
    ]

    if exclude_giorni_accumulati:
        numeric_cols.remove("giorni_accumulati")

    categorical_cols = ["zona_raccolta"]
    return numeric_cols, categorical_cols, numeric_cols + categorical_cols


def feature_names(numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    names = list(numeric_cols)
    for col in categorical_cols:
        names.extend(f"{col}={level}" for level in CATEGORICAL_LEVELS[col])
    return names


def create_full_data_table(
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    input_format: str,
    id_col: str,
    date_col: str,
    year_col: str,
    target_col: str,
    date_format: str,
    limit_rows: int,
) -> None:
    limit_sql = f"LIMIT {int(limit_rows)}" if limit_rows > 0 else ""

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE raw_data AS
        SELECT
            row_number() OVER () AS __row_id,
            *
        FROM {source_sql(input_path, input_format)}
        {limit_sql};
    """)

    cols = table_columns(con, "raw_data")
    if id_col not in cols:
        raise ValueError(f"Missing id column: {id_col}")
    if date_col not in cols:
        raise ValueError(f"Missing date column: {date_col}")

    if year_col in cols:
        year_expr = f"TRY_CAST({sql_ident(year_col)} AS INTEGER)"
    else:
        year_expr = "NULL::INTEGER"

    if "target_value" in cols:
        target_expr = "TRY_CAST(target_value AS DOUBLE)"
    elif target_col in cols:
        target_expr = f"""
            TRY_CAST(
                REPLACE(CAST({sql_ident(target_col)} AS VARCHAR), ',', '.')
                AS DOUBLE
            )
        """
    else:
        raise ValueError(f"Missing target column: {target_col}")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE full_data AS
        WITH typed AS (
            SELECT
                __row_id,
                CAST({sql_ident(id_col)} AS VARCHAR) AS {sql_ident(id_col)},
                COALESCE(
                    TRY_CAST({sql_ident(date_col)} AS DATE),
                    TRY_STRPTIME(
                        CAST({sql_ident(date_col)} AS VARCHAR),
                        {sql_lit(date_format)}
                    )::DATE
                ) AS data_dt,
                {year_expr} AS anno_from_col,
                {target_expr} AS target_value,
                {numeric_expr("coord_x", cols)},
                {numeric_expr("coord_y", cols)},
                {numeric_expr("densita_locale", cols)},
                {numeric_expr("giorni_accumulati", cols)},
                {numeric_expr("historical_annual_t", cols)},
                {numeric_expr("historical_collection_mean_t", cols)},
                {categorical_expr("zona_raccolta", cols)},
                {categorical_expr("cluster_label", cols)},
                {categorical_expr("frazione", cols)},
                {normalized_category_expr("frazione", cols, "__group_frazione")},
                {normalized_category_expr(
                    "cluster_label", cols, "__group_cluster_label"
                )}
            FROM raw_data
        ),
        enriched AS (
            SELECT
                *,
                COALESCE(
                    anno_from_col,
                    EXTRACT(YEAR FROM data_dt)::INTEGER
                ) AS anno_ref,
                EXTRACT(MONTH FROM data_dt)::INTEGER AS month,
                EXTRACT(DAY FROM data_dt)::INTEGER AS day_of_month,
                CAST(strftime(data_dt, '%j') AS INTEGER) AS day_of_year,
                CAST(strftime(data_dt, '%w') AS INTEGER) AS day_of_week,
                CAST(strftime(data_dt, '%W') AS INTEGER) AS week_of_year,
                EXTRACT(QUARTER FROM data_dt)::INTEGER AS quarter,
                sin(2 * pi() * CAST(strftime(data_dt, '%j') AS DOUBLE) / 366.0)
                    AS season_sin,
                cos(2 * pi() * CAST(strftime(data_dt, '%j') AS DOUBLE) / 366.0)
                    AS season_cos
            FROM typed
        ),
        eligible AS (
            SELECT *
            FROM enriched
            WHERE {sql_ident(id_col)} IS NOT NULL
              AND data_dt IS NOT NULL
              AND target_value IS NOT NULL
        ),
        lagged AS (
            SELECT
                *,
                COALESCE(
                    lag(target_value) OVER (
                        PARTITION BY {sql_ident(id_col)}, __group_frazione
                        ORDER BY data_dt, __row_id
                    ),
                    0.0
                ) AS lag_1_collection_t,
                COALESCE(
                    sum(target_value) OVER (
                        PARTITION BY {sql_ident(id_col)}, __group_frazione
                        ORDER BY data_dt
                        RANGE BETWEEN INTERVAL 7 DAY PRECEDING
                        AND INTERVAL 1 DAY PRECEDING
                    ),
                    0.0
                ) AS rolling_7d_sum_t,
                COALESCE(
                    sum(target_value) OVER (
                        PARTITION BY {sql_ident(id_col)}, __group_frazione
                        ORDER BY data_dt
                        RANGE BETWEEN INTERVAL 30 DAY PRECEDING
                        AND INTERVAL 1 DAY PRECEDING
                    ),
                    0.0
                ) AS rolling_30d_sum_t,
                COALESCE(
                    median(target_value) OVER (
                        PARTITION BY {sql_ident(id_col)}, __group_frazione
                        ORDER BY data_dt
                        RANGE BETWEEN INTERVAL 90 DAY PRECEDING
                        AND INTERVAL 1 DAY PRECEDING
                    ),
                    0.0
                ) AS rolling_90d_median_t,
                count(target_value) OVER (
                    PARTITION BY {sql_ident(id_col)}, __group_frazione
                    ORDER BY data_dt
                    RANGE BETWEEN INTERVAL 90 DAY PRECEDING
                    AND INTERVAL 1 DAY PRECEDING
                )::DOUBLE AS rolling_90d_count,
                COALESCE(
                    date_diff(
                        'day',
                        max(CASE WHEN target_value > 0 THEN data_dt END) OVER (
                            PARTITION BY {sql_ident(id_col)}, __group_frazione
                            ORDER BY data_dt
                            RANGE BETWEEN UNBOUNDED PRECEDING
                            AND INTERVAL 1 DAY PRECEDING
                        ),
                        data_dt
                    ),
                    0
                )::DOUBLE AS days_since_last_nonzero_collection
            FROM eligible
        )
        SELECT
            *,
            row_number() OVER (ORDER BY data_dt, __row_id) AS __time_rank,
            count(*) OVER () AS __total_rows
        FROM lagged;
    """)


def create_split_tables(
    con: duckdb.DuckDBPyConnection,
    split_ratios: tuple[float, float, float],
    train_end: date | None = None,
    validation_end: date | None = None,
    test_end: date | None = None,
) -> dict[str, int]:
    total_rows = con.execute("SELECT count(*) FROM full_data;").fetchone()[0]
    if total_rows < 3:
        raise ValueError("Need at least three target rows for train/validation/test.")

    if train_end is not None or validation_end is not None or test_end is not None:
        if train_end is None or validation_end is None:
            raise ValueError(
                "--train-end and --validation-end are both required for fixed splits."
            )
        if train_end >= validation_end:
            raise ValueError("--train-end must be earlier than --validation-end.")
        if test_end is not None and validation_end >= test_end:
            raise ValueError("--test-end must be later than --validation-end.")
        test_limit = (
            f" AND data_dt <= DATE {sql_lit(test_end.isoformat())}"
            if test_end is not None
            else ""
        )
        split_sql = {
            "train": f"data_dt <= DATE {sql_lit(train_end.isoformat())}",
            "validation": (
                f"data_dt > DATE {sql_lit(train_end.isoformat())} "
                f"AND data_dt <= DATE {sql_lit(validation_end.isoformat())}"
            ),
            "test": (
                f"data_dt > DATE {sql_lit(validation_end.isoformat())}"
                f"{test_limit}"
            ),
        }
    else:
        train_ratio, validation_ratio, _ = split_ratios
        train_cutoff = max(1, int(total_rows * train_ratio))
        validation_cutoff = max(
            train_cutoff + 1, int(total_rows * (train_ratio + validation_ratio))
        )
        validation_cutoff = min(validation_cutoff, total_rows - 1)

        split_sql = {
            "train": f"__time_rank <= {train_cutoff}",
            "validation": (
                f"__time_rank > {train_cutoff} "
                f"AND __time_rank <= {validation_cutoff}"
            ),
            "test": f"__time_rank > {validation_cutoff}",
        }

    counts = {}
    for split_name, where_sql in split_sql.items():
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {split_name}_data AS
            SELECT *
            FROM full_data
            WHERE {where_sql};
        """)
        counts[split_name] = con.execute(
            f"SELECT count(*) FROM {split_name}_data;"
        ).fetchone()[0]

    return counts


def discover_groups(
    con: duckdb.DuckDBPyConnection,
    min_train_rows: int,
) -> list[GroupSpec]:
    rows = con.execute(f"""
        SELECT
            __group_frazione,
            __group_cluster_label,
            count(*) AS train_rows
        FROM train_data
        GROUP BY 1, 2
        HAVING count(*) >= {int(min_train_rows)}
        ORDER BY __group_frazione, __group_cluster_label;
    """).fetchall()

    groups = []
    for frazione, cluster_label, train_rows in rows:
        key = f"frazione={frazione}|cluster_label={cluster_label}"
        groups.append(
            GroupSpec(
                key=key,
                frazione=str(frazione),
                cluster_label=str(cluster_label),
                train_rows=int(train_rows),
            )
        )

    if not groups:
        raise ValueError(
            "No trainable frazione/cluster_label groups were found. "
            "Lower --min-train-rows or check the input columns."
        )

    return groups


def group_where(group: GroupSpec) -> str:
    return (
        f"__group_frazione = {sql_lit(group.frazione)} "
        f"AND __group_cluster_label = {sql_lit(group.cluster_label)}"
    )


def compute_numeric_stats(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    numeric_cols: list[str],
    where_sql: str,
) -> dict[str, tuple[float, float]]:
    stats = {}

    for col in numeric_cols:
        mean_value, std_value = con.execute(f"""
            SELECT
                avg({sql_ident(col)}) AS mean_value,
                stddev_pop({sql_ident(col)}) AS std_value
            FROM {table_name}
            WHERE {where_sql}
              AND {sql_ident(col)} IS NOT NULL;
        """).fetchone()

        if mean_value is None:
            mean_value = 0.0

        if std_value is None or std_value == 0:
            std_value = 1.0

        stats[col] = (float(mean_value), float(std_value))

    return stats


def fetch_batch(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    last_time_rank: int,
    batch_size: int,
    cols: list[str],
    where_sql: str,
) -> pd.DataFrame:
    selected_cols = ", ".join(sql_ident(c) for c in ["__time_rank"] + cols)
    return con.execute(f"""
        SELECT {selected_cols}
        FROM {table_name}
        WHERE __time_rank > {last_time_rank}
          AND {where_sql}
        ORDER BY __time_rank
        LIMIT {batch_size};
    """).df()


def normalize_category_values(series: pd.Series, col: str) -> pd.Series:
    values = series.astype("string").str.strip()

    if col in {"zona_raccolta", "cluster_label"}:
        numeric_values = pd.to_numeric(values, errors="coerce")
        values = numeric_values.astype("Int64").astype("string")

    return values.fillna("__MISSING__")


def make_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
) -> np.ndarray:
    feature_blocks = []

    for col in numeric_cols:
        mean_value, std_value = numeric_stats[col]
        values = pd.to_numeric(df[col], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        values = values.fillna(mean_value)
        values = ((values - mean_value) / std_value).astype("float32")
        feature_blocks.append(values.to_numpy().reshape(-1, 1))

    for col in categorical_cols:
        values = normalize_category_values(df[col], col)
        for level in CATEGORICAL_LEVELS[col]:
            encoded = (values == level).astype("float32")
            feature_blocks.append(encoded.to_numpy().reshape(-1, 1))

    return np.hstack(feature_blocks).astype("float32")


def make_target_raw(series: pd.Series) -> np.ndarray:
    y = pd.to_numeric(series, errors="coerce")
    y = y.fillna(0).clip(lower=0)
    return y.to_numpy(dtype=np.float64)


def make_target_log(series: pd.Series) -> np.ndarray:
    return np.log1p(make_target_raw(series))


def clip_raw_predictions(
    predictions: np.ndarray, upper_bound: float | None = None
) -> np.ndarray:
    result = np.asarray(predictions, dtype=np.float64).copy()
    result[~np.isfinite(result)] = 0.0
    if upper_bound is not None:
        return np.clip(result, 0.0, upper_bound)
    return np.maximum(result, 0.0)


def prediction_log_to_raw(
    pred_log: np.ndarray, upper_bound: float | None = None
) -> np.ndarray:
    values = np.asarray(pred_log, dtype=np.float64)
    if upper_bound is not None:
        values = np.minimum(values, np.log1p(upper_bound))
    with np.errstate(over="ignore", invalid="ignore"):
        predictions = np.expm1(values)
    return clip_raw_predictions(predictions, upper_bound)


def prediction_to_raw(
    model: Regressor,
    features: np.ndarray,
    target_transform: str,
    upper_bound: float | None = None,
) -> np.ndarray:
    prediction = model.predict(features)
    if target_transform == "log1p":
        return prediction_log_to_raw(prediction, upper_bound)
    return clip_raw_predictions(prediction, upper_bound)


def make_progress_bar(enabled: bool, total: int, desc: str):
    if not enabled:
        return None

    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "--progress requires tqdm. Install it with: pip install tqdm"
        ) from exc

    return tqdm(total=total, desc=desc, unit="rows")


def build_model(args: argparse.Namespace, group_index: int) -> Regressor:
    if args.regressor == "gpr":
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(1.0, (1e-2, 1e2))
            + WhiteKernel(0.1, (1e-5, 1e1))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=args.seed + group_index,
        )
    if args.regressor == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=args.seed + group_index,
        )

    return SGDRegressor(
        loss="squared_error",
        penalty=args.penalty,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        learning_rate=args.learning_rate,
        eta0=args.eta0,
        power_t=args.power_t,
        max_iter=1,
        tol=None,
        average=args.average,
        random_state=args.seed + group_index,
    )


def fit_sgd(
    con: duckdb.DuckDBPyConnection,
    model: SGDRegressor,
    group: GroupSpec,
    epochs: int,
    batch_size: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
    progress: bool,
) -> SGDRegressor:
    fitted = False
    where_sql = group_where(group)

    for epoch in range(epochs):
        last_time_rank = 0
        batches = 0
        progress_bar = make_progress_bar(
            progress,
            group.train_rows,
            f"{group.key} epoch {epoch + 1}/{epochs}",
        )

        while True:
            batch = fetch_batch(
                con=con,
                table_name="train_data",
                last_time_rank=last_time_rank,
                batch_size=batch_size,
                cols=feature_cols + ["target_value"],
                where_sql=where_sql,
            )
            if batch.empty:
                break

            last_time_rank = int(batch["__time_rank"].max())
            x = make_features(batch, numeric_cols, categorical_cols, numeric_stats)
            y = make_target_log(batch["target_value"])
            model.partial_fit(x, y)
            fitted = True
            batches += 1

            if progress_bar is not None:
                progress_bar.update(len(batch))

        if progress_bar is not None:
            progress_bar.close()

        print(f"  finished epoch {epoch + 1}/{epochs}; batches={batches}")

    if not fitted:
        raise ValueError(f"No training batches were processed for {group.key}.")

    return model


def fetch_gpr_sample(
    con: duckdb.DuckDBPyConnection,
    group: GroupSpec,
    cols: list[str],
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    selected_cols = ", ".join(sql_ident(c) for c in ["__time_rank"] + cols)
    where_sql = group_where(group)
    if group.train_rows <= max_rows:
        return con.execute(f"""
            SELECT {selected_cols}
            FROM train_data
            WHERE {where_sql}
            ORDER BY __time_rank;
        """).df()

    return con.execute(f"""
        WITH bucketed AS (
            SELECT
                {selected_cols},
                ntile({int(max_rows)}) OVER (ORDER BY __time_rank) AS __sample_bucket
            FROM train_data
            WHERE {where_sql}
        ),
        sampled AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY __sample_bucket
                    ORDER BY hash(__time_rank, {int(seed)})
                ) AS __sample_rank
            FROM bucketed
        )
        SELECT {selected_cols}
        FROM sampled
        WHERE __sample_rank = 1
        ORDER BY __time_rank;
    """).df()


def fit_gpr(
    con: duckdb.DuckDBPyConnection,
    model: GaussianProcessRegressor,
    group: GroupSpec,
    max_rows: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
    seed: int,
) -> int:
    sample = fetch_gpr_sample(
        con,
        group,
        feature_cols + ["target_value"],
        max_rows,
        seed,
    )
    x = make_features(sample, numeric_cols, categorical_cols, numeric_stats)
    y = make_target_log(sample["target_value"])
    model.fit(x, y)
    print(f"  fitted GPR with {len(sample)}/{group.train_rows} training rows")
    return len(sample)


def fetch_group_rows(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    group: GroupSpec,
    cols: list[str],
) -> pd.DataFrame:
    selected_cols = ", ".join(sql_ident(c) for c in cols)
    return con.execute(f"""
        SELECT {selected_cols}
        FROM {table_name}
        WHERE {group_where(group)}
        ORDER BY __time_rank;
    """).df()


def fit_hist_gradient_boosting(
    con: duckdb.DuckDBPyConnection,
    model: HistGradientBoostingRegressor,
    group: GroupSpec,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
) -> int:
    columns = feature_cols + ["target_value"]
    train = fetch_group_rows(con, "train_data", group, columns)
    x_train = make_features(train, numeric_cols, categorical_cols, numeric_stats)
    y_train = make_target_raw(train["target_value"])
    model.set_params(early_stopping=False)
    model.fit(x_train, y_train)
    print(f"  fitted histogram booster with {len(train)} training rows")
    return len(train)


def empty_metric_sums() -> dict[str, float]:
    return {
        "rows": 0.0,
        "abs_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "log_squared_error_sum": 0.0,
        "y_sum": 0.0,
        "y_abs_sum": 0.0,
        "y_squared_sum": 0.0,
    }


def metric_sums_to_row(
    sums: dict[str, float],
    split_name: str,
    model_key: str,
    frazione: str,
    cluster_label: str,
    skipped_rows: int = 0,
) -> dict[str, float | str]:
    n_rows = int(sums["rows"])
    if n_rows == 0:
        return {
            "model_key": model_key,
            "frazione": frazione,
            "cluster_label": cluster_label,
            "split": split_name,
            "rows": 0.0,
            "skipped_rows": float(skipped_rows),
            "mae": np.nan,
            "rmse": np.nan,
            "log_rmse": np.nan,
            "wape": np.nan,
            "r2": np.nan,
        }

    y_mean = sums["y_sum"] / n_rows
    total_sum_squares = sums["y_squared_sum"] - n_rows * y_mean * y_mean
    r2 = (
        1.0 - sums["squared_error_sum"] / total_sum_squares
        if total_sum_squares > 0
        else np.nan
    )

    return {
        "model_key": model_key,
        "frazione": frazione,
        "cluster_label": cluster_label,
        "split": split_name,
        "rows": float(n_rows),
        "skipped_rows": float(skipped_rows),
        "mae": sums["abs_error_sum"] / n_rows,
        "rmse": float(np.sqrt(sums["squared_error_sum"] / n_rows)),
        "log_rmse": float(np.sqrt(sums["log_squared_error_sum"] / n_rows)),
        "wape": (
            sums["abs_error_sum"] / sums["y_abs_sum"]
            if sums["y_abs_sum"] != 0
            else np.nan
        ),
        "r2": float(r2),
    }


def lag_state_before_split(
    con: duckdb.DuckDBPyConnection,
    group: GroupSpec,
    split_name: str,
    id_col: str,
) -> LagState:
    split_start_rank, split_start_date = con.execute(f"""
        SELECT min(__time_rank), min(data_dt)
        FROM {split_name}_data;
    """).fetchone()
    state = LagState()
    if split_start_rank is None:
        return state

    history = con.execute(f"""
        WITH current_ids AS (
            SELECT DISTINCT {sql_ident(id_col)} AS id_value
            FROM {split_name}_data
            WHERE {group_where(group)}
        ),
        past AS (
            SELECT
                f.{sql_ident(id_col)} AS id_value,
                f.__group_frazione AS fraction_value,
                f.data_dt,
                f.__row_id,
                f.target_value
            FROM full_data f
            JOIN current_ids c
              ON f.{sql_ident(id_col)} = c.id_value
            WHERE f.__time_rank < {int(split_start_rank)}
              AND f.__group_frazione = {sql_lit(group.frazione)}
        ),
        selected AS (
            SELECT *
            FROM past
            WHERE data_dt >= DATE {sql_lit(str(split_start_date))} - INTERVAL 90 DAY
            UNION
            SELECT *
            FROM past
            QUALIFY row_number() OVER (
                PARTITION BY id_value
                ORDER BY data_dt DESC, __row_id DESC
            ) = 1
            UNION
            SELECT *
            FROM past
            WHERE target_value > 0
            QUALIFY row_number() OVER (
                PARTITION BY id_value
                ORDER BY data_dt DESC, __row_id DESC
            ) = 1
        )
        SELECT id_value, fraction_value, data_dt, target_value
        FROM selected
        ORDER BY id_value, data_dt, __row_id;
    """).fetchall()
    for civic, fraction, event_date, value in history:
        state.add(civic, fraction, event_date, value)
    return state


def recursive_group_predictions(
    con: duckdb.DuckDBPyConnection,
    model: Regressor,
    group: GroupSpec,
    split_name: str,
    id_col: str,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
    target_transform: str,
    prediction_upper_bound: float,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    columns = list(
        dict.fromkeys(
            [
                "__time_rank",
                id_col,
                "data_dt",
                "frazione",
                *feature_cols,
                "target_value",
            ]
        )
    )
    frame = fetch_group_rows(
        con, f"{split_name}_data", group, columns
    ).reset_index(drop=True)
    predictions = np.empty(len(frame), dtype=np.float64)
    if frame.empty:
        return frame, predictions, 0

    state = lag_state_before_split(con, group, split_name, id_col)
    batches = 0
    for event_date, positions in frame.groupby("data_dt", sort=True).indices.items():
        batch = frame.iloc[positions].copy()
        lag_rows = [
            state.features(civic, fraction, event_date)
            for civic, fraction in zip(
                batch[id_col], batch["frazione"], strict=True
            )
        ]
        for name in LAG_FEATURES:
            values = [row[name] for row in lag_rows]
            batch[name] = values
            frame.loc[positions, name] = values
        raw_predictions = prediction_to_raw(
            model,
            make_features(batch, numeric_cols, categorical_cols, numeric_stats),
            target_transform,
            prediction_upper_bound,
        )
        predictions[positions] = raw_predictions
        for civic, fraction, prediction in zip(
            batch[id_col], batch["frazione"], raw_predictions, strict=True
        ):
            state.add(civic, fraction, event_date, float(prediction))
        batches += 1
    return frame, predictions, batches


def evaluate_group_split(
    con: duckdb.DuckDBPyConnection,
    model: Regressor,
    group: GroupSpec,
    split_name: str,
    batch_size: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
    target_transform: str,
    id_col: str,
    prediction_upper_bound: float,
) -> dict[str, float]:
    if split_name != "train":
        frame, pred_raw, _ = recursive_group_predictions(
            con,
            model,
            group,
            split_name,
            id_col,
            feature_cols,
            numeric_cols,
            categorical_cols,
            numeric_stats,
            target_transform,
            prediction_upper_bound,
        )
        y_raw = make_target_raw(frame["target_value"])
        error = pred_raw - y_raw
        log_error = np.log1p(pred_raw) - np.log1p(y_raw)
        return {
            "rows": float(len(y_raw)),
            "abs_error_sum": float(np.abs(error).sum()),
            "squared_error_sum": float((error * error).sum()),
            "log_squared_error_sum": float((log_error * log_error).sum()),
            "y_sum": float(y_raw.sum()),
            "y_abs_sum": float(np.abs(y_raw).sum()),
            "y_squared_sum": float((y_raw * y_raw).sum()),
        }

    table_name = f"{split_name}_data"
    last_time_rank = 0
    sums = empty_metric_sums()
    where_sql = group_where(group)

    while True:
        batch = fetch_batch(
            con=con,
            table_name=table_name,
            last_time_rank=last_time_rank,
            batch_size=batch_size,
            cols=feature_cols + ["target_value"],
            where_sql=where_sql,
        )
        if batch.empty:
            break

        last_time_rank = int(batch["__time_rank"].max())
        x = make_features(batch, numeric_cols, categorical_cols, numeric_stats)
        y_log = make_target_log(batch["target_value"])
        y_raw = np.expm1(y_log)
        prediction = model.predict(x)
        if target_transform == "log1p":
            pred_raw = prediction_log_to_raw(prediction, prediction_upper_bound)
            log_error = np.log1p(pred_raw) - y_log
        else:
            pred_raw = clip_raw_predictions(prediction, prediction_upper_bound)
            log_error = np.log1p(pred_raw) - y_log

        error = pred_raw - y_raw

        n = len(y_raw)
        sums["rows"] += n
        sums["abs_error_sum"] += float(np.abs(error).sum())
        sums["squared_error_sum"] += float((error * error).sum())
        sums["log_squared_error_sum"] += float((log_error * log_error).sum())
        sums["y_sum"] += float(y_raw.sum())
        sums["y_abs_sum"] += float(np.abs(y_raw).sum())
        sums["y_squared_sum"] += float((y_raw * y_raw).sum())

    return sums


def add_metric_sums(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] += value


def export_predictions(
    con: duckdb.DuckDBPyConnection,
    trained_models: dict[str, dict[str, object]],
    output_path: Path,
    output_format: str,
    split_name: str,
    batch_size: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_transform: str,
    id_col: str,
) -> int:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pred_store (
            __time_rank BIGINT,
            model_key VARCHAR,
            pred_conferimento_t DOUBLE,
            lag_1_collection_t DOUBLE,
            rolling_7d_sum_t DOUBLE,
            rolling_30d_sum_t DOUBLE,
            rolling_90d_median_t DOUBLE,
            rolling_90d_count DOUBLE,
            days_since_last_nonzero_collection DOUBLE
        );
    """)

    table_name = f"{split_name}_data"
    batches = 0

    for model_key, info in trained_models.items():
        group = info["group"]
        model = info["model"]
        numeric_stats = info["numeric_stats"]
        prediction_upper_bound = info["prediction_upper_bound_t"]
        if split_name != "train":
            frame, predictions, recursive_batches = recursive_group_predictions(
                con,
                model,
                group,
                split_name,
                id_col,
                feature_cols,
                numeric_cols,
                categorical_cols,
                numeric_stats,
                target_transform,
                prediction_upper_bound,
            )
            pred_batch = pd.DataFrame(
                {
                    "__time_rank": frame["__time_rank"].to_numpy(),
                    "model_key": model_key,
                    "pred_conferimento_t": predictions,
                    **{name: frame[name].to_numpy() for name in LAG_FEATURES},
                }
            )
            con.register("pred_batch", pred_batch)
            con.execute("INSERT INTO pred_store SELECT * FROM pred_batch;")
            con.unregister("pred_batch")
            batches += recursive_batches
            continue
        last_time_rank = 0

        while True:
            batch = fetch_batch(
                con=con,
                table_name=table_name,
                last_time_rank=last_time_rank,
                batch_size=batch_size,
                cols=feature_cols + ["target_value"],
                where_sql=group_where(group),
            )
            if batch.empty:
                break

            last_time_rank = int(batch["__time_rank"].max())
            x = make_features(batch, numeric_cols, categorical_cols, numeric_stats)
            predictions = prediction_to_raw(
                model, x, target_transform, prediction_upper_bound
            )

            pred_batch = pd.DataFrame(
                {
                    "__time_rank": batch["__time_rank"].to_numpy(),
                    "model_key": model_key,
                    "pred_conferimento_t": predictions,
                    **{name: batch[name].to_numpy() for name in LAG_FEATURES},
                }
            )
            con.register("pred_batch", pred_batch)
            con.execute("INSERT INTO pred_store SELECT * FROM pred_batch;")
            con.unregister("pred_batch")
            batches += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT
                f.* EXCLUDE ({", ".join(LAG_FEATURES)}),
                p.model_key,
                p.pred_conferimento_t,
                {", ".join(f"p.{name}" for name in LAG_FEATURES)},
                p.pred_conferimento_t - f.target_value AS prediction_error,
                abs(p.pred_conferimento_t - f.target_value) AS absolute_error
            FROM {table_name} f
            LEFT JOIN pred_store p
            ON f.__time_rank = p.__time_rank
            ORDER BY f.__time_rank
        )
        TO {sql_lit(str(output_path))}
        {export_options(output_format)};
    """)

    return batches


def split_date_ranges(
    con: duckdb.DuckDBPyConnection,
) -> dict[str, tuple[str, str] | None]:
    ranges = {}
    for split_name in ["train", "validation", "test"]:
        result = con.execute(f"""
            SELECT min(data_dt), max(data_dt)
            FROM {split_name}_data;
        """).fetchone()
        if result[0] is None:
            ranges[split_name] = None
        else:
            ranges[split_name] = (str(result[0]), str(result[1]))
    return ranges


def save_model_bundle(
    trained_models: dict[str, dict[str, object]],
    args: argparse.Namespace,
    numeric_cols: list[str],
    categorical_cols: list[str],
    split_counts: dict[str, int],
    metrics: list[dict[str, float | str]],
    date_ranges: dict[str, tuple[str, str] | None],
) -> None:
    serializable_models = {}
    for model_key, info in trained_models.items():
        group = info["group"]
        model = info["model"]
        serializable_models[model_key] = {
            "model": model,
            "model_params": model.get_params(),
            "group_values": {
                "frazione": group.frazione,
                "cluster_label": group.cluster_label,
            },
            "train_rows": group.train_rows,
            "fit_rows": info["fit_rows"],
            "numeric_stats": info["numeric_stats"],
            "prediction_upper_bound_t": info["prediction_upper_bound_t"],
        }

    model_name = f"grouped_{args.regressor}"
    model_display_name = {
        "sgd": "Grouped SGDRegressor",
        "gpr": "Grouped GaussianProcessRegressor",
        "hist_gradient_boosting": "Grouped HistGradientBoostingRegressor",
    }[args.regressor]
    target_transform = (
        "identity" if args.regressor == "hist_gradient_boosting" else "log1p"
    )
    bundle = {
        "model_name": model_name,
        "model_display_name": model_display_name,
        "group_cols": GROUP_COLS,
        "models": serializable_models,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "categorical_levels": CATEGORICAL_LEVELS,
        "feature_names": feature_names(numeric_cols, categorical_cols),
        "lag_features": list(LAG_FEATURES),
        "held_out_lag_strategy": "recursive_predictions",
        "target_transform": target_transform,
        "inverse_target_transform": (
            "clipped_at_zero"
            if target_transform == "identity"
            else "expm1_clipped_at_zero"
        ),
        "target_col": args.target_col,
        "date_col": args.date_col,
        "id_col": args.id_col,
        "split_ratios": args.split_ratios,
        "split_dates": {
            "train_end": (
                args.train_end.isoformat() if args.train_end is not None else None
            ),
            "validation_end": (
                args.validation_end.isoformat()
                if args.validation_end is not None
                else None
            ),
            "test_end": (
                args.test_end.isoformat() if args.test_end is not None else None
            ),
        },
        "split_counts": split_counts,
        "date_ranges": date_ranges,
        "metrics": metrics,
        "min_train_rows": args.min_train_rows,
    }
    if args.regressor == "gpr":
        bundle.update(
            gpr_max_train_rows=args.gpr_max_train_rows,
            sampling="seeded row from each chronological ntile bucket",
        )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one regressor per frazione/cluster_label pair using "
            "chronological train/validation/test splits."
        )
    )
    parser.add_argument(
        "--regressor",
        choices=["sgd", "gpr", "hist_gradient_boosting"],
        default="sgd",
    )
    parser.add_argument("--gpr-max-train-rows", type=int, default=500)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("conferimenti_giornalieri_classificati.parquet"),
        help="CSV or parquet training dataset.",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "csv", "parquet"],
        default="auto",
    )
    parser.add_argument(
        "--split-ratios",
        type=parse_split_ratios,
        default=(0.70, 0.20, 0.10),
        metavar="TRAIN,VALIDATION,TEST",
        help="Chronological split ratios. Accepts 0.7,0.2,0.1 or 70,20,10.",
    )
    parser.add_argument(
        "--train-end",
        type=parse_iso_date,
        help="Inclusive training end date for a fixed chronological split.",
    )
    parser.add_argument(
        "--validation-end",
        type=parse_iso_date,
        help="Inclusive validation end date for a fixed chronological split.",
    )
    parser.add_argument(
        "--test-end",
        type=parse_iso_date,
        help="Optional inclusive test end date for a fixed chronological split.",
    )
    parser.add_argument("--id-col", default="id_civico")
    parser.add_argument("--date-col", default="data")
    parser.add_argument("--year-col", default="anno_produzione")
    parser.add_argument("--target-col", default="conferimento_t")
    parser.add_argument("--date-format", default="%Y-%m-%d")
    parser.add_argument("--exclude-giorni-accumulati", action="store_true")
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="Optional row cap for quick smoke tests. 0 means use all rows.",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=1,
        help="Skip frazione/cluster_label pairs with fewer training rows.",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--eta0", type=float, default=0.01)
    parser.add_argument("--penalty", type=parse_penalty, default="l2")
    parser.add_argument("--l1-ratio", type=float, default=0.15)
    parser.add_argument(
        "--learning-rate",
        choices=["constant", "optimal", "invscaling", "adaptive"],
        default="invscaling",
    )
    parser.add_argument("--power-t", type=float, default=0.25)
    parser.add_argument("--average", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--model-output",
        type=Path,
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
    )
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument(
        "--predictions-split",
        choices=["train", "validation", "test"],
        default="test",
    )
    parser.add_argument(
        "--predictions-format",
        choices=["csv", "parquet"],
        default="parquet",
    )
    parser.add_argument("--duckdb-memory-limit", default=None)
    parser.add_argument("--duckdb-threads", type=int, default=None)
    parser.add_argument("--duckdb-temp-directory", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.min_train_rows <= 0:
        raise ValueError("--min-train-rows must be greater than zero.")
    if args.gpr_max_train_rows <= 0:
        raise ValueError("--gpr-max-train-rows must be greater than zero.")

    artifact_stem = f"grouped_{args.regressor}_time_series"
    if args.model_output is None:
        args.model_output = Path("models") / f"{artifact_stem}.joblib"
    if args.metrics_output is None:
        args.metrics_output = Path(f"{artifact_stem}_metrics.csv")

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false;")
    if args.duckdb_memory_limit:
        con.execute(f"PRAGMA memory_limit={sql_lit(args.duckdb_memory_limit)};")
    if args.duckdb_threads:
        con.execute(f"PRAGMA threads={int(args.duckdb_threads)};")
    if args.duckdb_temp_directory:
        args.duckdb_temp_directory.mkdir(parents=True, exist_ok=True)
        con.execute(
            f"PRAGMA temp_directory={sql_lit(str(args.duckdb_temp_directory))};"
        )

    numeric_cols, categorical_cols, feature_cols = get_feature_columns(
        args.exclude_giorni_accumulati
    )

    print(f"loading {args.input}")
    create_full_data_table(
        con=con,
        input_path=args.input,
        input_format=args.input_format,
        id_col=args.id_col,
        date_col=args.date_col,
        year_col=args.year_col,
        target_col=args.target_col,
        date_format=args.date_format,
        limit_rows=args.limit_rows,
    )

    split_counts = create_split_tables(
        con,
        args.split_ratios,
        train_end=args.train_end,
        validation_end=args.validation_end,
        test_end=args.test_end,
    )
    date_ranges = split_date_ranges(con)
    print("split counts:", json.dumps(split_counts, sort_keys=True))
    print("date ranges:", json.dumps(date_ranges, sort_keys=True))

    groups = discover_groups(con, args.min_train_rows)
    print(f"training {len(groups)} frazione/cluster_label models")
    target_transform = (
        "identity" if args.regressor == "hist_gradient_boosting" else "log1p"
    )

    trained_models: dict[str, dict[str, object]] = {}
    for idx, group in enumerate(groups, start=1):
        print(f"[{idx}/{len(groups)}] {group.key}; train_rows={group.train_rows}")
        where_sql = group_where(group)
        numeric_stats = compute_numeric_stats(
            con=con,
            table_name="train_data",
            numeric_cols=numeric_cols,
            where_sql=where_sql,
        )
        prediction_upper_bound = float(
            con.execute(f"""
                SELECT max(target_value)
                FROM train_data
                WHERE {where_sql};
            """).fetchone()[0]
        )
        if not np.isfinite(prediction_upper_bound) or prediction_upper_bound < 0:
            raise ValueError(f"Invalid training target maximum for {group.key}.")
        model = build_model(args, idx - 1)
        if isinstance(model, GaussianProcessRegressor):
            fit_rows = fit_gpr(
                con=con,
                model=model,
                group=group,
                max_rows=args.gpr_max_train_rows,
                feature_cols=feature_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                numeric_stats=numeric_stats,
                seed=args.seed + idx - 1,
            )
        elif isinstance(model, HistGradientBoostingRegressor):
            fit_rows = fit_hist_gradient_boosting(
                con=con,
                model=model,
                group=group,
                feature_cols=feature_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                numeric_stats=numeric_stats,
            )
        else:
            fit_sgd(
                con=con,
                model=model,
                group=group,
                epochs=args.epochs,
                batch_size=args.batch_size,
                feature_cols=feature_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                numeric_stats=numeric_stats,
                progress=args.progress,
            )
            fit_rows = group.train_rows
        trained_models[group.key] = {
            "group": group,
            "model": model,
            "numeric_stats": numeric_stats,
            "fit_rows": fit_rows,
            "prediction_upper_bound_t": prediction_upper_bound,
        }

    metrics: list[dict[str, float | str]] = []
    aggregate_sums = {
        "train": empty_metric_sums(),
        "validation": empty_metric_sums(),
        "test": empty_metric_sums(),
    }

    for model_key, info in trained_models.items():
        group = info["group"]
        model = info["model"]
        numeric_stats = info["numeric_stats"]
        prediction_upper_bound = info["prediction_upper_bound_t"]

        for split_name in ["train", "validation", "test"]:
            sums = evaluate_group_split(
                con=con,
                model=model,
                group=group,
                split_name=split_name,
                batch_size=args.batch_size,
                feature_cols=feature_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                numeric_stats=numeric_stats,
                target_transform=target_transform,
                id_col=args.id_col,
                prediction_upper_bound=prediction_upper_bound,
            )
            add_metric_sums(aggregate_sums[split_name], sums)
            metrics.append(
                metric_sums_to_row(
                    sums=sums,
                    split_name=split_name,
                    model_key=model_key,
                    frazione=group.frazione,
                    cluster_label=group.cluster_label,
                )
            )

    aggregate_metrics = []
    for split_name in ["train", "validation", "test"]:
        predicted_rows = int(aggregate_sums[split_name]["rows"])
        skipped_rows = split_counts[split_name] - predicted_rows
        row = metric_sums_to_row(
            sums=aggregate_sums[split_name],
            split_name=split_name,
            model_key="__all_trained_models__",
            frazione="__all__",
            cluster_label="__all__",
            skipped_rows=skipped_rows,
        )
        aggregate_metrics.append(row)
        metrics.append(row)

    metrics_df = pd.DataFrame(metrics)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.metrics_output, index=False)
    print(pd.DataFrame(aggregate_metrics).to_string(index=False))

    save_model_bundle(
        trained_models=trained_models,
        args=args,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        split_counts=split_counts,
        metrics=metrics,
        date_ranges=date_ranges,
    )
    print(f"saved grouped model bundle to {args.model_output}")
    print(f"saved metrics to {args.metrics_output}")

    if args.predictions_output:
        batches = export_predictions(
            con=con,
            trained_models=trained_models,
            output_path=args.predictions_output,
            output_format=args.predictions_format,
            split_name=args.predictions_split,
            batch_size=args.batch_size,
            feature_cols=feature_cols,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            target_transform=target_transform,
            id_col=args.id_col,
        )
        print(
            f"saved {args.predictions_split} predictions to "
            f"{args.predictions_output}; batches={batches}"
        )


if __name__ == "__main__":
    main()
