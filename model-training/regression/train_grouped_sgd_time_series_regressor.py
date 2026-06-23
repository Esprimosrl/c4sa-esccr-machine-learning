#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDRegressor


CATEGORICAL_LEVELS = {
    "zona_raccolta": ["1", "2", "3", "4"],
}

GROUP_COLS = ["frazione", "cluster_label"]


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


def get_feature_columns(
    exclude_giorni_accumulati: bool,
) -> tuple[list[str], list[str], list[str]]:
    numeric_cols = [
        "coord_x",
        "coord_y",
        "densita_locale",
        "giorni_accumulati",
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
        )
        SELECT
            *,
            row_number() OVER (ORDER BY data_dt, __row_id) AS __time_rank,
            count(*) OVER () AS __total_rows
        FROM eligible;
    """)


def create_split_tables(
    con: duckdb.DuckDBPyConnection,
    split_ratios: tuple[float, float, float],
) -> dict[str, int]:
    total_rows = con.execute("SELECT count(*) FROM full_data;").fetchone()[0]
    if total_rows < 3:
        raise ValueError("Need at least three target rows for train/validation/test.")

    train_ratio, validation_ratio, _ = split_ratios
    train_cutoff = max(1, int(total_rows * train_ratio))
    validation_cutoff = max(
        train_cutoff + 1, int(total_rows * (train_ratio + validation_ratio))
    )
    validation_cutoff = min(validation_cutoff, total_rows - 1)

    split_sql = {
        "train": f"__time_rank <= {train_cutoff}",
        "validation": (
            f"__time_rank > {train_cutoff} AND __time_rank <= {validation_cutoff}"
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


def make_target_log(series: pd.Series) -> np.ndarray:
    y = pd.to_numeric(series, errors="coerce")
    y = y.fillna(0).clip(lower=0)
    return np.log1p(y.to_numpy(dtype=np.float64))


def prediction_log_to_raw(pred_log: np.ndarray) -> np.ndarray:
    pred = np.expm1(pred_log)
    pred[~np.isfinite(pred)] = 0.0
    return np.maximum(pred, 0)


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


def build_model(args: argparse.Namespace, group_index: int) -> SGDRegressor:
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


def empty_metric_sums() -> dict[str, float]:
    return {
        "rows": 0.0,
        "abs_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "log_squared_error_sum": 0.0,
        "y_sum": 0.0,
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
        "r2": float(r2),
    }


def evaluate_group_split(
    con: duckdb.DuckDBPyConnection,
    model: SGDRegressor,
    group: GroupSpec,
    split_name: str,
    batch_size: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
) -> dict[str, float]:
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
        pred_log = model.predict(x)
        pred_raw = prediction_log_to_raw(pred_log)

        error = pred_raw - y_raw
        log_error = pred_log - y_log

        n = len(y_raw)
        sums["rows"] += n
        sums["abs_error_sum"] += float(np.abs(error).sum())
        sums["squared_error_sum"] += float((error * error).sum())
        sums["log_squared_error_sum"] += float((log_error * log_error).sum())
        sums["y_sum"] += float(y_raw.sum())
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
) -> int:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pred_store (
            __time_rank BIGINT,
            model_key VARCHAR,
            pred_conferimento_t DOUBLE
        );
    """)

    table_name = f"{split_name}_data"
    batches = 0

    for model_key, info in trained_models.items():
        group = info["group"]
        model = info["model"]
        numeric_stats = info["numeric_stats"]
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
            predictions = prediction_log_to_raw(model.predict(x))

            pred_batch = pd.DataFrame(
                {
                    "__time_rank": batch["__time_rank"].to_numpy(),
                    "model_key": model_key,
                    "pred_conferimento_t": predictions,
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
                f.*,
                p.model_key,
                p.pred_conferimento_t,
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
            "numeric_stats": info["numeric_stats"],
        }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": "grouped_sgd",
            "model_display_name": "Grouped SGDRegressor",
            "group_cols": GROUP_COLS,
            "models": serializable_models,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "categorical_levels": CATEGORICAL_LEVELS,
            "feature_names": feature_names(numeric_cols, categorical_cols),
            "target_transform": "log1p",
            "inverse_target_transform": "expm1_clipped_at_zero",
            "target_col": args.target_col,
            "date_col": args.date_col,
            "id_col": args.id_col,
            "split_ratios": args.split_ratios,
            "split_counts": split_counts,
            "date_ranges": date_ranges,
            "metrics": metrics,
            "min_train_rows": args.min_train_rows,
        },
        args.model_output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one SGDRegressor per frazione/cluster_label pair using "
            "chronological train/validation/test splits."
        )
    )
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
        default=Path("models/grouped_sgd_time_series.joblib"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("grouped_sgd_time_series_metrics.csv"),
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

    con = duckdb.connect()
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

    split_counts = create_split_tables(con, args.split_ratios)
    date_ranges = split_date_ranges(con)
    print("split counts:", json.dumps(split_counts, sort_keys=True))
    print("date ranges:", json.dumps(date_ranges, sort_keys=True))

    groups = discover_groups(con, args.min_train_rows)
    print(f"training {len(groups)} frazione/cluster_label models")

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
        model = build_model(args, idx - 1)
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
        trained_models[group.key] = {
            "group": group,
            "model": model,
            "numeric_stats": numeric_stats,
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
        )
        print(
            f"saved {args.predictions_split} predictions to "
            f"{args.predictions_output}; batches={batches}"
        )


if __name__ == "__main__":
    main()
