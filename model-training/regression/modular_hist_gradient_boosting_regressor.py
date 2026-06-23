#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal
import argparse

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor


CATEGORICAL_LEVELS = {
    "zona_raccolta": ["1", "2", "3", "4"],
    "frazione": [
        "pulizia_stradale_recupero",
        "frazione_organica",
        "metallo",
        "rifiuti_cd",
        "altro_rd",
        "ingombranti_misti",
        "tessili",
        "selettiva",
        "carta_cartone",
        "vetro",
        "raee",
        "plastica",
        "legno",
    ],
    "zona_costruzione": [
        "A1",
        "A2",
        "A3Urb",
        "B1.2",
        "B3.4",
        "B3R",
        "B5",
        "Br",
        "C1",
        "C1PdF",
        "C2",
        "C2PdF",
        "C3",
        "C3PdF",
        "CI",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D8",
        "DRF",
        "DRS",
        "E1",
        "E2",
        "E3",
        "F",
        "F1",
        "F1-F5",
        "F2",
        "F2-F3",
        "F234",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8.1",
        "F8.3",
        "F9",
    ],
}

TrainingMode = Literal["full_batch", "incremental"]


@dataclass(frozen=True)
class ModelSpec:
    """Metadata needed by the generic training loop."""

    name: str
    display_name: str
    training_mode: TrainingMode
    factory: Callable[[argparse.Namespace], object]


def parse_bool_or_auto(value: str) -> bool | str:
    normalized = value.strip().lower()

    if normalized == "auto":
        return "auto"

    if normalized in {"true", "yes", "1"}:
        return True

    if normalized in {"false", "no", "0"}:
        return False

    raise argparse.ArgumentTypeError("Expected one of: auto, true, false.")


def build_hist_gradient_boosting_regressor(
    args: argparse.Namespace,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        max_depth=args.hist_max_depth,
        min_samples_leaf=args.hist_min_samples_leaf,
        l2_regularization=args.l2_regularization,
        max_bins=args.max_bins,
        early_stopping=args.early_stopping,
        validation_fraction=args.validation_fraction,
        n_iter_no_change=args.n_iter_no_change,
        tol=args.tol,
        random_state=args.seed,
    )


def build_gradient_boosting_regressor(
    args: argparse.Namespace,
) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        subsample=args.subsample,
        random_state=args.seed,
    )


def build_sgd_regressor(args: argparse.Namespace) -> SGDRegressor:
    return SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=args.alpha,
        learning_rate="invscaling",
        eta0=args.eta0,
        max_iter=1,
        tol=None,
        random_state=args.seed,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "hist_gradient_boosting": ModelSpec(
        name="hist_gradient_boosting",
        display_name="HistGradientBoostingRegressor",
        training_mode="full_batch",
        factory=build_hist_gradient_boosting_regressor,
    ),
    "gradient_boosting": ModelSpec(
        name="gradient_boosting",
        display_name="GradientBoostingRegressor",
        training_mode="full_batch",
        factory=build_gradient_boosting_regressor,
    ),
    "sgd": ModelSpec(
        name="sgd",
        display_name="SGDRegressor",
        training_mode="incremental",
        factory=build_sgd_regressor,
    ),
}


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
    raise ValueError("Could not detect input format. Use csv or parquet.")


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

    raise ValueError("Unsupported input format.")


def export_options(output_format: str) -> str:
    if output_format == "parquet":
        return "(FORMAT parquet)"
    if output_format == "csv":
        return "(FORMAT csv, HEADER)"
    raise ValueError("Unsupported output format.")


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


def create_standard_table(
    con: duckdb.DuckDBPyConnection,
    raw_table: str,
    out_table: str,
    id_col: str,
    date_col: str,
    year_col: str,
    target_col: str,
    date_format: str,
    require_target: bool,
) -> None:
    cols = table_columns(con, raw_table)

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
        target_expr = "NULL::DOUBLE"

    target_filter = "AND target_value IS NOT NULL" if require_target else ""

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {out_table} AS
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
                {categorical_expr("frazione", cols)}

            FROM {raw_table}
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

                sin(2 * pi() * CAST(strftime(data_dt, '%j') AS DOUBLE) / 366.0) AS season_sin,
                cos(2 * pi() * CAST(strftime(data_dt, '%j') AS DOUBLE) / 366.0) AS season_cos
            FROM typed
        )

        SELECT *
        FROM enriched
        WHERE {sql_ident(id_col)} IS NOT NULL
          AND data_dt IS NOT NULL
          {target_filter};
    """)


def compute_numeric_stats(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    numeric_cols: list[str],
) -> dict[str, tuple[float, float]]:
    stats = {}

    for col in numeric_cols:
        mean_value, std_value = con.execute(f"""
            SELECT
                avg({sql_ident(col)}) AS mean_value,
                stddev_pop({sql_ident(col)}) AS std_value
            FROM {table_name}
            WHERE {sql_ident(col)} IS NOT NULL;
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
    last_row_id: int,
    batch_size: int,
    cols: list[str],
) -> pd.DataFrame:
    selected_cols = ", ".join(sql_ident(c) for c in ["__row_id"] + cols)

    return con.execute(f"""
        SELECT {selected_cols}
        FROM {table_name}
        WHERE __row_id > {last_row_id}
        ORDER BY __row_id
        LIMIT {batch_size};
    """).df()


def normalize_category_values(series: pd.Series, col: str) -> pd.Series:
    values = series.astype("string").str.strip()

    if col == "zona_raccolta":
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
    return np.maximum(pred, 0)


def make_progress_bar(enabled: bool, total: int, desc: str):
    """Create a tqdm progress bar only when requested."""
    if not enabled:
        return None

    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "--progress requires tqdm. Install it with: pip install tqdm"
        ) from exc

    return tqdm(total=total, desc=desc, unit="rows")


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

    categorical_cols = [
        "zona_raccolta",
        "frazione",
    ]

    feature_cols = numeric_cols + categorical_cols
    return numeric_cols, categorical_cols, feature_cols


def load_training_matrix(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    batch_size: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
    progress: bool,
    total_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one in-memory training matrix.

    This is required by GradientBoostingRegressor because it does not support
    partial_fit. For very large datasets, use --model sgd or add a model whose
    spec uses training_mode='incremental'.
    """
    last_row_id = 0
    x_blocks = []
    y_blocks = []

    progress_bar = make_progress_bar(
        enabled=progress,
        total=total_rows,
        desc="Loading training matrix",
    )

    while True:
        batch = fetch_batch(
            con=con,
            table_name=table_name,
            last_row_id=last_row_id,
            batch_size=batch_size,
            cols=feature_cols + ["target_value"],
        )

        if batch.empty:
            break

        last_row_id = int(batch["__row_id"].max())

        target = pd.to_numeric(batch["target_value"], errors="coerce")
        valid_mask = target.notna()

        if valid_mask.any():
            valid_batch = batch.loc[valid_mask].copy()
            x_blocks.append(
                make_features(
                    df=valid_batch,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    numeric_stats=numeric_stats,
                )
            )
            y_blocks.append(make_target_log(valid_batch["target_value"]))

        if progress_bar is not None:
            progress_bar.update(len(batch))

    if progress_bar is not None:
        progress_bar.close()

    if not x_blocks:
        raise ValueError("No valid training rows were found.")

    return np.vstack(x_blocks), np.concatenate(y_blocks)


def evaluate_arrays(
    model: object, x: np.ndarray, y_log: np.ndarray
) -> dict[str, float]:
    y_raw = np.expm1(y_log)
    pred_log = model.predict(x)
    pred_raw = prediction_log_to_raw(pred_log)

    error = pred_raw - y_raw
    log_error = pred_log - y_log

    return {
        "rows": float(len(y_raw)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "log_rmse": float(np.sqrt(np.mean(log_error * log_error))),
    }


def evaluate_model_on_table(
    con: duckdb.DuckDBPyConnection,
    model: object,
    table_name: str,
    batch_size: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
) -> dict[str, float] | None:
    """
    Stream a table in batches and compute metrics without loading it fully in RAM.
    Metrics are computed on the original conferimento_kg scale.
    """
    last_row_id = 0
    n_rows = 0
    abs_error_sum = 0.0
    squared_error_sum = 0.0
    log_squared_error_sum = 0.0

    while True:
        batch = fetch_batch(
            con=con,
            table_name=table_name,
            last_row_id=last_row_id,
            batch_size=batch_size,
            cols=feature_cols + ["target_value"],
        )

        if batch.empty:
            break

        last_row_id = int(batch["__row_id"].max())

        y = pd.to_numeric(batch["target_value"], errors="coerce")
        valid_mask = y.notna()

        if not valid_mask.any():
            continue

        batch = batch.loc[valid_mask].copy()
        y_log = make_target_log(batch["target_value"])
        y_raw = np.expm1(y_log)

        x = make_features(
            df=batch,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            numeric_stats=numeric_stats,
        )

        pred_log = model.predict(x)
        pred_raw = prediction_log_to_raw(pred_log)

        error = pred_raw - y_raw
        log_error = pred_log - y_log

        n = len(y_raw)
        n_rows += n
        abs_error_sum += float(np.abs(error).sum())
        squared_error_sum += float((error * error).sum())
        log_squared_error_sum += float((log_error * log_error).sum())

    if n_rows == 0:
        return None

    return {
        "rows": float(n_rows),
        "mae": abs_error_sum / n_rows,
        "rmse": float(np.sqrt(squared_error_sum / n_rows)),
        "log_rmse": float(np.sqrt(log_squared_error_sum / n_rows)),
    }


def build_staged_learning_history(
    model: object,
    x: np.ndarray,
    y_log: np.ndarray,
) -> list[dict[str, float]]:
    """Return per-stage training metrics for estimators exposing staged_predict."""
    if not hasattr(model, "staged_predict"):
        return []

    y_raw = np.expm1(y_log)
    history = []

    for stage, pred_log in enumerate(model.staged_predict(x), start=1):
        pred_raw = prediction_log_to_raw(pred_log)
        error = pred_raw - y_raw
        log_error = pred_log - y_log

        history.append(
            {
                "stage": float(stage),
                "rows": float(len(y_raw)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error * error))),
                "log_rmse": float(np.sqrt(np.mean(log_error * log_error))),
            }
        )

    return history


def save_learning_outputs(
    learning_history: list[dict[str, float]],
    plot_output: str,
    history_output: str,
) -> None:
    """Save learning metrics as CSV and PNG only when --learning-plot is used."""
    if not learning_history:
        return

    history_df = pd.DataFrame(learning_history)

    history_path = Path(history_output)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_path, index=False)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "--learning-plot requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    x_col = "stage" if "stage" in history_df.columns else "epoch"

    plot_path = Path(plot_output)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df[x_col], history_df["mae"], marker="o", label="Train MAE")
    ax.plot(history_df[x_col], history_df["rmse"], marker="o", label="Train RMSE")
    ax.set_xlabel(x_col.title())
    ax.set_ylabel("Error on original kg scale")
    ax.set_title("Learning curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("Saved learning history to:", history_path)
    print("Saved learning plot to:", plot_path)


def create_fit_table(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    out_table: str,
    max_train_rows: int,
    seed: int,
) -> str:
    """Optionally create a deterministic sampled training table for very large data."""
    if max_train_rows <= 0:
        return source_table

    source_rows = con.execute(f"SELECT count(*) FROM {source_table}").fetchone()[0]

    if source_rows <= max_train_rows:
        return source_table

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {out_table} AS
        SELECT *
        FROM {source_table}
        ORDER BY hash(__row_id + {int(seed)})
        LIMIT {int(max_train_rows)};
    """)

    return out_table


def fit_full_batch_model(
    con: duckdb.DuckDBPyConnection,
    model: object,
    train_table_name: str,
    train_rows: int,
    batch_size: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
    progress: bool,
    learning_plot: bool,
) -> tuple[object, list[dict[str, float]]]:
    x_train, y_train = load_training_matrix(
        con=con,
        table_name=train_table_name,
        batch_size=batch_size,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_stats=numeric_stats,
        progress=progress,
        total_rows=train_rows,
    )

    model.fit(x_train, y_train)

    metrics = evaluate_arrays(model, x_train, y_train)
    print(f"  train MAE: {metrics['mae']:.4f}, train RMSE: {metrics['rmse']:.4f}")

    learning_history = []
    if learning_plot:
        learning_history = build_staged_learning_history(model, x_train, y_train)
        if not learning_history:
            final_stage = float(
                getattr(model, "n_iter_", getattr(model, "max_iter", 1))
            )
            learning_history = [{"stage": final_stage, **metrics}]

    return model, learning_history


def fit_incremental_model(
    con: duckdb.DuckDBPyConnection,
    model: object,
    epochs: int,
    train_table_name: str,
    train_rows: int,
    batch_size: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
    progress: bool,
    learning_plot: bool,
) -> tuple[object, list[dict[str, float]]]:
    if not hasattr(model, "partial_fit"):
        raise TypeError("Incremental training requires a model with partial_fit.")

    first_batch = True
    learning_history = []

    for epoch in range(epochs):
        last_row_id = 0
        batch_count = 0
        progress_bar = make_progress_bar(
            enabled=progress,
            total=train_rows,
            desc=f"Training epoch {epoch + 1}/{epochs}",
        )

        while True:
            batch = fetch_batch(
                con=con,
                table_name=train_table_name,
                last_row_id=last_row_id,
                batch_size=batch_size,
                cols=feature_cols + ["target_value"],
            )

            if batch.empty:
                break

            last_row_id = int(batch["__row_id"].max())
            batch_count += 1

            x = make_features(
                df=batch,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                numeric_stats=numeric_stats,
            )
            y = make_target_log(batch["target_value"])

            model.partial_fit(x, y)
            first_batch = False

            if progress_bar is not None:
                progress_bar.update(len(batch))

        if progress_bar is not None:
            progress_bar.close()

        message = f"  finished epoch {epoch + 1}/{epochs}, batches: {batch_count}"

        if learning_plot:
            epoch_metrics = evaluate_model_on_table(
                con=con,
                model=model,
                table_name=train_table_name,
                batch_size=batch_size,
                feature_cols=feature_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                numeric_stats=numeric_stats,
            )

            if epoch_metrics is not None:
                epoch_metrics["epoch"] = float(epoch + 1)
                learning_history.append(epoch_metrics)
                message += (
                    f", train MAE: {epoch_metrics['mae']:.4f}, "
                    f"train RMSE: {epoch_metrics['rmse']:.4f}"
                )

        print(message)

    if first_batch:
        raise ValueError("No training batches were processed.")

    return model, learning_history


def fit_model(
    con: duckdb.DuckDBPyConnection,
    model_spec: ModelSpec,
    args: argparse.Namespace,
    train_table_name: str,
    train_rows: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
) -> tuple[object, list[dict[str, float]]]:
    model = model_spec.factory(args)

    if model_spec.training_mode == "full_batch":
        return fit_full_batch_model(
            con=con,
            model=model,
            train_table_name=train_table_name,
            train_rows=train_rows,
            batch_size=args.batch_size,
            feature_cols=feature_cols,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            numeric_stats=numeric_stats,
            progress=args.progress,
            learning_plot=args.learning_plot,
        )

    if model_spec.training_mode == "incremental":
        return fit_incremental_model(
            con=con,
            model=model,
            epochs=args.epochs,
            train_table_name=train_table_name,
            train_rows=train_rows,
            batch_size=args.batch_size,
            feature_cols=feature_cols,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            numeric_stats=numeric_stats,
            progress=args.progress,
            learning_plot=args.learning_plot,
        )

    raise ValueError(f"Unsupported training mode: {model_spec.training_mode}")


def predict_to_table(
    con: duckdb.DuckDBPyConnection,
    model: object,
    predict_rows: int,
    batch_size: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
    progress: bool,
) -> int:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pred_store (
            __row_id BIGINT,
            pred_conferimento_kg DOUBLE
        );
    """)

    last_row_id = 0
    prediction_batches = 0
    progress_bar = make_progress_bar(
        enabled=progress,
        total=predict_rows,
        desc="Predicting",
    )

    while True:
        batch = fetch_batch(
            con=con,
            table_name="predict_data",
            last_row_id=last_row_id,
            batch_size=batch_size,
            cols=feature_cols + ["target_value"],
        )

        if batch.empty:
            break

        last_row_id = int(batch["__row_id"].max())
        prediction_batches += 1

        x = make_features(
            df=batch,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            numeric_stats=numeric_stats,
        )

        pred_log = model.predict(x)
        pred = prediction_log_to_raw(pred_log)

        pred_batch = pd.DataFrame(
            {
                "__row_id": batch["__row_id"].to_numpy(),
                "pred_conferimento_kg": pred,
            }
        )

        con.register("pred_batch", pred_batch)
        con.execute("INSERT INTO pred_store SELECT * FROM pred_batch;")
        con.unregister("pred_batch")

        if progress_bar is not None:
            progress_bar.update(len(batch))

    if progress_bar is not None:
        progress_bar.close()

    return prediction_batches


def export_predictions(
    con: duckdb.DuckDBPyConnection,
    output_path: Path,
    output_format: str,
) -> tuple[int, float | None, float | None]:
    metrics = con.execute("""
        SELECT
            count(*) AS evaluated_rows,
            avg(abs(s.pred_conferimento_kg - p.target_value)) AS mae,
            sqrt(avg(
                (s.pred_conferimento_kg - p.target_value)
                * (s.pred_conferimento_kg - p.target_value)
            )) AS rmse
        FROM predict_data p
        JOIN pred_store s
        ON p.__row_id = s.__row_id
        WHERE p.target_value IS NOT NULL;
    """).fetchone()

    con.execute(f"""
        COPY (
            SELECT
                p.*,
                s.pred_conferimento_kg,

                CASE
                    WHEN p.target_value IS NOT NULL
                    THEN s.pred_conferimento_kg - p.target_value
                    ELSE NULL
                END AS prediction_error,

                CASE
                    WHEN p.target_value IS NOT NULL
                    THEN abs(s.pred_conferimento_kg - p.target_value)
                    ELSE NULL
                END AS absolute_error

            FROM predict_data p
            JOIN pred_store s
            ON p.__row_id = s.__row_id
            ORDER BY p.__row_id
        )
        TO {sql_lit(str(output_path))}
        {export_options(output_format)};
    """)

    return metrics


def save_model_bundle(
    model: object,
    model_spec: ModelSpec,
    args: argparse.Namespace,
    numeric_stats: dict[str, tuple[float, float]],
    numeric_cols: list[str],
    categorical_cols: list[str],
    feature_cols: list[str],
    learning_history: list[dict[str, float]],
) -> None:
    joblib.dump(
        {
            "model": model,
            "model_name": model_spec.name,
            "model_display_name": model_spec.display_name,
            "model_params": model.get_params() if hasattr(model, "get_params") else {},
            "numeric_stats": numeric_stats,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "categorical_levels": CATEGORICAL_LEVELS,
            "feature_cols": feature_cols,
            "learning_history": learning_history,
            "target_transform": "log1p",
            "inverse_target_transform": "expm1_clipped_at_zero",
            "target_col": args.target_col,
            "date_col": args.date_col,
            "id_col": args.id_col,
        },
        args.model_output,
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-input", required=True)
    parser.add_argument(
        "--train-input-format", choices=["auto", "csv", "parquet"], default="auto"
    )

    parser.add_argument("--predict-input", required=True)
    parser.add_argument(
        "--predict-input-format", choices=["auto", "csv", "parquet"], default="auto"
    )

    parser.add_argument("--output", default="regressor_predictions.parquet")
    parser.add_argument(
        "--output-format", choices=["parquet", "csv"], default="parquet"
    )
    parser.add_argument("--model-output", default="regressor.joblib")

    parser.add_argument("--id-col", default="id_civico")
    parser.add_argument("--date-col", default="data")
    parser.add_argument("--year-col", default="anno_produzione")
    parser.add_argument("--target-col", default="conferimento_kg")
    parser.add_argument("--date-format", default="%d/%m/%Y")

    parser.add_argument("--batch-size", type=int, default=25000)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help=(
            "Optional cap for fitting rows. Use 0 to train on all rows. "
            "When capped, rows are deterministically sampled with --seed."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY),
        default="hist_gradient_boosting",
        help="Model to train. Add new models by registering another ModelSpec.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars during training and prediction. Requires tqdm.",
    )
    parser.add_argument(
        "--learning-plot",
        action="store_true",
        help=(
            "Save training metrics. Classic GradientBoostingRegressor uses staged "
            "predictions; HistGradientBoostingRegressor stores final-fit metrics; "
            "SGDRegressor uses per-epoch metrics."
        ),
    )
    parser.add_argument(
        "--learning-plot-output",
        default="learning_curve.png",
        help="PNG path used when --learning-plot is enabled.",
    )
    parser.add_argument(
        "--learning-history-output",
        default="learning_history.csv",
        help="CSV path for training metrics when --learning-plot is enabled.",
    )
    parser.add_argument(
        "--exclude-giorni-accumulati",
        action="store_true",
        help="Exclude giorni_accumulati from the model.",
    )


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    hist_gradient_boosting_group = parser.add_argument_group(
        "HistGradientBoostingRegressor"
    )
    hist_gradient_boosting_group.add_argument("--max-iter", type=int, default=300)
    hist_gradient_boosting_group.add_argument(
        "--learning-rate", type=float, default=0.05
    )
    hist_gradient_boosting_group.add_argument("--max-leaf-nodes", type=int, default=31)
    hist_gradient_boosting_group.add_argument(
        "--hist-max-depth", type=int, default=None
    )
    hist_gradient_boosting_group.add_argument(
        "--hist-min-samples-leaf", type=int, default=20
    )
    hist_gradient_boosting_group.add_argument(
        "--l2-regularization", type=float, default=0.0
    )
    hist_gradient_boosting_group.add_argument("--max-bins", type=int, default=255)
    hist_gradient_boosting_group.add_argument(
        "--early-stopping",
        type=parse_bool_or_auto,
        default="auto",
        help="Use auto, true, or false.",
    )
    hist_gradient_boosting_group.add_argument(
        "--validation-fraction", type=float, default=0.1
    )
    hist_gradient_boosting_group.add_argument(
        "--n-iter-no-change", type=int, default=10
    )
    hist_gradient_boosting_group.add_argument("--tol", type=float, default=1e-7)

    gradient_boosting_group = parser.add_argument_group("GradientBoostingRegressor")
    gradient_boosting_group.add_argument("--n-estimators", type=int, default=200)
    gradient_boosting_group.add_argument("--max-depth", type=int, default=3)
    gradient_boosting_group.add_argument("--min-samples-leaf", type=int, default=5)
    gradient_boosting_group.add_argument("--subsample", type=float, default=1.0)

    sgd_group = parser.add_argument_group("SGDRegressor")
    sgd_group.add_argument("--epochs", type=int, default=5)
    sgd_group.add_argument("--alpha", type=float, default=0.0001)
    sgd_group.add_argument("--eta0", type=float, default=0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a modular regressor and predict day-by-day production."
    )
    add_common_arguments(parser)
    add_model_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_spec = MODEL_REGISTRY[args.model]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE train_raw AS
        SELECT
            row_number() OVER () AS __row_id,
            *
        FROM {source_sql(Path(args.train_input), args.train_input_format)};
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE predict_raw AS
        SELECT
            row_number() OVER () AS __row_id,
            *
        FROM {source_sql(Path(args.predict_input), args.predict_input_format)};
    """)

    create_standard_table(
        con=con,
        raw_table="train_raw",
        out_table="train_data",
        id_col=args.id_col,
        date_col=args.date_col,
        year_col=args.year_col,
        target_col=args.target_col,
        date_format=args.date_format,
        require_target=True,
    )

    create_standard_table(
        con=con,
        raw_table="predict_raw",
        out_table="predict_data",
        id_col=args.id_col,
        date_col=args.date_col,
        year_col=args.year_col,
        target_col=args.target_col,
        date_format=args.date_format,
        require_target=False,
    )

    numeric_cols, categorical_cols, feature_cols = get_feature_columns(
        exclude_giorni_accumulati=args.exclude_giorni_accumulati,
    )

    numeric_stats = compute_numeric_stats(
        con=con,
        table_name="train_data",
        numeric_cols=numeric_cols,
    )

    train_rows = con.execute("SELECT count(*) FROM train_data").fetchone()[0]
    predict_rows = con.execute("SELECT count(*) FROM predict_data").fetchone()[0]

    fit_table_name = create_fit_table(
        con=con,
        source_table="train_data",
        out_table="train_fit_data",
        max_train_rows=args.max_train_rows,
        seed=args.seed,
    )
    fit_rows = con.execute(f"SELECT count(*) FROM {fit_table_name}").fetchone()[0]

    print(f"Training {model_spec.display_name}...")
    print(f"  training rows: {train_rows}")
    if fit_rows != train_rows:
        print(f"  fitting rows after sampling cap: {fit_rows}")
    print(f"  prediction rows: {predict_rows}")
    print(f"  batch size: {args.batch_size}")
    print(f"  training mode: {model_spec.training_mode}")

    if model_spec.training_mode == "full_batch":
        print(
            "  note: this model is loaded into memory for fitting because it "
            "does not support partial_fit"
        )
    else:
        print(f"  epochs: {args.epochs}")

    model, learning_history = fit_model(
        con=con,
        model_spec=model_spec,
        args=args,
        train_table_name=fit_table_name,
        train_rows=fit_rows,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_stats=numeric_stats,
    )

    if args.learning_plot:
        save_learning_outputs(
            learning_history=learning_history,
            plot_output=args.learning_plot_output,
            history_output=args.learning_history_output,
        )

    print("Predicting...")
    prediction_batches = predict_to_table(
        con=con,
        model=model,
        predict_rows=predict_rows,
        batch_size=args.batch_size,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_stats=numeric_stats,
        progress=args.progress,
    )
    print(f"  prediction batches: {prediction_batches}")

    metrics = export_predictions(
        con=con,
        output_path=output_path,
        output_format=args.output_format,
    )

    save_model_bundle(
        model=model,
        model_spec=model_spec,
        args=args,
        numeric_stats=numeric_stats,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        feature_cols=feature_cols,
        learning_history=learning_history,
    )

    print("\nSaved predictions to:", output_path)
    print("Saved model to:", args.model_output)

    if metrics[0] > 0:
        print("\nEvaluation on prediction file:")
        print(f"  evaluated rows: {metrics[0]}")
        print(f"  MAE:  {metrics[1]:.4f}")
        print(f"  RMSE: {metrics[2]:.4f}")

    print("\nFeatures used:")
    print("  numeric:", numeric_cols)
    print("  categorical one-hot:", categorical_cols)


if __name__ == "__main__":
    main()
