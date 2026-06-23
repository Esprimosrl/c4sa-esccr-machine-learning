#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import os

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from kmeans_feature_tables import (
    KMEANS_EXTRA_CATEGORICAL_COLS,
    create_kmeans_feature_table,
)
from modular_hist_gradient_boosting_regressor import (
    CATEGORICAL_LEVELS,
    compute_numeric_stats,
    create_fit_table,
    export_options,
    fetch_batch,
    get_feature_columns,
    make_features,
    make_progress_bar,
    sql_ident,
    sql_lit,
    source_sql,
)


SYNTHETIC_DATE_NUMERIC_COLS = {
    "month",
    "day_of_month",
    "day_of_year",
    "day_of_week",
    "week_of_year",
    "quarter",
    "season_sin",
    "season_cos",
}


def ensure_writable_output_path(path: Path, label: str) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    if not os.access(parent, os.W_OK):
        raise PermissionError(
            f"Cannot write {label} to {path}. "
            f"The directory {parent} is not writable. "
            "Use a relative path like ./kmeans_assignments.parquet "
            "or an absolute path under a writable directory like /tmp."
        )


def parse_bool_or_auto(value: str) -> bool | str:
    normalized = value.strip().lower()

    if normalized == "auto":
        return "auto"

    if normalized in {"true", "yes", "1"}:
        return True

    if normalized in {"false", "no", "0"}:
        return False

    raise argparse.ArgumentTypeError("Expected one of: auto, true, false.")


def maybe_drop_synthetic_date_features(
    numeric_cols: list[str],
    categorical_cols: list[str],
    input_mode: str,
    ignore_synthetic_date_features: bool | str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    should_ignore = ignore_synthetic_date_features is True or (
        ignore_synthetic_date_features == "auto" and input_mode == "annual_wide"
    )

    if not should_ignore:
        return numeric_cols, categorical_cols, numeric_cols + categorical_cols, []

    kept_numeric_cols = [
        col for col in numeric_cols if col not in SYNTHETIC_DATE_NUMERIC_COLS
    ]
    dropped_cols = [col for col in numeric_cols if col in SYNTHETIC_DATE_NUMERIC_COLS]

    return kept_numeric_cols, categorical_cols, kept_numeric_cols + categorical_cols, dropped_cols


def adapt_feature_columns_for_input_mode(
    numeric_cols: list[str],
    categorical_cols: list[str],
    input_mode: str,
    annual_numeric_cols: tuple[str, ...],
) -> tuple[list[str], list[str], list[str]]:
    if input_mode != "annual_wide":
        return numeric_cols, categorical_cols, numeric_cols + categorical_cols

    annual_numeric = list(annual_numeric_cols)
    annual_categorical = ["zona_raccolta"]
    return numeric_cols + annual_numeric, annual_categorical, numeric_cols + annual_numeric + annual_categorical


def positive_or_zero_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Feature weights must be >= 0.")
    return parsed


def build_feature_weight_vector(
    numeric_cols: list[str],
    categorical_cols: list[str],
    annual_numeric_cols: tuple[str, ...],
    coord_weight: float,
    annual_kg_weight: float,
    other_numeric_weight: float,
    categorical_weight: float,
) -> np.ndarray:
    annual_numeric_set = set(annual_numeric_cols)
    weights = []

    for col in numeric_cols:
        if col in {"coord_x", "coord_y"}:
            weights.append(coord_weight)
        elif col in annual_numeric_set:
            weights.append(annual_kg_weight)
        else:
            weights.append(other_numeric_weight)

    for col in categorical_cols:
        levels = CATEGORICAL_LEVELS.get(col, [])
        weights.extend([categorical_weight] * len(levels))

    return np.asarray(weights, dtype=np.float32)


def apply_feature_weights(x: np.ndarray, feature_weights: np.ndarray) -> np.ndarray:
    if x.shape[1] != len(feature_weights):
        raise ValueError(
            f"Feature weight length mismatch: matrix has {x.shape[1]} columns, "
            f"weights have {len(feature_weights)} columns."
        )

    return x * feature_weights.reshape(1, -1)


def categorical_one_hot_selects(categorical_cols: list[str]) -> list[str]:
    selects = []

    for col in categorical_cols:
        if col not in CATEGORICAL_LEVELS:
            continue

        value_expr = f"TRIM(CAST(f.{sql_ident(col)} AS VARCHAR))"
        if col == "zona_raccolta":
            value_expr = (
                "CAST(TRY_CAST("
                f"TRIM(CAST(f.{sql_ident(col)} AS VARCHAR))"
                " AS INTEGER) AS VARCHAR)"
            )

        for level in CATEGORICAL_LEVELS[col]:
            selects.append(f"""
                CASE
                    WHEN COALESCE({value_expr}, '__MISSING__') = {sql_lit(level)}
                    THEN 1
                    ELSE 0
                END AS {sql_ident(f"{col}_{level}")}
            """)

    return selects


def load_feature_matrix(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    batch_size: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
    feature_weights: np.ndarray,
    progress: bool,
    total_rows: int,
) -> np.ndarray:
    last_row_id = 0
    x_blocks = []
    progress_bar = make_progress_bar(
        enabled=progress,
        total=total_rows,
        desc="Loading fitting matrix",
    )

    while True:
        batch = fetch_batch(
            con=con,
            table_name=table_name,
            last_row_id=last_row_id,
            batch_size=batch_size,
            cols=feature_cols,
        )

        if batch.empty:
            break

        last_row_id = int(batch["__row_id"].max())
        x_blocks.append(
            apply_feature_weights(
                make_features(
                    df=batch,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    numeric_stats=numeric_stats,
                ),
                feature_weights,
            )
        )

        if progress_bar is not None:
            progress_bar.update(len(batch))

    if progress_bar is not None:
        progress_bar.close()

    if not x_blocks:
        raise ValueError("No rows were found for K-means.")

    return np.vstack(x_blocks)


def write_cluster_assignments(
    con: duckdb.DuckDBPyConnection,
    model: KMeans,
    table_name: str,
    output_path: Path,
    output_format: str,
    batch_size: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_stats: dict[str, tuple[float, float]],
    feature_weights: np.ndarray,
    progress: bool,
    total_rows: int,
) -> int:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE cluster_store (
            __row_id BIGINT,
            cluster_label INTEGER,
            cluster_distance DOUBLE
        );
    """)

    last_row_id = 0
    batch_count = 0
    progress_bar = make_progress_bar(
        enabled=progress,
        total=total_rows,
        desc="Assigning clusters",
    )

    while True:
        batch = fetch_batch(
            con=con,
            table_name=table_name,
            last_row_id=last_row_id,
            batch_size=batch_size,
            cols=feature_cols,
        )

        if batch.empty:
            break

        last_row_id = int(batch["__row_id"].max())
        batch_count += 1
        x = apply_feature_weights(
            make_features(
                df=batch,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                numeric_stats=numeric_stats,
            ),
            feature_weights,
        )
        labels = model.predict(x)
        distances = model.transform(x).min(axis=1)

        cluster_batch = pd.DataFrame(
            {
                "__row_id": batch["__row_id"].to_numpy(),
                "cluster_label": labels.astype(np.int32),
                "cluster_distance": distances.astype(np.float64),
            }
        )
        con.register("cluster_batch", cluster_batch)
        con.execute("INSERT INTO cluster_store SELECT * FROM cluster_batch;")
        con.unregister("cluster_batch")

        if progress_bar is not None:
            progress_bar.update(len(batch))

    if progress_bar is not None:
        progress_bar.close()

    one_hot_selects = categorical_one_hot_selects(categorical_cols)
    extra_selects = one_hot_selects + [
        "c.cluster_label",
        "c.cluster_distance",
    ]
    extra_select_sql = ",\n                ".join(extra_selects)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT
                f.*,
                {extra_select_sql}
            FROM {table_name} f
            JOIN cluster_store c
            ON f.__row_id = c.__row_id
            ORDER BY f.__row_id
        )
        TO {sql_lit(str(output_path))}
        {export_options(output_format)};
    """)

    return batch_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train K-means using the regressor feature expansion."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-format", choices=["auto", "csv", "parquet"], default="auto")
    parser.add_argument("--output", default="kmeans_assignments.parquet")
    parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--model-output", default="kmeans_model.joblib")

    parser.add_argument("--id-col", default="id_civico")
    parser.add_argument("--date-col", default="data")
    parser.add_argument("--year-col", default="anno_produzione")
    parser.add_argument("--target-col", default="conferimento_kg")
    parser.add_argument("--date-format", default="%d/%m/%Y")
    parser.add_argument(
        "--annual-date-mm-dd",
        default="07-01",
        help=(
            "Synthetic MM-DD date used when reading annual wide CSV files "
            "that have anno but no date column."
        ),
    )
    parser.add_argument(
        "--ignore-synthetic-date-features",
        type=parse_bool_or_auto,
        default="auto",
        help=(
            "Use auto, true, or false. Auto drops synthetic date-derived features "
            "only for annual wide CSV inputs."
        ),
    )

    parser.add_argument("--n-clusters", type=int, default=4)
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
    parser.add_argument(
        "--max-assignment-rows",
        type=int,
        default=0,
        help=(
            "Optional cap for output cluster assignments. Use 0 to assign all rows. "
            "When capped, rows are deterministically sampled with --seed."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--coord-weight",
        type=positive_or_zero_float,
        default=1.0,
        help="Weight for coord_x and coord_y after standardization.",
    )
    parser.add_argument(
        "--annual-kg-weight",
        type=positive_or_zero_float,
        default=1.0,
        help="Weight for annual *_kg numeric columns after standardization.",
    )
    parser.add_argument(
        "--other-numeric-weight",
        type=positive_or_zero_float,
        default=1.0,
        help="Weight for non-coordinate, non-annual-kg numeric columns.",
    )
    parser.add_argument(
        "--categorical-weight",
        type=positive_or_zero_float,
        default=1.0,
        help="Weight for one-hot categorical columns.",
    )
    parser.add_argument(
        "--exclude-giorni-accumulati",
        action="store_true",
        help="Exclude giorni_accumulati from the feature matrix.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars while loading features and assigning clusters. Requires tqdm.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.n_clusters < 1:
        raise ValueError("--n-clusters must be a positive integer.")

    output_path = Path(args.output)
    model_path = Path(args.model_output)
    ensure_writable_output_path(output_path, "--output")
    ensure_writable_output_path(model_path, "--model-output")

    con = duckdb.connect()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE raw_data AS
        SELECT
            row_number() OVER () AS __row_id,
            *
        FROM {source_sql(Path(args.input), args.input_format)};
    """)

    feature_table_info = create_kmeans_feature_table(
        con=con,
        raw_table="raw_data",
        out_table="feature_data",
        id_col=args.id_col,
        date_col=args.date_col,
        year_col=args.year_col,
        target_col=args.target_col,
        date_format=args.date_format,
        annual_date_mm_dd=args.annual_date_mm_dd,
    )

    numeric_cols, categorical_cols, feature_cols = get_feature_columns(
        exclude_giorni_accumulati=args.exclude_giorni_accumulati,
    )
    numeric_cols, categorical_cols, feature_cols, dropped_date_cols = (
        maybe_drop_synthetic_date_features(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            input_mode=feature_table_info.mode,
            ignore_synthetic_date_features=args.ignore_synthetic_date_features,
        )
    )
    numeric_cols, categorical_cols, feature_cols = adapt_feature_columns_for_input_mode(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        input_mode=feature_table_info.mode,
        annual_numeric_cols=feature_table_info.annual_fraction_columns,
    )
    categorical_cols = categorical_cols + list(KMEANS_EXTRA_CATEGORICAL_COLS)
    feature_cols = numeric_cols + categorical_cols
    numeric_stats = compute_numeric_stats(
        con=con,
        table_name="feature_data",
        numeric_cols=numeric_cols,
    )
    feature_weights = build_feature_weight_vector(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        annual_numeric_cols=feature_table_info.annual_fraction_columns,
        coord_weight=args.coord_weight,
        annual_kg_weight=args.annual_kg_weight,
        other_numeric_weight=args.other_numeric_weight,
        categorical_weight=args.categorical_weight,
    )

    total_rows = con.execute("SELECT count(*) FROM feature_data").fetchone()[0]
    fit_table_name = create_fit_table(
        con=con,
        source_table="feature_data",
        out_table="kmeans_fit_data",
        max_train_rows=args.max_train_rows,
        seed=args.seed,
    )
    fit_rows = con.execute(f"SELECT count(*) FROM {fit_table_name}").fetchone()[0]
    assignment_table_name = create_fit_table(
        con=con,
        source_table="feature_data",
        out_table="kmeans_assignment_data",
        max_train_rows=args.max_assignment_rows,
        seed=args.seed,
    )
    assignment_rows = con.execute(
        f"SELECT count(*) FROM {assignment_table_name}"
    ).fetchone()[0]

    if args.n_clusters > fit_rows:
        raise ValueError("--n-clusters cannot be greater than the fitting row count.")

    print("Training KMeans...")
    print(f"  input mode: {feature_table_info.mode}")
    if feature_table_info.mode == "annual_wide":
        print(f"  annual synthetic date: {feature_table_info.annual_date_mm_dd}")
        print(
            "  annual fraction columns:",
            list(feature_table_info.annual_fraction_columns),
        )
        if feature_table_info.ignored_t_columns:
            print(
                "  ignored annual total columns:",
                list(feature_table_info.ignored_t_columns),
            )
    if dropped_date_cols:
        print("  ignored synthetic date features:", dropped_date_cols)
    print(f"  input rows: {total_rows}")
    if fit_rows != total_rows:
        print(f"  fitting rows after sampling cap: {fit_rows}")
    if assignment_rows != total_rows:
        print(f"  assignment rows after sampling cap: {assignment_rows}")
    print(f"  clusters: {args.n_clusters}")
    print("  numeric:", numeric_cols)
    print("  categorical one-hot:", categorical_cols)
    print(
        "  feature weights: "
        f"coord={args.coord_weight}, annual_kg={args.annual_kg_weight}, "
        f"other_numeric={args.other_numeric_weight}, "
        f"categorical={args.categorical_weight}"
    )

    x_fit = load_feature_matrix(
        con=con,
        table_name=fit_table_name,
        batch_size=args.batch_size,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_stats=numeric_stats,
        feature_weights=feature_weights,
        progress=args.progress,
        total_rows=fit_rows,
    )

    model = KMeans(
        n_clusters=args.n_clusters,
        init="k-means++",
        n_init=args.n_init,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.seed,
    )
    model.fit(x_fit)
    print(f"  inertia: {model.inertia_:.4f}")
    print(f"  iterations: {model.n_iter_}")

    joblib.dump(
        {
            "model": model,
            "model_name": "kmeans",
            "numeric_stats": numeric_stats,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "categorical_levels": CATEGORICAL_LEVELS,
            "feature_cols": feature_cols,
            "ignored_synthetic_date_features": dropped_date_cols,
            "annual_numeric_cols": feature_table_info.annual_fraction_columns,
            "feature_weights": feature_weights,
            "feature_weight_params": {
                "coord_weight": args.coord_weight,
                "annual_kg_weight": args.annual_kg_weight,
                "other_numeric_weight": args.other_numeric_weight,
                "categorical_weight": args.categorical_weight,
            },
            "id_col": args.id_col,
            "date_col": args.date_col,
            "target_col": args.target_col,
            "n_clusters": args.n_clusters,
            "inertia": float(model.inertia_),
            "fit_rows": int(len(x_fit)),
        },
        model_path,
    )

    batch_count = write_cluster_assignments(
        con=con,
        model=model,
        table_name=assignment_table_name,
        output_path=output_path,
        output_format=args.output_format,
        batch_size=args.batch_size,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_stats=numeric_stats,
        feature_weights=feature_weights,
        progress=args.progress,
        total_rows=assignment_rows,
    )

    print("Saved model to:", model_path)
    print("Saved cluster assignments to:", output_path)
    print(f"  assignment batches: {batch_count}")


if __name__ == "__main__":
    main()
