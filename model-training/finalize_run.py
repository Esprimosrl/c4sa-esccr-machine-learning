#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

import duckdb
import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_parquet(path: Path) -> pd.DataFrame:
    return duckdb.connect().execute(
        "SELECT * FROM read_parquet(?)", [str(path)]
    ).df()


def selected_arima_predictions(
    frame: pd.DataFrame, metrics_path: Path
) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    validation = metrics[
        (metrics["split"] == "validation")
        & metrics["model"].isin(["sarima", "sarimax", "seasonal_naive_7d"])
        & (metrics["frazione"] != "__macro__")
    ].copy()
    validation["wape"] = pd.to_numeric(validation["wape"], errors="coerce")
    validation["rmse"] = pd.to_numeric(validation["rmse"], errors="coerce")
    validation = validation[
        np.isfinite(validation["wape"]) & np.isfinite(validation["rmse"])
    ]
    selected = (
        validation.sort_values(
            ["frazione", "wape", "rmse", "model"], kind="stable"
        )
        .drop_duplicates("frazione")
        .set_index("frazione")["model"]
        .to_dict()
    )
    pieces = []
    for fraction, rows in frame.groupby("frazione", sort=True):
        model = selected.get(fraction)
        if model is None:
            continue
        prefix = {
            "sarima": "pred_civic_sarima",
            "sarimax": "pred_civic_sarimax",
            "seasonal_naive_7d": "pred_civic_seasonal_naive_7d",
        }[model]
        piece = rows.copy()
        piece["prediction_t"] = piece[f"{prefix}_t"]
        piece["prediction_lower_t"] = piece[f"{prefix}_lower_t"]
        piece["prediction_upper_t"] = piece[f"{prefix}_upper_t"]
        piece["model"] = model
        piece["forecast_method"] = model
        pieces.append(piece)
    if not pieces:
        raise ValueError("No validation-selected ARIMA civic predictions.")
    return pd.concat(pieces, ignore_index=True)


def canonicalize(
    frame: pd.DataFrame,
    pipeline: str,
    model_name: str,
    metrics_path: Path | None,
) -> pd.DataFrame:
    result = frame.copy()
    if "pred_civic_sarima_t" in result:
        if metrics_path is None:
            raise ValueError("ARIMA civic predictions require metrics.")
        result = selected_arima_predictions(result, metrics_path)
    elif "selected_prediction_t" in result:
        result["prediction_t"] = result["selected_prediction_t"]
        result["model"] = result["selected_model"].astype("string")
        result["forecast_method"] = result["model"]
    elif "pred_conferimento_t" in result:
        result["prediction_t"] = result["pred_conferimento_t"]
        result["model"] = model_name
        if "forecast_method" not in result:
            result["forecast_method"] = model_name
    elif "pred_group_total_t" in result:
        result["prediction_t"] = result["pred_group_total_t"]
        result["model"] = model_name
        if "forecast_method" not in result:
            result["forecast_method"] = model_name
    else:
        raise ValueError("Unsupported native prediction schema.")

    aliases = {"data_dt": "data"}
    result = result.rename(
        columns={source: target for source, target in aliases.items() if source in result}
    )
    for column, default in {
        "id_civico": pd.NA,
        "prediction_lower_t": np.nan,
        "prediction_upper_t": np.nan,
        "zona_raccolta": pd.NA,
        "cluster_label": pd.NA,
    }.items():
        if column not in result:
            result[column] = default
    result["pipeline"] = pipeline
    result["data"] = pd.to_datetime(result["data"], errors="raise").dt.date
    result["prediction_t"] = pd.to_numeric(
        result["prediction_t"], errors="raise"
    )
    if (
        result["frazione"].isna().any()
        or (~np.isfinite(result["prediction_t"])).any()
        or (result["prediction_t"] < 0).any()
    ):
        raise ValueError("Canonical predictions contain invalid values.")
    return result[CANONICAL_COLUMNS].sort_values(
        ["data", "zona_raccolta", "frazione", "cluster_label", "id_civico"],
        kind="stable",
        na_position="last",
    )


def write_parquet(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.register("frame", frame)
    con.execute(
        "COPY frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(output)]
    )
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Write common pipeline artifacts.")
    parser.add_argument("--pipeline", choices=["legacy", "zone-fraction"], required=True)
    parser.add_argument("--source", choices=["file", "quantumleap"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--test-input", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, action="append", default=[])
    parser.add_argument("--metrics-path", type=Path, action="append", default=[])
    parser.add_argument("--input-path", type=Path, action="append", default=[])
    parser.add_argument("--future-path", type=Path)
    parser.add_argument("--future-input", type=Path)
    parser.add_argument("--future-output", type=Path)
    args = parser.parse_args()

    metrics = args.metrics_path[-1] if args.metrics_path else None
    canonical = canonicalize(
        read_parquet(args.test_input),
        args.pipeline,
        args.model_name,
        metrics,
    )
    write_parquet(canonical, args.test_output)

    future_path = args.future_path
    if args.future_input:
        if args.future_output is None:
            parser.error("--future-output is required with --future-input")
        canonical_future = canonicalize(
            read_parquet(args.future_input),
            args.pipeline,
            args.model_name,
            metrics,
        )
        write_parquet(canonical_future, args.future_output)
        future_path = args.future_output
    artifacts = [
        *args.model_path,
        *args.metrics_path,
        args.test_input,
        args.test_output,
    ]
    if future_path:
        if not future_path.is_file():
            raise FileNotFoundError(
                f"future predictions do not exist: {future_path}"
            )
        artifacts.append(future_path)
    manifest = {
        "schema_version": 1,
        "pipeline": args.pipeline,
        "source": args.source,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selected_model_policy": (
            "validation WAPE, RMSE, model name"
            if args.pipeline == "zone-fraction"
            else args.model_name
        ),
        "inputs": [
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path.resolve()),
            }
            for path in args.input_path
            if path.is_file()
        ],
        "predictions": {
            "test": str(args.test_output.resolve()),
            "future": str(future_path.resolve()) if future_path else None,
        },
        "artifacts": [
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path.resolve()),
            }
            for path in artifacts
            if path.is_file()
        ],
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
