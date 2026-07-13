#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tempfile

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType


SCHEMA_VERSION = 1
TARGET_OPSET = 17


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fitted_scaler(means: np.ndarray, scales: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = means
    scaler.scale_ = scales
    scaler.var_ = scales**2
    scaler.n_features_in_ = len(means)
    scaler.n_samples_seen_ = 1
    return scaler


def model_schema(bundle: dict) -> tuple[list[str], dict[str, list[str]]]:
    numeric_names = list(bundle["numeric_cols"])
    categorical_levels = {
        name: list(bundle["categorical_levels"][name])
        for name in bundle["categorical_cols"]
    }
    return numeric_names, categorical_levels


def expanded_feature_names(
    numeric_names: list[str], categorical_levels: dict[str, list[str]]
) -> list[str]:
    return numeric_names + [
        f"{name}={level}"
        for name, levels in categorical_levels.items()
        for level in levels
    ]


def validate_conversion(
    session: ort.InferenceSession,
    estimator: object,
    means: np.ndarray,
    base_scales: np.ndarray,
    weights: np.ndarray,
    numeric_count: int,
) -> None:
    rows = np.vstack([means, means + base_scales])
    rows[:, numeric_count:] = 0.0
    rows[0, numeric_count] = 1.0
    rows[1, numeric_count + 1] = 1.0

    normalized = ((rows - means) / base_scales * weights).astype(np.float32)
    expected_labels = estimator.predict(normalized)
    expected_distances = estimator.transform(normalized).min(axis=1)
    actual_labels, scores = session.run(None, {"features": rows})

    np.testing.assert_array_equal(actual_labels, expected_labels)
    np.testing.assert_allclose(
        scores.min(axis=1), expected_distances, rtol=1e-5, atol=1e-4
    )


def export_bundle(input_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use a new versioned directory."
        )

    bundle = joblib.load(input_path)
    if bundle.get("model_name") != "kmeans":
        raise ValueError("Input is not a KMeans model bundle.")

    estimator = bundle["model"]
    numeric_names, categorical_levels = model_schema(bundle)
    feature_names = expanded_feature_names(numeric_names, categorical_levels)
    weights = np.asarray(bundle["feature_weights"], dtype=np.float64)
    if len(weights) != len(feature_names):
        raise ValueError("Feature weight count does not match the model input.")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("ONNX export requires positive finite feature weights.")
    if estimator.n_features_in_ != len(feature_names):
        raise ValueError("KMeans feature count does not match bundle metadata.")

    numeric_stats = bundle["numeric_stats"]
    categorical_count = len(feature_names) - len(numeric_names)
    means = np.array(
        [numeric_stats[name][0] for name in numeric_names]
        + [0.0] * categorical_count,
        dtype=np.float64,
    )
    base_scales = np.array(
        [numeric_stats[name][1] for name in numeric_names]
        + [1.0] * categorical_count,
        dtype=np.float64,
    )
    if not np.isfinite(means).all() or not np.isfinite(base_scales).all():
        raise ValueError("Normalization values must be finite.")
    if (base_scales <= 0).any():
        raise ValueError("Normalization scales must be positive.")

    pipeline = Pipeline(
        [
            ("scale_and_weight", fitted_scaler(means, base_scales / weights)),
            ("model", estimator),
        ]
    )
    onnx_model = convert_sklearn(
        pipeline,
        name="annual_kmeans",
        initial_types=[
            ("features", DoubleTensorType([None, len(feature_names)]))
        ],
        target_opset=TARGET_OPSET,
    )
    onnx.checker.check_model(onnx_model)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        model_path = stage / "model.onnx"
        onnx.save_model(onnx_model, model_path)
        session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if [(item.name, item.type) for item in inputs] != [
            ("features", "tensor(double)")
        ]:
            raise ValueError("Converted model has an unexpected input.")
        if [(item.name, item.type) for item in outputs] != [
            ("label", "tensor(int64)"),
            ("scores", "tensor(double)"),
        ]:
            raise ValueError("Converted model has unexpected outputs.")
        validate_conversion(
            session,
            estimator,
            means,
            base_scales,
            weights,
            len(numeric_names),
        )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "model_version": file_sha256(input_path),
            "task": "annual_clustering",
            "model": "model.onnx",
            "n_clusters": int(bundle["n_clusters"]),
            "input": {
                "name": "features",
                "dtype": "float64",
                "feature_names": feature_names,
                "numeric_names": numeric_names,
                "categorical_levels": categorical_levels,
                "numeric_defaults": {
                    "giorni_accumulati": float(
                        numeric_stats["giorni_accumulati"][0]
                    )
                },
            },
            "outputs": {
                "label": {"name": "label", "dtype": "int64"},
                "scores": {"name": "scores", "dtype": "float64"},
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        stage.chmod(0o755)
        os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a KMeans joblib bundle as an ONNX package."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = export_bundle(args.input, args.output_dir)
    print(
        f"exported {manifest['n_clusters']} clusters to {args.output_dir}; "
        f"version={manifest['model_version']}"
    )


if __name__ == "__main__":
    main()
