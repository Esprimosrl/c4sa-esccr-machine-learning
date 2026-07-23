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
from onnx import TensorProto, helper, numpy_helper
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType
from skl2onnx.operator_converters import random_forest as tree_converter


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


def add_target_postprocessing(
    model: onnx.ModelProto,
    target_transform: str,
    model_name: str,
    upper_bound: float | None,
) -> None:
    if len(model.graph.output) != 1:
        raise ValueError("Expected the converted regressor to have one output.")

    log_output = model.graph.output[0].name
    del model.graph.output[:]
    model.graph.initializer.extend(
        [numpy_helper.from_array(np.array([0.0], dtype=np.float64), "__post_zero")]
    )
    nonnegative_output = (
        "__post_nonnegative" if upper_bound is not None else "prediction_t"
    )
    if upper_bound is not None:
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(
                    np.array([upper_bound], dtype=np.float64), "__post_upper"
                )
            ]
        )
    if model_name != "grouped_hist_gradient_boosting" and target_transform == "log1p":
        model.graph.initializer.extend(
            [numpy_helper.from_array(np.array([1.0], dtype=np.float64), "__post_one")]
        )
    if model_name == "grouped_hist_gradient_boosting":
        model.graph.node.extend(
            [
                helper.make_node(
                    "Cast", [log_output], ["__post_double"], to=TensorProto.DOUBLE
                ),
                helper.make_node("Exp", ["__post_double"], ["__post_raw"]),
                helper.make_node(
                    "Max", ["__post_raw", "__post_zero"], [nonnegative_output]
                ),
            ]
        )
    elif target_transform == "log1p":
        model.graph.node.extend(
            [
                helper.make_node("Exp", [log_output], ["__post_exp"]),
                helper.make_node(
                    "Sub", ["__post_exp", "__post_one"], ["__post_raw"]
                ),
                helper.make_node(
                    "Max", ["__post_raw", "__post_zero"], [nonnegative_output]
                ),
            ]
        )
    elif target_transform == "identity":
        model.graph.node.extend(
            [helper.make_node("Max", [log_output, "__post_zero"], [nonnegative_output])]
        )
    else:
        raise ValueError(f"Unsupported target transform: {target_transform}")
    if upper_bound is not None:
        model.graph.node.extend(
            [
                helper.make_node(
                    "Min",
                    [nonnegative_output, "__post_upper"],
                    ["prediction_t"],
                )
            ]
        )
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(
                "prediction_t", TensorProto.DOUBLE, [None, 1]
            )
        ]
    )


def validate_conversion(
    session: ort.InferenceSession,
    estimator: object,
    means: np.ndarray,
    scales: np.ndarray,
    numeric_count: int,
    target_transform: str,
    upper_bound: float | None,
) -> None:
    rows = np.vstack([means, means + scales])
    rows[:, numeric_count:] = 0.0
    rows[0, numeric_count] = 1.0
    rows[1, numeric_count + 1] = 1.0

    normalized = ((rows - means) / scales).astype(np.float32)
    expected = estimator.predict(normalized)
    if target_transform == "log1p":
        expected = np.expm1(expected)
    expected = np.maximum(expected, 0.0)
    if upper_bound is not None:
        expected = np.minimum(expected, upper_bound)
    actual = session.run(["prediction_t"], {"features": rows})[0].reshape(-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def convert_pipeline(
    pipeline: Pipeline, name: str, feature_count: int
) -> onnx.ModelProto:
    original = tree_converter.add_tree_to_attribute_pairs_hist_gradient_boosting

    def compatible_hist_converter(*args, **kwargs):
        # skl2onnx 1.20 forwards numpy booleans that ONNX 1.22 rejects as ints.
        original(*args, **kwargs)
        attrs = args[0]
        attrs["nodes_missing_value_tracks_true"] = [
            int(value) for value in attrs["nodes_missing_value_tracks_true"]
        ]

    if isinstance(pipeline[-1], HistGradientBoostingRegressor):
        tree_converter.add_tree_to_attribute_pairs_hist_gradient_boosting = (
            compatible_hist_converter
        )
    try:
        return convert_sklearn(
            pipeline,
            name=name,
            initial_types=[("features", DoubleTensorType([None, feature_count]))],
            target_opset=TARGET_OPSET,
        )
    finally:
        tree_converter.add_tree_to_attribute_pairs_hist_gradient_boosting = original


def export_bundle(input_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use a new versioned directory."
        )

    bundle = joblib.load(input_path)
    model_name = bundle.get("model_name")
    if model_name not in {
        "grouped_sgd",
        "grouped_gpr",
        "grouped_hist_gradient_boosting",
    }:
        raise ValueError("Input is not a supported grouped regressor bundle.")
    target_transform = bundle.get("target_transform", "log1p")

    numeric_cols = list(bundle["numeric_cols"])
    feature_names = list(bundle["feature_names"])
    if feature_names[: len(numeric_cols)] != numeric_cols:
        raise ValueError("Bundle feature order does not start with numeric columns.")
    if len(feature_names) - len(numeric_cols) < 2:
        raise ValueError("Expected at least two categorical feature columns.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        models_dir = stage / "models"
        models_dir.mkdir()
        groups = []
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

        for index, (model_key, info) in enumerate(
            sorted(bundle["models"].items())
        ):
            estimator = info["model"]
            numeric_stats = info["numeric_stats"]
            upper_bound = info.get("prediction_upper_bound_t")
            if upper_bound is not None:
                upper_bound = float(upper_bound)
                if not np.isfinite(upper_bound) or upper_bound < 0:
                    raise ValueError(
                        f"Invalid prediction upper bound for {model_key}."
                    )
            categorical_count = len(feature_names) - len(numeric_cols)
            means = np.array(
                [numeric_stats[col][0] for col in numeric_cols]
                + [0.0] * categorical_count,
                dtype=np.float64,
            )
            scales = np.array(
                [numeric_stats[col][1] for col in numeric_cols]
                + [1.0] * categorical_count,
                dtype=np.float64,
            )
            if not np.isfinite(means).all() or not np.isfinite(scales).all():
                raise ValueError(f"Non-finite normalization values for {model_key}.")
            if (scales <= 0).any():
                raise ValueError(f"Non-positive normalization scale for {model_key}.")

            pipeline = Pipeline(
                [
                    ("scale", fitted_scaler(means, scales)),
                    ("model", estimator),
                ]
            )
            onnx_model = convert_pipeline(
                pipeline,
                f"{model_name}_{index:03d}",
                len(feature_names),
            )
            add_target_postprocessing(
                onnx_model, target_transform, model_name, upper_bound
            )
            onnx.checker.check_model(onnx_model)

            relative_path = Path("models") / f"{index:03d}.onnx"
            model_path = stage / relative_path
            onnx.save_model(onnx_model, model_path)
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            validate_conversion(
                session,
                estimator,
                means,
                scales,
                len(numeric_cols),
                target_transform,
                upper_bound,
            )
            del session

            group_values = info["group_values"]
            groups.append(
                {
                    "frazione": str(group_values["frazione"]),
                    "cluster_label": int(group_values["cluster_label"]),
                    "model": relative_path.as_posix(),
                    **(
                        {"prediction_upper_bound_t": float(upper_bound)}
                        if upper_bound is not None
                        else {}
                    ),
                }
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "model_version": file_sha256(input_path),
            "task": "grouped_regression",
            "input": {
                "name": "features",
                "dtype": "float64",
                "feature_names": feature_names,
            },
            "output": {
                "name": "prediction_t",
                "dtype": "float64",
                "unit": "t",
            },
            "groups": groups,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.chmod(0o755)
        models_dir.chmod(0o755)
        os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a grouped regressor joblib bundle as an ONNX package."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = export_bundle(args.input, args.output_dir)
    print(
        f"exported {len(manifest['groups'])} models to {args.output_dir}; "
        f"version={manifest['model_version']}"
    )


if __name__ == "__main__":
    main()
