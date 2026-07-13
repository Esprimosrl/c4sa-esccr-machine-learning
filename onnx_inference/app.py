from __future__ import annotations

from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
import json
import math
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


SUPPORTED_FEATURES = {
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
    "zona_raccolta=1",
    "zona_raccolta=2",
    "zona_raccolta=3",
    "zona_raccolta=4",
}


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: date
    anno_produzione: int
    coord_x: FiniteFloat
    coord_y: FiniteFloat
    densita_locale: FiniteFloat
    giorni_accumulati: FiniteFloat
    zona_raccolta: int = Field(ge=1, le=4)
    frazione: str = Field(min_length=1)
    cluster_label: int


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[PredictionInput] = Field(min_length=1)


class UnknownGroup(ValueError):
    def __init__(self, index: int, frazione: str, cluster_label: int):
        self.index = index
        self.frazione = frazione
        self.cluster_label = cluster_label
        super().__init__(
            f"No model for frazione={frazione}, cluster_label={cluster_label}."
        )


def feature_values(instance: PredictionInput) -> dict[str, float]:
    value_date = instance.data
    day_of_year = int(value_date.strftime("%j"))
    values = {
        "coord_x": float(instance.coord_x),
        "coord_y": float(instance.coord_y),
        "densita_locale": float(instance.densita_locale),
        "giorni_accumulati": float(instance.giorni_accumulati),
        "anno_ref": float(instance.anno_produzione),
        "month": float(value_date.month),
        "day_of_month": float(value_date.day),
        "day_of_year": float(day_of_year),
        "day_of_week": float(value_date.strftime("%w")),
        "week_of_year": float(value_date.strftime("%W")),
        "quarter": float((value_date.month - 1) // 3 + 1),
        "season_sin": math.sin(2 * math.pi * day_of_year / 366.0),
        "season_cos": math.cos(2 * math.pi * day_of_year / 366.0),
    }
    for zone in range(1, 5):
        values[f"zona_raccolta={zone}"] = float(
            instance.zona_raccolta == zone
        )
    return values


class ModelPackage:
    def __init__(
        self,
        model_version: str,
        feature_names: list[str],
        input_name: str,
        output_name: str,
        sessions: dict[tuple[str, int], ort.InferenceSession],
    ):
        self.model_version = model_version
        self.feature_names = feature_names
        self.input_name = input_name
        self.output_name = output_name
        self.sessions = sessions

    @classmethod
    def load(cls, model_dir: Path) -> ModelPackage:
        manifest_path = model_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported model package schema_version.")
        if manifest.get("task") != "grouped_regression":
            raise ValueError("Unsupported model package task.")

        input_spec = manifest["input"]
        output_spec = manifest["output"]
        feature_names = list(input_spec["feature_names"])
        unsupported = set(feature_names) - SUPPORTED_FEATURES
        if unsupported:
            raise ValueError(f"Unsupported features: {sorted(unsupported)}")
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Model feature_names contain duplicates.")
        if input_spec.get("dtype") != "float64":
            raise ValueError("Only float64 model inputs are supported.")
        if output_spec.get("dtype") != "float64":
            raise ValueError("Only float64 model outputs are supported.")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        root = model_dir.resolve()
        sessions = {}
        for group in manifest["groups"]:
            key = (str(group["frazione"]), int(group["cluster_label"]))
            if key in sessions:
                raise ValueError(f"Duplicate model group: {key}")

            model_path = (root / group["model"]).resolve()
            if not model_path.is_relative_to(root):
                raise ValueError("Model path escapes the package directory.")
            session = ort.InferenceSession(
                model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if len(inputs) != 1 or inputs[0].name != input_spec["name"]:
                raise ValueError(f"Unexpected model input for group {key}.")
            if inputs[0].type != "tensor(double)":
                raise ValueError(f"Unexpected model input type for group {key}.")
            if len(inputs[0].shape) != 2 or inputs[0].shape[1] != len(
                feature_names
            ):
                raise ValueError(f"Unexpected model input shape for group {key}.")
            if len(outputs) != 1 or outputs[0].name != output_spec["name"]:
                raise ValueError(f"Unexpected model output for group {key}.")
            if outputs[0].type != "tensor(double)":
                raise ValueError(f"Unexpected model output type for group {key}.")
            if len(outputs[0].shape) != 2 or outputs[0].shape[1] != 1:
                raise ValueError(f"Unexpected model output shape for group {key}.")
            sessions[key] = session

        if not sessions:
            raise ValueError("Model package contains no groups.")
        return cls(
            model_version=str(manifest["model_version"]),
            feature_names=feature_names,
            input_name=input_spec["name"],
            output_name=output_spec["name"],
            sessions=sessions,
        )

    def predict(self, instances: list[PredictionInput]) -> list[float]:
        grouped: dict[tuple[str, int], list[tuple[int, PredictionInput]]] = (
            defaultdict(list)
        )
        for index, instance in enumerate(instances):
            key = (instance.frazione, instance.cluster_label)
            if key not in self.sessions:
                raise UnknownGroup(index, *key)
            grouped[key].append((index, instance))

        predictions = [0.0] * len(instances)
        for key, rows in grouped.items():
            vectors = []
            for _, item in rows:
                values = feature_values(item)
                vectors.append([values[name] for name in self.feature_names])
            matrix = np.asarray(
                vectors,
                dtype=np.float64,
            )
            output = self.sessions[key].run(
                [self.output_name], {self.input_name: matrix}
            )[0].reshape(-1)
            if len(output) != len(rows) or not np.isfinite(output).all():
                raise RuntimeError(f"Invalid inference output for group {key}.")
            for (index, _), value in zip(rows, output, strict=True):
                predictions[index] = float(value)
        return predictions


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("MODEL_DIR", "/models"))
    app.state.model_package = ModelPackage.load(model_dir)
    yield


app = FastAPI(title="ONNX grouped regressor", version="1", lifespan=lifespan)


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def ready(request: Request) -> dict[str, str | int]:
    package = request.app.state.model_package
    return {
        "status": "ready",
        "model_version": package.model_version,
        "group_count": len(package.sessions),
    }


@app.post("/v1/predict")
def predict(
    payload: PredictionRequest, request: Request
) -> dict[str, str | list[dict[str, float]]]:
    package = request.app.state.model_package
    try:
        values = package.predict(payload.instances)
    except UnknownGroup as exc:
        raise HTTPException(
            status_code=422,
            detail={"index": exc.index, "message": str(exc)},
        ) from exc
    return {
        "model_version": package.model_version,
        "predictions": [{"prediction_t": value} for value in values],
    }
