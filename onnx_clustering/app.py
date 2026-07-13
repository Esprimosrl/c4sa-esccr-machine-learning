from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
import json
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


ANNUAL_FIELDS = (
    "altro_rd_t",
    "carta_cartone_t",
    "frazione_organica_t",
    "ingombranti_misti_t",
    "legno_t",
    "metallo_t",
    "plastica_t",
    "pulizia_stradale_recupero_t",
    "raee_t",
    "rifiuti_cd_t",
    "selettiva_t",
    "tessili_t",
    "vetro_t",
)
SUPPORTED_NUMERIC_NAMES = {
    "coord_x",
    "coord_y",
    "densita_locale",
    "giorni_accumulati",
    "anno_ref",
    *ANNUAL_FIELDS,
}
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anno_produzione: int
    coord_x: FiniteFloat
    coord_y: FiniteFloat
    densita_locale: NonNegativeFloat
    altro_rd_t: NonNegativeFloat
    carta_cartone_t: NonNegativeFloat
    frazione_organica_t: NonNegativeFloat
    ingombranti_misti_t: NonNegativeFloat
    legno_t: NonNegativeFloat
    metallo_t: NonNegativeFloat
    plastica_t: NonNegativeFloat
    pulizia_stradale_recupero_t: NonNegativeFloat
    raee_t: NonNegativeFloat
    rifiuti_cd_t: NonNegativeFloat
    selettiva_t: NonNegativeFloat
    tessili_t: NonNegativeFloat
    vetro_t: NonNegativeFloat
    zona_raccolta: int = Field(ge=1, le=4)
    zona_costruzione: str | None = None


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[PredictionInput] = Field(min_length=1)


class UnknownCategory(ValueError):
    def __init__(self, index: int, name: str, value: str):
        self.index = index
        super().__init__(f"Unknown {name}: {value!r}.")


class ModelPackage:
    def __init__(
        self,
        model_version: str,
        n_clusters: int,
        feature_names: list[str],
        numeric_names: list[str],
        categorical_levels: dict[str, list[str]],
        numeric_defaults: dict[str, float],
        input_name: str,
        label_name: str,
        scores_name: str,
        session: ort.InferenceSession,
    ):
        self.model_version = model_version
        self.n_clusters = n_clusters
        self.feature_names = feature_names
        self.numeric_names = numeric_names
        self.categorical_levels = categorical_levels
        self.numeric_defaults = numeric_defaults
        self.input_name = input_name
        self.label_name = label_name
        self.scores_name = scores_name
        self.session = session

    @classmethod
    def load(cls, model_dir: Path) -> ModelPackage:
        manifest = json.loads(
            (model_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported model package schema.")
        if manifest.get("task") != "annual_clustering":
            raise ValueError("Model package is not for annual clustering.")

        input_spec = manifest["input"]
        outputs = manifest["outputs"]
        feature_names = list(input_spec["feature_names"])
        numeric_names = list(input_spec["numeric_names"])
        categorical_levels = {
            name: list(levels)
            for name, levels in input_spec["categorical_levels"].items()
        }
        expected_features = numeric_names + [
            f"{name}={level}"
            for name, levels in categorical_levels.items()
            for level in levels
        ]
        if feature_names != expected_features:
            raise ValueError("Manifest feature order is inconsistent.")
        if set(numeric_names) != SUPPORTED_NUMERIC_NAMES:
            raise ValueError("Model numeric features do not match the API.")
        if set(categorical_levels) != {
            "zona_raccolta",
            "zona_costruzione",
        }:
            raise ValueError("Model categorical features do not match the API.")
        if input_spec.get("dtype") != "float64":
            raise ValueError("Only float64 model inputs are supported.")
        if outputs["label"].get("dtype") != "int64":
            raise ValueError("Only int64 labels are supported.")
        if outputs["scores"].get("dtype") != "float64":
            raise ValueError("Only float64 scores are supported.")

        root = model_dir.resolve()
        model_path = (root / manifest["model"]).resolve()
        if not model_path.is_relative_to(root):
            raise ValueError("Model path escapes the package directory.")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = session.get_inputs()
        model_outputs = session.get_outputs()
        if len(inputs) != 1 or (
            inputs[0].name,
            inputs[0].type,
            len(inputs[0].shape),
            inputs[0].shape[1],
        ) != (input_spec["name"], "tensor(double)", 2, len(feature_names)):
            raise ValueError("Unexpected model input.")
        if [(item.name, item.type) for item in model_outputs] != [
            (outputs["label"]["name"], "tensor(int64)"),
            (outputs["scores"]["name"], "tensor(double)"),
        ]:
            raise ValueError("Unexpected model outputs.")
        n_clusters = int(manifest["n_clusters"])
        if len(model_outputs[1].shape) != 2 or (
            model_outputs[1].shape[1] != n_clusters
        ):
            raise ValueError("Unexpected model score shape.")

        return cls(
            model_version=str(manifest["model_version"]),
            n_clusters=n_clusters,
            feature_names=feature_names,
            numeric_names=numeric_names,
            categorical_levels=categorical_levels,
            numeric_defaults={
                name: float(value)
                for name, value in input_spec["numeric_defaults"].items()
            },
            input_name=input_spec["name"],
            label_name=outputs["label"]["name"],
            scores_name=outputs["scores"]["name"],
            session=session,
        )

    def _feature_values(
        self, instance: PredictionInput, index: int
    ) -> dict[str, float]:
        raw = instance.model_dump()
        values = {
            name: float(raw[name])
            for name in self.numeric_names
            if name in raw
        }
        values["anno_ref"] = float(instance.anno_produzione)
        values["giorni_accumulati"] = self.numeric_defaults[
            "giorni_accumulati"
        ]

        categories = {
            "zona_raccolta": str(instance.zona_raccolta),
            "zona_costruzione": (
                instance.zona_costruzione.strip()
                if instance.zona_costruzione is not None
                else None
            ),
        }
        for name, levels in self.categorical_levels.items():
            value = categories[name]
            if value is not None and value not in levels:
                raise UnknownCategory(index, name, value)
            for level in levels:
                values[f"{name}={level}"] = float(value == level)
        return values

    def predict(self, instances: list[PredictionInput]) -> list[tuple[int, float]]:
        vectors = []
        for index, instance in enumerate(instances):
            values = self._feature_values(instance, index)
            vectors.append([values[name] for name in self.feature_names])
        matrix = np.asarray(vectors, dtype=np.float64)
        labels, scores = self.session.run(
            [self.label_name, self.scores_name], {self.input_name: matrix}
        )
        distances = scores.min(axis=1)
        if (
            len(labels) != len(instances)
            or scores.shape != (len(instances), self.n_clusters)
            or not np.isfinite(distances).all()
        ):
            raise RuntimeError("Invalid clustering inference output.")
        return [
            (int(label), float(distance))
            for label, distance in zip(labels, distances, strict=True)
        ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("MODEL_DIR", "/models"))
    app.state.model_package = ModelPackage.load(model_dir)
    yield


app = FastAPI(title="ONNX annual clustering", version="1", lifespan=lifespan)


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def ready(request: Request) -> dict[str, str | int]:
    package = request.app.state.model_package
    return {
        "status": "ready",
        "model_version": package.model_version,
        "n_clusters": package.n_clusters,
    }


@app.post("/v1/predict")
def predict(payload: PredictionRequest, request: Request) -> dict:
    package = request.app.state.model_package
    try:
        values = package.predict(payload.instances)
    except UnknownCategory as exc:
        raise HTTPException(
            status_code=422,
            detail={"index": exc.index, "message": str(exc)},
        ) from exc
    return {
        "model_version": package.model_version,
        "predictions": [
            {"cluster_label": label, "cluster_distance": distance}
            for label, distance in values
        ],
    }
