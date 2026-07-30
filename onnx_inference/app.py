from __future__ import annotations

from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
import json
import math
import os
import sys
import time

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError

from quantumleap import (
    QuantumLeapClient,
    QuantumLeapError,
    QuantumLeapTimeout,
    parse_datetime,
)
from inference_data import DAILY_ATTRS, annual_rows, attribute, forecast_events, latest_annual, split_years, twin_id
from orion import OrionError, OrionTimeout, publish_entities
from lag_features import LAG_FEATURES, LagState

TRAINING_DIR = Path(__file__).resolve().parents[1] / "model-training"
if TRAINING_DIR.is_dir():
    sys.path.insert(0, str(TRAINING_DIR))

from zone_fraction_benchmark import (
    daily_sarimax_exog,
    forecast_schedule,
    sarima_forecast,
)


SUPPORTED_FEATURES = {
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
    historical_annual_t: FiniteFloat = Field(default=0, ge=0)
    historical_collection_mean_t: FiniteFloat = Field(default=0, ge=0)
    lag_1_collection_t: FiniteFloat = Field(default=0, ge=0)
    rolling_7d_sum_t: FiniteFloat = Field(default=0, ge=0)
    rolling_30d_sum_t: FiniteFloat = Field(default=0, ge=0)
    rolling_90d_median_t: FiniteFloat = Field(default=0, ge=0)
    rolling_90d_count: FiniteFloat = Field(default=0, ge=0)
    days_since_last_nonzero_collection: FiniteFloat = Field(default=0, ge=0)
    zona_raccolta: int = Field(ge=1, le=4)
    frazione: str = Field(min_length=1)
    cluster_label: int


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[PredictionInput] = Field(min_length=1)


class QuantumLeapPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_date: str
    to_date: str
    entity_ids: list[str] | None = None


class QuantumLeapForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_date: str
    to_date: str
    forecast_from_date: date
    forecast_to_date: date
    civic_ids: list[str] | None = Field(default=None, min_length=1)


class ZoneFractionPredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    giorni_dal_ritiro_precedente: FiniteFloat | None = None
    month: FiniteFloat | None = None
    day_of_week: FiniteFloat | None = None
    week_of_year: FiniteFloat | None = None
    day_of_year_sin: FiniteFloat | None = None
    day_of_year_cos: FiniteFloat | None = None
    day_of_week_sin: FiniteFloat | None = None
    day_of_week_cos: FiniteFloat | None = None
    is_holiday: FiniteFloat | None = None
    is_after_holiday: FiniteFloat | None = None
    is_summer: FiniteFloat | None = None
    is_christmas_period: FiniteFloat | None = None
    lag_1_collection_t: FiniteFloat | None = None
    lag_7d_collection_t: FiniteFloat | None = None
    rolling_4_collection_mean_t: FiniteFloat | None = None
    rolling_4_collection_median_t: FiniteFloat | None = None
    rolling_7d_mean_t: FiniteFloat | None = None
    rolling_7d_sum_t: FiniteFloat | None = None
    rolling_7d_count: FiniteFloat | None = None
    lag_1_missing: FiniteFloat | None = None
    lag_7d_missing: FiniteFloat | None = None
    rolling_4_missing: FiniteFloat | None = None
    rolling_7d_missing: FiniteFloat | None = None
    zona_raccolta: str = Field(min_length=1)
    frazione: str = Field(min_length=1)
    cluster_label: int


class ZoneFractionPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[ZoneFractionPredictionInput] = Field(min_length=1)


class ZoneFractionForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecast_from_date: date
    forecast_to_date: date
    zones: list[str] | None = Field(default=None, min_length=1)
    fractions: list[str] | None = Field(default=None, min_length=1)
    cluster_labels: list[int] | None = Field(default=None, min_length=1)


QUANTUMLEAP_FEATURES = [
    "id_civico",
    "frazione",
    "anno_produzione",
    "coord_x",
    "coord_y",
    "densita_locale",
    "zona_raccolta",
    "giorni_accumulati",
    "cluster_label",
    "conferimento_t",
]


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
        "historical_annual_t": float(instance.historical_annual_t),
        "historical_collection_mean_t": float(
            instance.historical_collection_mean_t
        ),
        "lag_1_collection_t": float(instance.lag_1_collection_t),
        "rolling_7d_sum_t": float(instance.rolling_7d_sum_t),
        "rolling_30d_sum_t": float(instance.rolling_30d_sum_t),
        "rolling_90d_median_t": float(instance.rolling_90d_median_t),
        "rolling_90d_count": float(instance.rolling_90d_count),
        "days_since_last_nonzero_collection": float(
            instance.days_since_last_nonzero_collection
        ),
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
    task = "grouped_regression"

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


class ZoneFractionPackage:
    task = "zone_fraction_regression"

    def __init__(
        self,
        model_version: str,
        tabular_version: str,
        sarimax_version: str,
        inputs: list[dict[str, str]],
        input_names: set[str],
        preprocessor: object | None,
        transformed_feature_count: int | None,
        input_name: str | None,
        output_name: str,
        session: ort.InferenceSession,
        sarimax_models: dict[str, dict[str, object]],
    ):
        self.model_version = model_version
        self.tabular_version = tabular_version
        self.sarimax_version = sarimax_version
        self.inputs = inputs
        self.input_names = input_names
        self.preprocessor = preprocessor
        self.transformed_feature_count = transformed_feature_count
        self.input_name = input_name
        self.output_name = output_name
        self.session = session
        self.sarimax_models = sarimax_models

    @property
    def group_count(self) -> int:
        return len(self.sarimax_models)

    @classmethod
    def load(cls, model_dir: Path, manifest: dict[str, object]) -> ZoneFractionPackage:
        root = model_dir.resolve()
        tabular = manifest["tabular"]
        sarimax = manifest["sarimax"]
        tabular_path = (root / tabular["model"]).resolve()
        sarimax_path = (root / sarimax["model"]).resolve()
        preprocessor_path = (
            (root / tabular["preprocessor"]).resolve()
            if tabular.get("preprocessor")
            else None
        )
        if (
            not tabular_path.is_relative_to(root)
            or not sarimax_path.is_relative_to(root)
            or (
                preprocessor_path is not None
                and not preprocessor_path.is_relative_to(root)
            )
        ):
            raise ValueError("Model path escapes the package directory.")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            tabular_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = list(tabular["inputs"])
        input_names = {str(item["name"]) for item in inputs}
        preprocessor = None
        transformed_feature_count = None
        input_name = None
        if preprocessor_path is not None:
            preprocessor = joblib.load(preprocessor_path)
            transformed_feature_count = int(tabular["transformed_feature_count"])
            session_inputs = session.get_inputs()
            if (
                len(session_inputs) != 1
                or session_inputs[0].type != "tensor(float)"
                or len(session_inputs[0].shape) != 2
                or session_inputs[0].shape[1] != transformed_feature_count
            ):
                raise ValueError(
                    "Zone/fraction ONNX transformed input does not match "
                    "the manifest."
                )
            input_name = session_inputs[0].name
        elif input_names != {item.name for item in session.get_inputs()}:
            raise ValueError("Zone/fraction ONNX inputs do not match the manifest.")
        output_name = str(tabular["output"]["name"])
        if [item.name for item in session.get_outputs()] != [output_name]:
            raise ValueError("Zone/fraction ONNX output does not match the manifest.")

        native_bundle = joblib.load(sarimax_path)
        if (
            native_bundle.get("contract")
            != "zone-fraction-scheduled-withdrawal-v1"
            or native_bundle.get("model_name") != "zone_fraction_sarima"
        ):
            raise ValueError("Unsupported native zone/fraction SARIMAX bundle.")
        sarimax_models = native_bundle["sarimax_models"]
        if len(sarimax_models) != int(sarimax["group_count"]):
            raise ValueError("SARIMAX group count does not match the manifest.")
        return cls(
            str(manifest["model_version"]),
            str(tabular["version"]),
            str(sarimax["version"]),
            inputs,
            input_names,
            preprocessor,
            transformed_feature_count,
            input_name,
            output_name,
            session,
            sarimax_models,
        )

    def predict(self, instances: list[ZoneFractionPredictionInput]) -> list[float]:
        rows = [instance.model_dump() for instance in instances]
        if self.preprocessor is not None:
            matrix = np.asarray(
                self.preprocessor.transform(pd.DataFrame(rows)),
                dtype=np.float32,
            )
            if (
                matrix.ndim != 2
                or matrix.shape[1] != self.transformed_feature_count
            ):
                raise RuntimeError(
                    "Zone/fraction preprocessor produced an unexpected shape."
                )
            feed = {self.input_name: matrix}
        else:
            feed = {}
            for spec in self.inputs:
                name = str(spec["name"])
                dtype = str(spec["dtype"])
                values = [row[name] for row in rows]
                if dtype in {"float32", "float64"}:
                    array = np.asarray(
                        [np.nan if value is None else value for value in values],
                        dtype=np.float32 if dtype == "float32" else np.float64,
                    )
                elif dtype == "int64":
                    array = np.asarray(values, dtype=np.int64)
                elif dtype == "string":
                    array = np.asarray(values, dtype=object)
                else:
                    raise ValueError(f"Unsupported model input dtype: {dtype}")
                feed[name] = array.reshape(-1, 1)
        output = self.session.run([self.output_name], feed)[0].reshape(-1)
        if len(output) != len(instances) or not np.isfinite(output).all():
            raise RuntimeError("Invalid zone/fraction inference output.")
        return output.astype(float).tolist()

    @staticmethod
    def _group_values(model_key: str) -> dict[str, str]:
        try:
            return dict(part.split("=", 1) for part in model_key.split("|"))
        except ValueError as exc:
            raise ValueError(f"Invalid SARIMAX model key: {model_key}") from exc

    def forecast(
        self,
        payload: ZoneFractionForecastRequest,
        calendar_path: Path,
    ) -> list[dict[str, object]]:
        if payload.forecast_from_date > payload.forecast_to_date:
            raise ValueError("forecast_from_date must not be after forecast_to_date")

        entries = [
            (self._group_values(model_key), info)
            for model_key, info in self.sarimax_models.items()
        ]
        zones = sorted({values["zona_raccolta"] for values, _ in entries})
        fractions = sorted({values["frazione"] for values, _ in entries})
        schedule_start = min(
            (
                pd.Timestamp(info["model"].model.data.dates[-1]).date()
                + pd.Timedelta(days=1)
            )
            for _, info in entries
        )
        if payload.forecast_from_date < schedule_start:
            raise ValueError(
                f"forecast dates must begin on or after {schedule_start.isoformat()}"
            )
        schedule = forecast_schedule(
            calendar_path,
            zones,
            fractions,
            schedule_start,
            payload.forecast_to_date,
        )

        selected_zones = set(payload.zones or zones)
        selected_fractions = set(payload.fractions or fractions)
        selected_clusters = (
            set(payload.cluster_labels) if payload.cluster_labels is not None else None
        )
        rows = []
        for values, info in entries:
            zone = values["zona_raccolta"]
            fraction = values["frazione"]
            cluster_label = (
                int(values["cluster_label"])
                if "cluster_label" in values
                else None
            )
            if zone not in selected_zones or fraction not in selected_fractions:
                continue
            if (
                selected_clusters is not None
                and cluster_label not in selected_clusters
            ):
                continue
            group_schedule = schedule[
                (schedule["zona_raccolta"] == zone)
                & (schedule["frazione"] == fraction)
            ].sort_values("data")
            if group_schedule.empty:
                continue
            index = pd.date_range(
                schedule_start, payload.forecast_to_date, freq="D"
            )
            exog = daily_sarimax_exog(
                group_schedule,
                schedule_start,
                payload.forecast_to_date,
            ).reindex(index)
            predictions = sarima_forecast(
                info["model"],
                schedule_start,
                payload.forecast_to_date,
                exog,
            )
            for event_date in pd.to_datetime(group_schedule["data"]).dt.date:
                if event_date < payload.forecast_from_date:
                    continue
                value = float(predictions.loc[pd.Timestamp(event_date)])
                if not np.isfinite(value):
                    raise RuntimeError(
                        f"SARIMAX produced an invalid forecast for {values}."
                    )
                rows.append(
                    {
                        "scenario_seed": int(values["scenario_seed"]),
                        "data": event_date.isoformat(),
                        "zona_raccolta": zone,
                        "frazione": fraction,
                        "cluster_label": cluster_label,
                        "prediction_t": value,
                    }
                )
        return rows


def load_model_package(model_dir: Path) -> ModelPackage | ZoneFractionPackage:
    manifest = json.loads(
        (model_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("task") == "zone_fraction_regression":
        return ZoneFractionPackage.load(model_dir, manifest)
    return ModelPackage.load(model_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("MODEL_DIR", "/models"))
    app.state.model_package = load_model_package(model_dir)
    ql_url = os.environ.get("QL_URL", "http://localhost:8668")
    service = os.environ.get("FIWARE_SERVICE", "openiot")
    service_path = os.environ.get("FIWARE_SERVICE_PATH", "/")
    timeout = float(os.environ.get("QL_TIMEOUT_SECONDS", "30"))
    app.state.quantumleap = QuantumLeapClient(
        ql_url,
        os.environ.get("ENTITY_TYPE", "RifiutiConferimento"),
        service,
        service_path,
        timeout,
    )
    twin_type = os.environ.get("TWIN_ENTITY_TYPE", "RifiutiDigitalTwin")
    app.state.twin_quantumleap = QuantumLeapClient(
        ql_url, twin_type, service, service_path, timeout
    )
    app.state.forecast = {
        "calendar": Path(os.environ.get("FORECAST_CALENDAR", "/app/calendario_raccolta.csv")),
        "label_wait": float(os.environ.get("QL_LABEL_WAIT_SECONDS", "5")),
        "orion_url": os.environ.get("ORION_URL", "http://localhost:1026"),
        "orion_timeout": float(os.environ.get("ORION_TIMEOUT_SECONDS", "30")),
        "service": service,
        "service_path": service_path,
        "entity_type": twin_type,
    }
    yield


app = FastAPI(title="Waste regression runtime", version="1", lifespan=lifespan)


def legacy_package(request: Request) -> ModelPackage:
    package = request.app.state.model_package
    if not isinstance(package, ModelPackage):
        raise HTTPException(
            status_code=409,
            detail="This endpoint requires a legacy grouped-regression package.",
        )
    return package


def zone_fraction_package(request: Request) -> ZoneFractionPackage:
    package = request.app.state.model_package
    if not isinstance(package, ZoneFractionPackage):
        raise HTTPException(
            status_code=409,
            detail="This endpoint requires a zone/fraction package.",
        )
    return package


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def ready(request: Request) -> dict[str, str | int]:
    package = request.app.state.model_package
    return {
        "status": "ready",
        "task": package.task,
        "model_version": package.model_version,
        "group_count": (
            len(package.sessions)
            if isinstance(package, ModelPackage)
            else package.group_count
        ),
    }


@app.post("/v1/predict")
def predict(
    payload: PredictionRequest, request: Request
) -> dict[str, str | list[dict[str, float]]]:
    package = legacy_package(request)
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


@app.post("/v1/predict/zone-fraction")
def predict_zone_fraction(
    payload: ZoneFractionPredictionRequest, request: Request
) -> dict[str, str | list[dict[str, float]]]:
    package = zone_fraction_package(request)
    values = package.predict(payload.instances)
    return {
        "model_version": package.tabular_version,
        "predictions": [{"prediction_t": value} for value in values],
    }


@app.post("/v1/forecast/zone-fraction")
def forecast_zone_fraction(
    payload: ZoneFractionForecastRequest, request: Request
) -> dict[str, str | list[dict[str, object]]]:
    package = zone_fraction_package(request)
    try:
        rows = package.forecast(payload, request.app.state.forecast["calendar"])
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "model_version": package.sarimax_version,
        "predictions": rows,
    }


@app.post("/v1/predict/quantumleap")
def predict_quantumleap(payload: QuantumLeapPredictionRequest, request: Request):
    package = legacy_package(request)
    try:
        rows = request.app.state.quantumleap.fetch(
            QUANTUMLEAP_FEATURES,
            payload.from_date,
            payload.to_date,
            entity_ids=payload.entity_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except QuantumLeapTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except QuantumLeapError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not rows:
        raise HTTPException(status_code=404, detail="No QuantumLeap records matched.")
    rows.sort(key=lambda row: (parse_datetime(row["timestamp"]), row["entity_id"]))
    try:
        predictions = [0.0] * len(rows)
        lag_state = LagState()
        positions_by_date: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            positions_by_date[row["timestamp"].split("T", 1)[0]].append(index)
        for event_day, positions in positions_by_date.items():
            instances = []
            for index in positions:
                row = rows[index]
                lag_values = lag_state.features(
                    row["id_civico"], row["frazione"], event_day
                )
                instances.append(
                    PredictionInput(
                        data=event_day,
                        anno_produzione=row["anno_produzione"],
                        coord_x=row["coord_x"],
                        coord_y=row["coord_y"],
                        densita_locale=row["densita_locale"],
                        giorni_accumulati=row["giorni_accumulati"],
                        zona_raccolta=row["zona_raccolta"],
                        frazione=row["frazione"],
                        cluster_label=row["cluster_label"],
                        **lag_values,
                    )
                )
            batch_predictions = package.predict(instances)
            for index, prediction in zip(positions, batch_predictions, strict=True):
                predictions[index] = prediction
                row = rows[index]
                lag_state.add(
                    row["id_civico"],
                    row["frazione"],
                    event_day,
                    row["conferimento_t"],
                )
    except UnknownGroup as exc:
        raise HTTPException(
            status_code=422,
            detail={"index": exc.index, "message": str(exc)},
        ) from exc
    except (ValidationError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502, detail=f"Malformed QuantumLeap data: {exc}"
        ) from exc

    return {
        "model_version": package.model_version,
        "predictions": [
            {
                "entity_id": row["entity_id"],
                "timestamp": row["timestamp"],
                "prediction_t": value,
            }
            for row, value in zip(rows, predictions, strict=True)
        ],
    }


LABEL_FEATURES = [
    "id_civico",
    "forecast_year",
    "classification_year",
    "cluster_label",
    "cluster_distance",
    "classifier_model_version",
]


@app.post("/v1/forecast/quantumleap")
def forecast_quantumleap(payload: QuantumLeapForecastRequest, request: Request):
    package = legacy_package(request)
    if payload.forecast_from_date > payload.forecast_to_date:
        raise HTTPException(status_code=422, detail="Invalid forecast date range.")
    try:
        if payload.forecast_from_date <= parse_datetime(payload.to_date).date():
            raise ValueError("Forecast dates must follow the observation range.")
        rows = request.app.state.quantumleap.fetch(
            list(DAILY_ATTRS), payload.from_date, payload.to_date
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except QuantumLeapTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except QuantumLeapError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not rows:
        raise HTTPException(status_code=404, detail="No QuantumLeap records matched.")
    try:
        annual = annual_rows(rows, payload.civic_ids)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Malformed QuantumLeap data: {exc}") from exc
    if not annual:
        raise HTTPException(status_code=404, detail="No civics matched.")

    config = request.app.state.forecast
    targets = []
    for year, year_start, year_end in split_years(
        payload.forecast_from_date, payload.forecast_to_date
    ):
        for row in latest_annual(annual, year):
            targets.append((row, year, year_start, year_end))
    ids = [twin_id(config["entity_type"], row["id_civico"], year) for row, year, _, _ in targets]
    deadline = time.monotonic() + max(config["label_wait"], 0)
    labels = {}
    try:
        while True:
            label_rows = request.app.state.twin_quantumleap.fetch_latest(
                LABEL_FEATURES, id_pattern=".*"
            )
            labels = {row["entity_id"]: row for row in label_rows}
            if all(entity_id in labels for entity_id in ids) or time.monotonic() >= deadline:
                break
            time.sleep(0.25)
    except QuantumLeapTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except QuantumLeapError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    missing = [entity_id for entity_id in ids if entity_id not in labels]
    if missing:
        raise HTTPException(status_code=404, detail=f"Missing digital-twin labels: {', '.join(missing)}")

    model_fractions = sorted({fraction for fraction, _ in package.sessions})
    events = forecast_events(
        config["calendar"],
        [row["zona_raccolta"] for row, _, _, _ in targets],
        model_fractions,
        payload.forecast_from_date,
        payload.forecast_to_date,
    )
    forecast_rows: dict[date, list[tuple[str, dict, dict, int]]] = defaultdict(list)
    for row, year, _, _ in targets:
        entity_id = twin_id(config["entity_type"], row["id_civico"], year)
        label = int(labels[entity_id]["cluster_label"])
        for event in events:
            if event["zona_raccolta"] != str(row["zona_raccolta"]) or event["data"].year != year:
                continue
            if (event["frazione"], label) not in package.sessions:
                continue
            forecast_rows[event["data"]].append((entity_id, row, event, label))
    lag_state = LagState()
    target_civics = {str(row["id_civico"]) for row, _, _, _ in targets}
    for history_row in sorted(rows, key=lambda item: parse_datetime(item["timestamp"])):
        if str(history_row["id_civico"]) in target_civics:
            lag_state.add(
                history_row["id_civico"],
                history_row["frazione"],
                history_row["timestamp"],
                history_row["conferimento_t"],
            )
    grouped: dict[str, dict[date, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    try:
        for event_date, descriptors in sorted(forecast_rows.items()):
            instances = []
            for _, row, event, label in descriptors:
                lag_values = lag_state.features(
                    row["id_civico"], event["frazione"], event_date
                )
                instances.append(PredictionInput(
                    data=event_date,
                    anno_produzione=row["anno_produzione"],
                    coord_x=row["coord_x"],
                    coord_y=row["coord_y"],
                    densita_locale=row["densita_locale"],
                    giorni_accumulati=event["giorni_accumulati"],
                    historical_annual_t=row[f"{event['frazione']}_t"],
                    historical_collection_mean_t=(
                        row[f"{event['frazione']}_t"]
                        / event["annual_collection_count"]
                    ),
                    zona_raccolta=int(row["zona_raccolta"]),
                    frazione=event["frazione"],
                    cluster_label=label,
                    **lag_values,
                ))
            values = package.predict(instances)
            for (entity_id, row, event, _), value in zip(
                descriptors, values, strict=True
            ):
                lag_state.add(
                    row["id_civico"], event["frazione"], event_date, value
                )
                grouped[entity_id][event_date][event["frazione"]] = value
    except UnknownGroup as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    observed = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    entities = []
    predictions = []
    for row, year, year_start, year_end in targets:
        entity_id = twin_id(config["entity_type"], row["id_civico"], year)
        label = labels[entity_id]
        future_values = [
            {"date": event_date.isoformat(), "values_t": fractions}
            for event_date, fractions in sorted(grouped[entity_id].items())
        ]
        entity = {
            "id": entity_id,
            "type": config["entity_type"],
            "id_civico": attribute(str(row["id_civico"]), "Text"),
            "forecast_year": attribute(year, "Integer"),
            "classification_year": attribute(int(label["classification_year"]), "Integer"),
            "cluster_label": attribute(int(label["cluster_label"]), "Integer"),
            "cluster_distance": attribute(float(label["cluster_distance"]), "Number"),
            "classifier_model_version": attribute(str(label["classifier_model_version"]), "Text"),
            "regressor_model_version": attribute(package.model_version, "Text"),
            "forecast_from": attribute(f"{year_start.isoformat()}T00:00:00Z", "DateTime"),
            "forecast_to": attribute(f"{year_end.isoformat()}T23:59:59Z", "DateTime"),
            "dateObserved": attribute(observed, "DateTime"),
            "future_values": attribute(future_values, "StructuredValue"),
        }
        entities.append(entity)
        predictions.append({
            "entity_id": entity_id,
            "id_civico": str(row["id_civico"]),
            "forecast_year": year,
            "classification_year": int(label["classification_year"]),
            "cluster_label": int(label["cluster_label"]),
            "future_values": future_values,
        })
    try:
        publish_entities(
            entities,
            config["orion_url"],
            config["service"],
            config["service_path"],
            config["orion_timeout"],
        )
    except OrionTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except OrionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"model_version": package.model_version, "predictions": predictions}
