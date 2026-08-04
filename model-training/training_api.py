from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal
from uuid import uuid4
import logging
import os
import sys
import tempfile
import threading

from fastapi import BackgroundTasks, FastAPI, HTTPException, status as http_status
from pydantic import BaseModel, ConfigDict, Field, model_validator

import ml_pipeline_graph as pipeline


ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForecastOutput(StrictModel):
    forecast_from_date: date
    forecast_to_date: date
    forecast_calendar_path: Path | None = None

    @model_validator(mode="after")
    def validate_range(self) -> ForecastOutput:
        if self.forecast_from_date > self.forecast_to_date:
            raise ValueError("forecast_from_date must not be after forecast_to_date")
        return self


class OrionPublication(ForecastOutput):
    pass


class FileTrainingRequest(StrictModel):
    source: Literal["file"]
    pipeline: Literal["legacy", "zone-fraction"] = "legacy"
    year: int = Field(ge=1, le=9999)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    regressor: Literal["sgd", "gpr", "hist_gradient_boosting", "arima"] = "sgd"
    arima_group_level: bool = False
    gpr_max_train_rows: int = Field(default=500, gt=0)
    annual_input_path: Path
    production_input_path: Path | None = None
    collections_input_path: Path | None = None
    calendar_input_path: Path
    output_dir: Path
    force: bool = False
    forecast: ForecastOutput | None = None
    orion: OrionPublication | None = None
    include_arima: bool = False
    exclude_episodic_fractions: bool = False
    hgb_loss: Literal["poisson", "squared_error"] = "poisson"
    validation_strategy: Literal["fixed", "k-fold"] = "fixed"
    validation_folds: int = Field(default=5, ge=2)

    @model_validator(mode="after")
    def validate_forecast(self) -> FileTrainingRequest:
        if self.forecast and self.orion:
            raise ValueError("use orion alone when publishing a forecast")
        horizon = self.orion or self.forecast
        if horizon and horizon.forecast_from_date <= date(self.year, 12, 31):
            raise ValueError("forecast dates must be after the training year")
        if self.pipeline == "zone-fraction":
            if (
                self.production_input_path is None
                or self.collections_input_path is None
            ):
                raise ValueError(
                    "zone-fraction file input requires production_input_path "
                    "and collections_input_path"
                )
            return self
        if self.orion and self.regressor == "arima":
            raise ValueError("legacy ARIMA supports files, not Orion publication")
        if self.arima_group_level and self.regressor != "arima":
            raise ValueError("arima_group_level requires regressor=arima")
        if (
            horizon
            and self.regressor == "arima"
            and not self.arima_group_level
            and horizon.forecast_calendar_path is None
        ):
            raise ValueError("per-civic ARIMA forecasts require a calendar path")
        return self


class QuantumLeapTrainingRequest(StrictModel):
    source: Literal["quantumleap"]
    pipeline: Literal["legacy", "zone-fraction"] = "legacy"
    year: int | None = Field(default=None, ge=1, le=9999)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    regressor: Literal["sgd", "gpr", "hist_gradient_boosting", "arima"] = "sgd"
    arima_group_level: bool = False
    gpr_max_train_rows: int = Field(default=500, gt=0)
    from_date: str
    to_date: str
    ql_id_pattern: str | None = None
    output_dir: Path
    calendar_input_path: Path | None = None
    orion: OrionPublication | None = None
    forecast: ForecastOutput | None = None
    include_arima: bool = False
    exclude_episodic_fractions: bool = False
    hgb_loss: Literal["poisson", "squared_error"] = "poisson"
    validation_strategy: Literal["fixed", "k-fold"] = "fixed"
    validation_folds: int = Field(default=5, ge=2)

    @model_validator(mode="after")
    def validate_ranges(self) -> QuantumLeapTrainingRequest:
        self.from_date, self.to_date = pipeline.normalize_training_range(
            self.from_date, self.to_date
        )
        training_start = datetime.fromisoformat(
            self.from_date.replace("Z", "+00:00")
        ).date()
        training_end = datetime.fromisoformat(
            self.to_date.replace("Z", "+00:00")
        ).date()
        if self.year is None:
            self.year = training_start.year
        if self.pipeline == "zone-fraction":
            if (
                training_start.year != training_end.year
                or self.year != training_start.year
            ):
                raise ValueError("zone-fraction QuantumLeap input must cover one year")
            if self.calendar_input_path is None:
                raise ValueError(
                    "zone-fraction QuantumLeap input requires calendar_input_path"
                )
        if self.forecast and self.orion:
            raise ValueError("use orion alone when publishing a forecast")
        horizon = self.orion or self.forecast
        if self.orion:
            if self.pipeline == "legacy" and self.regressor == "arima":
                raise ValueError("ARIMA supports batch files, not Orion publication")
            if (
                self.orion.forecast_calendar_path is None
                and self.calendar_input_path is None
            ):
                raise ValueError("Orion publication requires a calendar path")
            if self.orion.forecast_from_date <= training_end:
                raise ValueError("forecast dates must be after to_date")
        if (
            self.pipeline == "legacy"
            and self.arima_group_level
            and self.regressor != "arima"
        ):
            raise ValueError("arima_group_level requires regressor=arima")
        if horizon:
            if horizon.forecast_from_date <= training_end:
                raise ValueError("forecast dates must be after to_date")
            if (
                self.pipeline == "legacy"
                and self.regressor == "arima"
                and not self.arima_group_level
                and horizon.forecast_calendar_path is None
            ):
                raise ValueError("per-civic ARIMA forecasts require a calendar path")
        return self


TrainingRequest = Annotated[
    FileTrainingRequest | QuantumLeapTrainingRequest,
    Field(discriminator="source"),
]


class JobStatus(StrictModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    error: str | None = None


jobs: dict[str, JobStatus] = {}
jobs_lock = threading.Lock()
# one active in-process job; use a durable queue if multi-worker dispatch is needed.
active_job_id: str | None = None

app = FastAPI(title="ML training pipeline", version="1")


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_file(path: Path, field: str) -> Path:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise HTTPException(
            status_code=422, detail=f"{field} must be an existing file: {resolved}"
        )
    return resolved


def prepare_request(payload: TrainingRequest) -> TrainingRequest:
    output_dir = resolve_path(payload.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"output_dir cannot be created: {output_dir}: {exc.strerror}",
        ) from exc
    if not output_dir.is_dir() or not os.access(output_dir, os.W_OK | os.X_OK):
        raise HTTPException(
            status_code=422,
            detail=f"output_dir must be a writable directory: {output_dir}",
        )
    if isinstance(payload, FileTrainingRequest):
        updates: dict[str, object] = {
            "annual_input_path": validate_file(
                payload.annual_input_path, "annual_input_path"
            ),
            "calendar_input_path": validate_file(
                payload.calendar_input_path, "calendar_input_path"
            ),
            "output_dir": output_dir,
        }
        if payload.production_input_path:
            updates["production_input_path"] = validate_file(
                payload.production_input_path, "production_input_path"
            )
        if payload.collections_input_path:
            updates["collections_input_path"] = validate_file(
                payload.collections_input_path, "collections_input_path"
            )
        if payload.forecast:
            forecast_updates = {}
            if payload.forecast.forecast_calendar_path:
                forecast_updates["forecast_calendar_path"] = validate_file(
                    payload.forecast.forecast_calendar_path,
                    "forecast.forecast_calendar_path",
                )
            updates["forecast"] = payload.forecast.model_copy(update=forecast_updates)
        if payload.orion:
            orion_updates = {}
            if payload.orion.forecast_calendar_path:
                orion_updates["forecast_calendar_path"] = validate_file(
                    payload.orion.forecast_calendar_path,
                    "orion.forecast_calendar_path",
                )
            updates["orion"] = payload.orion.model_copy(update=orion_updates)
        return payload.model_copy(update=updates)
    updates: dict[str, object] = {"output_dir": output_dir}
    if payload.calendar_input_path:
        updates["calendar_input_path"] = validate_file(
            payload.calendar_input_path, "calendar_input_path"
        )
    if payload.orion:
        orion_updates = {}
        if payload.orion.forecast_calendar_path:
            orion_updates["forecast_calendar_path"] = validate_file(
                payload.orion.forecast_calendar_path,
                "orion.forecast_calendar_path",
            )
        updates["orion"] = payload.orion.model_copy(update=orion_updates)
    if payload.forecast:
        forecast_updates = {}
        if payload.forecast.forecast_calendar_path:
            forecast_updates["forecast_calendar_path"] = validate_file(
                payload.forecast.forecast_calendar_path,
                "forecast.forecast_calendar_path",
            )
        updates["forecast"] = payload.forecast.model_copy(update=forecast_updates)
    return payload.model_copy(update=updates)


def positive_env(name: str, default: str, cast: type[float] | type[int]):
    value = cast(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def quantumleap_args(payload: QuantumLeapTrainingRequest) -> SimpleNamespace:
    publication = payload.orion
    forecast = payload.forecast or publication
    return SimpleNamespace(
        from_date=payload.from_date,
        to_date=payload.to_date,
        output_dir=payload.output_dir,
        pipeline=payload.pipeline,
        anno=str(payload.year),
        regressor=payload.regressor,
        arima_group_level=payload.arima_group_level,
        gpr_max_train_rows=payload.gpr_max_train_rows,
        ql_id_pattern=payload.ql_id_pattern,
        ql_url=os.environ.get("QL_URL", "http://localhost:8668"),
        ql_entity_type=os.environ.get("ENTITY_TYPE", "RifiutiConferimento"),
        fiware_service=os.environ.get("FIWARE_SERVICE", "openiot"),
        fiware_service_path=os.environ.get("FIWARE_SERVICE_PATH", "/"),
        ql_timeout_seconds=positive_env("QL_TIMEOUT_SECONDS", "30", float),
        publish_orion=publication is not None,
        forecast_from_date=(
            forecast.forecast_from_date.isoformat() if forecast else None
        ),
        forecast_to_date=(forecast.forecast_to_date.isoformat() if forecast else None),
        forecast_calendar=(
            str(forecast.forecast_calendar_path or payload.calendar_input_path)
            if forecast
            else None
        ),
        annual_seed=42,
        scenario_seed=42,
        spillover_days=14,
        missing_calendar_mode="error",
        generator_version="quantumleap",
        exclude_episodic_fractions=payload.exclude_episodic_fractions,
        hgb_loss=payload.hgb_loss,
        include_arima=payload.include_arima or forecast is not None,
        validation_strategy=payload.validation_strategy,
        validation_folds=payload.validation_folds,
        orion_url=os.environ.get("ORION_URL", "http://localhost:1026"),
        orion_entity_type=os.environ.get(
            "ORION_TWIN_ENTITY_TYPE", "RifiutiDigitalTwin"
        ),
        orion_timeout_seconds=positive_env("ORION_TIMEOUT_SECONDS", "30", float),
        orion_batch_size=positive_env("ORION_BATCH_SIZE", "50", int),
    )


def file_args(payload: FileTrainingRequest) -> SimpleNamespace:
    horizon = payload.orion or payload.forecast
    return SimpleNamespace(
        pipeline=payload.pipeline,
        seed=payload.seed,
        anno=str(payload.year),
        output_dir=payload.output_dir,
        annual_input=payload.annual_input_path,
        production_input=payload.production_input_path,
        collections_input=payload.collections_input_path,
        calendar_input=payload.calendar_input_path,
        annual_seed=42,
        scenario_seed=42,
        spillover_days=14,
        missing_calendar_mode="error",
        generator_version="api",
        exclude_episodic_fractions=payload.exclude_episodic_fractions,
        hgb_loss=payload.hgb_loss,
        include_arima=payload.include_arima,
        validation_strategy=payload.validation_strategy,
        validation_folds=payload.validation_folds,
        regressor=payload.regressor,
        gpr_max_train_rows=payload.gpr_max_train_rows,
        arima_group_level=payload.arima_group_level,
        arima_group_by="cluster",
        forecast_from_date=(
            horizon.forecast_from_date.isoformat() if horizon else None
        ),
        forecast_to_date=horizon.forecast_to_date.isoformat() if horizon else None,
        forecast_calendar=str(
            (horizon.forecast_calendar_path if horizon else None)
            or payload.calendar_input_path
        ),
        publish_orion=payload.orion is not None,
        orion_url=os.environ.get("ORION_URL", "http://localhost:1026"),
        orion_entity_type=os.environ.get(
            "ORION_TWIN_ENTITY_TYPE", "RifiutiDigitalTwin"
        ),
        orion_timeout_seconds=positive_env("ORION_TIMEOUT_SECONDS", "30", float),
        orion_batch_size=positive_env("ORION_BATCH_SIZE", "50", int),
        fiware_service=os.environ.get("FIWARE_SERVICE", "openiot"),
        fiware_service_path=os.environ.get("FIWARE_SERVICE_PATH", "/"),
    )


def update_job(job_id: str, job_status: str, error: str | None = None) -> None:
    with jobs_lock:
        jobs[job_id] = JobStatus(job_id=job_id, status=job_status, error=error)


def run_job(job_id: str, payload: TrainingRequest) -> None:
    global active_job_id
    update_job(job_id, "running")
    try:
        if isinstance(payload, FileTrainingRequest):
            graph = pipeline.build_file_pipeline_graph(
                sys.executable, file_args(payload)
            )
            pipeline.run_graph(graph, force=payload.force, dry_run=False)
        else:
            with tempfile.TemporaryDirectory(prefix="quantumleap-training-") as tmp:
                args = quantumleap_args(payload)
                graph = (
                    pipeline.build_zone_fraction_quantumleap_graph(
                        sys.executable,
                        str(payload.seed),
                        str(payload.year),
                        payload.output_dir,
                        payload.calendar_input_path,
                        args,
                    )
                    if payload.pipeline == "zone-fraction"
                    else pipeline.build_quantumleap_graph(
                        sys.executable,
                        str(payload.seed),
                        Path(tmp),
                        args,
                    )
                )
                pipeline.run_graph(graph, force=False, dry_run=False)
    except Exception as exc:
        logger.exception("Training job %s failed", job_id)
        update_job(job_id, "failed", str(exc))
    else:
        update_job(job_id, "succeeded")
    finally:
        with jobs_lock:
            if active_job_id == job_id:
                active_job_id = None


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/training-jobs",
    response_model=JobStatus,
    status_code=http_status.HTTP_202_ACCEPTED,
)
def create_training_job(
    payload: TrainingRequest, background_tasks: BackgroundTasks
) -> JobStatus:
    global active_job_id
    payload = prepare_request(payload)
    with jobs_lock:
        if active_job_id is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "A training job is already active.",
                    "job_id": active_job_id,
                },
            )
        job_id = str(uuid4())
        job = JobStatus(job_id=job_id, status="queued")
        jobs[job_id] = job
        active_job_id = job_id
    background_tasks.add_task(run_job, job_id, payload)
    return job


@app.get("/v1/training-jobs/{job_id}", response_model=JobStatus)
def get_training_job(job_id: str) -> JobStatus:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return job
