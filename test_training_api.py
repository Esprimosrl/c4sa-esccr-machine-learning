from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi import BackgroundTasks, HTTPException
from pydantic import TypeAdapter, ValidationError


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "model-training"))
import training_api  # noqa: E402


request_adapter = TypeAdapter(training_api.TrainingRequest)


@pytest.fixture(autouse=True)
def reset_jobs():
    with training_api.jobs_lock:
        training_api.jobs.clear()
        training_api.active_job_id = None
    yield
    with training_api.jobs_lock:
        training_api.jobs.clear()
        training_api.active_job_id = None


def local_files(tmp_path: Path) -> tuple[Path, Path]:
    annual = tmp_path / "annual.csv"
    calendar = tmp_path / "calendar.csv"
    annual.write_text("id_civico\n1\n", encoding="utf-8")
    calendar.write_text("zona_raccolta\n1\n", encoding="utf-8")
    return annual, calendar


def submit(payload: dict):
    background = BackgroundTasks()
    job = training_api.create_training_job(
        request_adapter.validate_python(payload), background
    )
    task = background.tasks[0]
    task.func(*task.args, **task.kwargs)
    return job


def test_file_job_uses_requested_paths_seed_and_force(tmp_path, monkeypatch):
    annual, calendar = local_files(tmp_path)
    captured = {}

    def run_graph(graph, force, dry_run):
        captured.update(graph=graph, force=force, dry_run=dry_run)

    monkeypatch.setattr(training_api.pipeline, "run_graph", run_graph)
    job = submit(
        {
            "source": "file",
            "year": 2024,
            "seed": 123,
            "regressor": "hist_gradient_boosting",
            "annual_input_path": str(annual),
            "calendar_input_path": str(calendar),
            "output_dir": str(tmp_path / "outputs"),
            "force": True,
        }
    )

    assert job.status == "queued"
    status = training_api.get_training_job(job.job_id)
    assert status.model_dump() == {
        "job_id": job.job_id,
        "status": "succeeded",
        "error": None,
    }
    graph = captured["graph"]
    assert graph["kmeans"].inputs == (annual.resolve(),)
    assert calendar.resolve() in graph["daily_distribution"].inputs
    assert captured["force"] is True
    assert captured["dry_run"] is False
    for name in ("kmeans", "daily_distribution", "grouped_hist_gradient_boosting"):
        command = graph[name].command
        assert command[command.index("--seed") + 1] == "123"


def test_quantumleap_job_can_publish_to_orion(tmp_path, monkeypatch):
    _, calendar = local_files(tmp_path)
    captured = {}
    monkeypatch.setenv("QL_URL", "http://quantumleap:8668")
    monkeypatch.setenv("ENTITY_TYPE", "Waste")
    monkeypatch.setenv("FIWARE_SERVICE", "tenant")
    monkeypatch.setenv("FIWARE_SERVICE_PATH", "/andria")
    monkeypatch.setenv("ORION_URL", "http://orion:1026")

    def run_graph(graph, force, dry_run):
        captured.update(graph=graph, force=force, dry_run=dry_run)

    monkeypatch.setattr(training_api.pipeline, "run_graph", run_graph)
    job = submit(
        {
            "source": "quantumleap",
            "seed": 7,
            "regressor": "gpr",
            "gpr_max_train_rows": 321,
            "from_date": "2024-01-01",
            "to_date": "2024-12-31",
            "ql_id_pattern": ".*:vetro",
            "output_dir": str(tmp_path / "outputs"),
            "orion": {
                "forecast_from_date": "2025-01-01",
                "forecast_to_date": "2025-01-31",
                "forecast_calendar_path": str(calendar),
            },
        }
    )

    assert training_api.get_training_job(job.job_id).status == "succeeded"
    graph = captured["graph"]
    assert list(graph) == [
        "quantumleap_fetch",
        "annual_aggregation",
        "kmeans",
        "cluster_join",
        "grouped_gpr",
        "publish_digital_twins",
    ]
    fetch = graph["quantumleap_fetch"].command
    assert "http://quantumleap:8668" in fetch
    assert "2024-01-01T00:00:00Z" in fetch
    assert "2024-12-31T23:59:59Z" in fetch
    assert ".*:vetro" in fetch
    publish = graph["publish_digital_twins"].command
    assert "http://orion:1026" in publish
    assert str(calendar.resolve()) in publish
    gpr_command = graph["grouped_gpr"].command
    assert gpr_command[gpr_command.index("--regressor") + 1] == "gpr"
    assert gpr_command[gpr_command.index("--gpr-max-train-rows") + 1] == "321"
    assert graph["publish_digital_twins"].needs == ("grouped_gpr",)
    assert any("grouped_gpr_time_series.joblib" in value for value in publish)
    for name in ("kmeans", "grouped_gpr"):
        command = graph[name].command
        assert command[command.index("--seed") + 1] == "7"


def test_arima_job_routes_future_batch_outputs(tmp_path, monkeypatch):
    annual, calendar = local_files(tmp_path)
    captured = {}

    def run_graph(graph, force, dry_run):
        captured["graph"] = graph

    monkeypatch.setattr(training_api.pipeline, "run_graph", run_graph)
    job = submit(
        {
            "source": "file",
            "year": 2024,
            "regressor": "arima",
            "annual_input_path": str(annual),
            "calendar_input_path": str(calendar),
            "output_dir": str(tmp_path / "outputs"),
            "forecast": {
                "forecast_from_date": "2025-01-07",
                "forecast_to_date": "2025-01-31",
                "forecast_calendar_path": str(calendar),
            },
        }
    )

    assert training_api.get_training_job(job.job_id).status == "succeeded"
    node = captured["graph"]["grouped_arima"]
    assert "train_grouped_arima_time_series.py" in node.command[1]
    assert "--regressor" not in node.command
    assert str(calendar.resolve()) in node.command
    assert any("arima_test_predictions_per_civic.parquet" in str(p) for p in node.outputs)
    assert any("arima_future_predictions_per_civic.parquet" in str(p) for p in node.outputs)

    grouped_job = submit(
        {
            "source": "file",
            "year": 2024,
            "regressor": "arima",
            "arima_group_level": True,
            "annual_input_path": str(annual),
            "calendar_input_path": str(calendar),
            "output_dir": str(tmp_path / "grouped-outputs"),
            "forecast": {
                "forecast_from_date": "2025-01-07",
                "forecast_to_date": "2025-01-31",
            },
        }
    )
    assert training_api.get_training_job(grouped_job.job_id).status == "succeeded"
    grouped_node = captured["graph"]["grouped_arima"]
    assert "--group-level" in grouped_node.command
    assert "--forecast-calendar" not in grouped_node.command
    assert any("grouped_arima_time_series_group_level.joblib" in str(p) for p in grouped_node.outputs)
    assert any("arima_test_predictions_grouped.parquet" in str(p) for p in grouped_node.outputs)
    assert any("arima_future_predictions_grouped.parquet" in str(p) for p in grouped_node.outputs)

    with pytest.raises(ValidationError, match="not Orion"):
        request_adapter.validate_python(
            {
                "source": "quantumleap",
                "regressor": "arima",
                "from_date": "2024-01-01",
                "to_date": "2024-12-31",
                "output_dir": str(tmp_path / "outputs"),
                "orion": {
                    "forecast_from_date": "2025-01-01",
                    "forecast_to_date": "2025-01-31",
                    "forecast_calendar_path": str(calendar),
                },
            }
        )


def test_failed_job_and_request_errors_are_reported(tmp_path, monkeypatch):
    annual, calendar = local_files(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("training exploded")

    monkeypatch.setattr(training_api.pipeline, "run_graph", fail)
    job = submit(
        {
            "source": "file",
            "year": 2024,
            "annual_input_path": str(annual),
            "calendar_input_path": str(calendar),
            "output_dir": str(tmp_path / "outputs"),
        }
    )
    assert training_api.get_training_job(job.job_id).error == "training exploded"
    with pytest.raises(HTTPException) as unknown:
        training_api.get_training_job("unknown")
    assert unknown.value.status_code == 404

    with pytest.raises(ValidationError):
        request_adapter.validate_python(
            {
                "source": "quantumleap",
                "from_date": "2024-01-01",
                "to_date": "2024-12-31",
                "output_dir": str(tmp_path / "outputs"),
                "orion": {
                    "forecast_from_date": "2024-12-31",
                    "forecast_to_date": "2025-01-01",
                    "forecast_calendar_path": str(calendar),
                },
            }
        )

    with training_api.jobs_lock:
        training_api.active_job_id = "active-job"
    with pytest.raises(HTTPException) as busy:
        training_api.create_training_job(
            request_adapter.validate_python(
                {
                    "source": "file",
                    "year": 2024,
                    "annual_input_path": str(annual),
                    "calendar_input_path": str(calendar),
                    "output_dir": str(tmp_path / "outputs"),
                }
            ),
            BackgroundTasks(),
        )
    assert busy.value.status_code == 409
    assert busy.value.detail["job_id"] == "active-job"


def test_unwritable_output_directory_is_rejected(tmp_path):
    annual, calendar = local_files(tmp_path)
    read_only = tmp_path / "read-only"
    read_only.mkdir(mode=0o500)
    try:
        with pytest.raises(HTTPException) as rejected:
            training_api.prepare_request(
                request_adapter.validate_python(
                    {
                        "source": "file",
                        "year": 2024,
                        "annual_input_path": str(annual),
                        "calendar_input_path": str(calendar),
                        "output_dir": str(read_only),
                    }
                )
            )
        assert rejected.value.status_code == 422
        assert "writable directory" in rejected.value.detail
    finally:
        read_only.chmod(0o700)
