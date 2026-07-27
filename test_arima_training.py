from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import importlib.util
import subprocess
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).parent


def load_trainer():
    path = ROOT / "model-training/regression/train_grouped_arima_time_series.py"
    spec = importlib.util.spec_from_file_location("grouped_arima_trainer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_trainer()


def test_weekly_naive_and_three_expanding_validation_folds():
    folds = trainer.rolling_validation_folds(28, 31)

    assert folds == [
        trainer.RollingFold(22, 25),
        trainer.RollingFold(25, 28),
        trainer.RollingFold(28, 31),
    ]
    assert trainer.seasonal_naive(np.arange(14, dtype=float), 10).tolist() == [
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        7.0,
        8.0,
        9.0,
    ]


def test_daily_series_discovers_six_by_twelve_and_fills_missing_days():
    con = duckdb.connect()
    rows = [
        ("1", date(2024, 1, 1), fraction, str(cluster), 1.0)
        for fraction in [f"fraction_{index}" for index in range(12)]
        for cluster in range(6)
    ]
    rows.append(("1", date(2024, 1, 3), "fraction_0", "0", 2.0))
    con.execute("""
        CREATE TABLE observations (
            id_civico VARCHAR,
            data DATE,
            frazione VARCHAR,
            cluster_label VARCHAR,
            target_value DOUBLE
        )
    """)
    con.executemany("INSERT INTO observations VALUES (?, ?, ?, ?, ?)", rows)

    calendar, groups, series = trainer.load_daily_series(con)

    assert len(groups) == 72
    assert len(calendar) == 3
    assert series["frazione=fraction_0|cluster_label=0"].tolist() == [1.0, 0.0, 2.0]


def test_daily_series_discovers_four_by_twelve_zone_groups():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE observations (
            data DATE,
            frazione VARCHAR,
            zona_raccolta VARCHAR,
            target_value DOUBLE
        )
    """)
    con.executemany(
        "INSERT INTO observations VALUES (DATE '2024-01-01', ?, ?, 1.0)",
        [
            (f"fraction_{fraction}", str(zone))
            for fraction in range(12)
            for zone in range(1, 5)
        ],
    )

    calendar, groups, series = trainer.load_daily_series(con, "zone")

    assert len(calendar) == 1
    assert len(groups) == 48
    assert series["frazione=fraction_0|zona_raccolta=1"].tolist() == [1.0]


def test_civic_allocation_preserves_group_total_and_uses_history(tmp_path):
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE observations (
            id_civico VARCHAR,
            data DATE,
            frazione VARCHAR,
            cluster_label VARCHAR,
            zona_raccolta VARCHAR,
            target_value DOUBLE
        )
    """)
    con.executemany(
        "INSERT INTO observations VALUES (?, ?, 'vetro', '0', '1', ?)",
        [
            ("a", date(2024, 1, 1), 1.0),
            ("b", date(2024, 1, 1), 3.0),
            ("a", date(2024, 1, 2), 2.0),
            ("b", date(2024, 1, 2), 6.0),
        ],
    )
    forecasts = pd.DataFrame(
        [(date(2024, 1, 2), "vetro", "0", 12.0)],
        columns=["data", "frazione", "cluster_label", "pred_group_total_t"],
    )

    trainer.create_test_predictions(
        con, forecasts, pd.Timestamp("2024-01-02"), tmp_path / "predictions.parquet"
    )
    predictions = con.execute(
        "SELECT * FROM read_parquet(?) ORDER BY id_civico",
        [str(tmp_path / "predictions.parquet")],
    ).df()

    assert predictions["allocation_share"].tolist() == pytest.approx([0.25, 0.75])
    assert predictions["pred_conferimento_t"].tolist() == pytest.approx([3.0, 9.0])
    assert predictions["pred_conferimento_t"].sum() == pytest.approx(12.0)


def test_civic_allocation_uses_equal_shares_without_history(tmp_path):
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE observations (
            id_civico VARCHAR,
            data DATE,
            frazione VARCHAR,
            cluster_label VARCHAR,
            zona_raccolta VARCHAR,
            target_value DOUBLE
        )
    """)
    con.executemany(
        "INSERT INTO observations VALUES (?, DATE '2024-01-02', 'vetro', '0', '1', 1.0)",
        [("a",), ("b",)],
    )
    forecasts = pd.DataFrame(
        [(date(2024, 1, 2), "vetro", "0", 10.0)],
        columns=["data", "frazione", "cluster_label", "pred_group_total_t"],
    )

    trainer.create_test_predictions(
        con, forecasts, pd.Timestamp("2024-01-02"), tmp_path / "predictions.parquet"
    )
    predictions = con.execute(
        "SELECT allocation_share, pred_conferimento_t FROM read_parquet(?) ORDER BY id_civico",
        [str(tmp_path / "predictions.parquet")],
    ).fetchall()

    assert predictions == [(0.5, 5.0), (0.5, 5.0)]


def test_future_forecast_falls_back_to_weekly_naive_when_sarima_is_non_finite(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        trainer, "forecast_raw", lambda model, steps: np.full(steps, np.nan)
    )
    group = trainer.GroupSpec(
        "frazione=vetro|cluster_label=0", "vetro", "cluster_label", "0"
    )
    model_info = {
        "model": object(),
        "seasonal_naive_history": np.arange(1, 8, dtype=float),
    }
    output = tmp_path / "future.parquet"

    trainer.create_future_predictions(
        duckdb.connect(),
        [group],
        {group.key: model_info},
        pd.Timestamp("2024-01-07"),
        date(2024, 1, 8),
        date(2024, 1, 17),
        None,
        output,
        group_level=True,
    )

    future = duckdb.connect().execute(
        "SELECT pred_group_total_t, forecast_method FROM read_parquet(?) ORDER BY data",
        [str(output)],
    ).fetchall()
    assert future == [
        (value, "seasonal_naive")
        for value in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0]
    ]
    assert model_info["future_forecast_method"] == "seasonal_naive"


def test_tiny_arima_training_writes_test_and_future_civic_forecasts(tmp_path):
    start = date(2024, 1, 1)
    rows = []
    for day_index in range(35):
        event_date = start + timedelta(days=day_index)
        seasonal = [0.5, 1.0, 1.5, 1.0, 0.5, 0.25, 0.25][day_index % 7]
        for civic, weight in (("a", 1.0), ("b", 3.0)):
            rows.append(
                {
                    "id_civico": civic,
                    "data": event_date,
                    "anno_produzione": 2024,
                    "frazione": "vetro",
                    "cluster_label": 0,
                    "zona_raccolta": 1,
                    "conferimento_t": seasonal * weight,
                    "historical_annual_t": 10.0 * weight,
                }
            )
    input_path = tmp_path / "history.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame(
        {
            "zona_raccolta": [1] * 7,
            "frazione": ["vetro"] * 7,
            "weekday": [
                "lunedi",
                "martedi",
                "mercoledi",
                "giovedi",
                "venerdi",
                "sabato",
                "domenica",
            ],
        }
    ).to_csv(calendar, index=False)
    model = tmp_path / "arima.joblib"
    metrics = tmp_path / "metrics.csv"
    test_predictions = tmp_path / "test.parquet"
    future_predictions = tmp_path / "future.parquet"

    subprocess.run(
        [
            sys.executable,
            "model-training/regression/train_grouped_arima_time_series.py",
            "--input",
            str(input_path),
            "--model-output",
            str(model),
            "--metrics-output",
            str(metrics),
            "--predictions-output",
            str(test_predictions),
            "--forecast-from-date",
            "2024-02-05",
            "--forecast-to-date",
            "2024-02-06",
            "--forecast-calendar",
            str(calendar),
            "--future-predictions-output",
            str(future_predictions),
        ],
        cwd=ROOT,
        check=True,
    )

    bundle = joblib.load(model)
    assert bundle["model_name"] == "grouped_arima"
    assert len(bundle["models"]) == 1
    assert len(bundle["rolling_validation_folds"]) == 3
    fitted = next(iter(bundle["models"].values()))
    assert len(fitted["rolling_validation_rmse"]) == 3
    assert len(fitted["seasonal_naive_rolling_validation_rmse"]) == 3
    metric_rows = pd.read_csv(metrics)
    group_metrics = metric_rows[metric_rows["level"] == "group_total"]
    assert set(group_metrics["forecast_method"]) == {"sarima", "seasonal_naive"}
    assert metrics.exists() and test_predictions.exists() and future_predictions.exists()
    future = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?) ORDER BY data, id_civico",
        [str(future_predictions)],
    ).df()
    assert len(future) == 4
    assert future["pred_conferimento_t"].ge(0).all()
    assert future.groupby("data")["allocation_share"].sum().tolist() == pytest.approx(
        [1.0, 1.0]
    )

    grouped_model = tmp_path / "arima_grouped.joblib"
    grouped_metrics = tmp_path / "grouped_metrics.csv"
    grouped_test = tmp_path / "grouped_test.parquet"
    grouped_future = tmp_path / "grouped_future.parquet"
    subprocess.run(
        [
            sys.executable,
            "model-training/regression/train_grouped_arima_time_series.py",
            "--group-level",
            "--input",
            str(input_path),
            "--model-output",
            str(grouped_model),
            "--metrics-output",
            str(grouped_metrics),
            "--predictions-output",
            str(grouped_test),
            "--forecast-from-date",
            "2024-02-05",
            "--forecast-to-date",
            "2024-02-06",
            "--future-predictions-output",
            str(grouped_future),
        ],
        cwd=ROOT,
        check=True,
    )
    grouped_bundle = joblib.load(grouped_model)
    assert grouped_bundle["prediction_level"] == "group"
    grouped_test_rows = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?)", [str(grouped_test)]
    ).df()
    grouped_future_rows = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?)", [str(grouped_future)]
    ).df()
    assert len(grouped_test_rows) == 4
    assert len(grouped_future_rows) == 2
    assert "id_civico" not in grouped_test_rows
    assert "actual_group_total_t" in grouped_test_rows
    assert "seasonal_naive_group_total_t" in grouped_test_rows
    assert "pred_group_total_t" in grouped_future_rows

    zone_model = tmp_path / "arima_zone.joblib"
    zone_metrics = tmp_path / "zone_metrics.csv"
    zone_test = tmp_path / "zone_test.parquet"
    zone_future = tmp_path / "zone_future.parquet"
    subprocess.run(
        [
            sys.executable,
            "model-training/regression/train_grouped_arima_time_series.py",
            "--group-level",
            "--group-by",
            "zone",
            "--input",
            str(input_path),
            "--model-output",
            str(zone_model),
            "--metrics-output",
            str(zone_metrics),
            "--predictions-output",
            str(zone_test),
            "--forecast-from-date",
            "2024-02-05",
            "--forecast-to-date",
            "2024-02-06",
            "--future-predictions-output",
            str(zone_future),
        ],
        cwd=ROOT,
        check=True,
    )
    zone_bundle = joblib.load(zone_model)
    assert zone_bundle["group_cols"] == ["frazione", "zona_raccolta"]
    zone_test_rows = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?)", [str(zone_test)]
    ).df()
    zone_future_rows = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?)", [str(zone_future)]
    ).df()
    assert "zona_raccolta" in zone_test_rows
    assert "cluster_label" not in zone_test_rows
    assert "zona_raccolta" in zone_future_rows
