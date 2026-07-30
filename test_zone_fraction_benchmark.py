from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).parent


def load_module(name: str, relative_path: str):
    spec = spec_from_file_location(name, ROOT / relative_path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader
    spec.loader.exec_module(module)
    return module


benchmark = load_module(
    "zone_fraction_benchmark",
    "model-training/zone_fraction_benchmark.py",
)
pipeline = load_module("complete_parquet_graph", "model-training/ml_pipeline_graph.py")


def test_expanding_validation_folds_keep_future_rows_out_of_training():
    frame = pd.DataFrame({"data": pd.date_range("2024-01-01", periods=8, freq="D")})

    folds = benchmark.expanding_validation_folds(frame, 3)

    assert [
        (
            frame.loc[train_index, "data"].dt.day.tolist(),
            frame.loc[validation_index, "data"].dt.day.tolist(),
        )
        for train_index, validation_index in folds
    ] == [
        ([1, 2], [3, 4]),
        ([1, 2, 3, 4], [5, 6]),
        ([1, 2, 3, 4, 5, 6], [7, 8]),
    ]


def test_sarimax_exog_uses_schedule_intervals_and_italian_holidays():
    group = pd.DataFrame(
        {
            "data": pd.to_datetime(["2024-12-25", "2024-12-27"]),
            "giorni_dal_ritiro_precedente": [7, 2],
        }
    )

    exog = benchmark.daily_sarimax_exog(
        group, date(2024, 12, 24), date(2024, 12, 27)
    )

    assert list(exog) == list(benchmark.SARIMAX_EXOG_COLUMNS)
    assert exog.loc[pd.Timestamp("2024-12-25"), "giorni_dal_ritiro_precedente"] == 7
    assert exog.loc[pd.Timestamp("2024-12-26"), "giorni_dal_ritiro_precedente"] == 0
    assert exog.loc[pd.Timestamp("2024-12-27"), "giorni_dal_ritiro_precedente"] == 2
    assert exog.loc[pd.Timestamp("2024-12-25"), "is_holiday"] == 1
    assert exog.loc[pd.Timestamp("2024-12-27"), "is_after_holiday"] == 1


def test_future_model_selection_uses_validation_wape_then_rmse():
    metrics = pd.DataFrame(
        [
            ("sarima", "validation", "vetro", 0.1, 0.2),
            ("sarimax", "validation", "vetro", 0.1, 0.1),
            ("seasonal_naive_7d", "validation", "vetro", 0.2, 0.01),
            ("sarima", "test", "vetro", 0.001, 0.001),
            ("sarima", "validation", "plastica", 0.3, 0.1),
            ("sarimax", "validation", "plastica", 0.2, 0.2),
        ],
        columns=["model", "split", "frazione", "wape", "rmse"],
    )

    assert benchmark.select_future_models(metrics) == {
        "plastica": "sarimax",
        "vetro": "sarimax",
    }


def test_zone_fraction_future_forecast_is_canonical_and_preserves_group_total(
    tmp_path,
):
    withdrawals = pd.DataFrame(
        {
            "scenario_seed": [42, 42, 42],
            "anno_produzione": [2024, 2024, 2024],
            "data": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
            "zona_raccolta": ["1", "1", "1"],
            "frazione": ["vetro", "vetro", "vetro"],
            "cluster_label": [0, 0, 0],
            "conferimento_t": [3.0, 3.0, 3.0],
            "giorni_dal_ritiro_precedente": [7, 7, 7],
        }
    )
    withdrawals = benchmark.add_features(withdrawals)
    civic = pd.DataFrame(
        {
            "id_civico": ["a", "b"] * 3,
            "data": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-08",
                 "2024-01-15", "2024-01-15"]
            ),
            "zona_raccolta": ["1"] * 6,
            "frazione": ["vetro"] * 6,
            "cluster_label": [0, 1] * 3,
            "conferimento_t": [1.0, 2.0] * 3,
            "coord_x": [16.0, 16.1] * 3,
            "coord_y": [41.0, 41.1] * 3,
            "densita_locale": [100.0, 110.0] * 3,
        }
    )
    withdrawals_path = tmp_path / "withdrawals.parquet"
    civic_path = tmp_path / "civic.parquet"
    con = duckdb.connect()
    benchmark.write_parquet(con, withdrawals, withdrawals_path)
    benchmark.write_parquet(con, civic, civic_path)
    con.close()
    model_key = "scenario_seed=42|zona_raccolta=1|frazione=vetro"
    model_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model_name": "zone_fraction_sarima",
            "models": {model_key: {"model": None}},
            "sarimax_models": {model_key: {"model": None}},
            "civic_allocator": {
                "selected_by_fraction": {
                    "vetro": {"allocator": "historical", "setting": 0.90}
                },
                "models": {},
            },
        },
        model_path,
    )
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame(
        [
            ("baseline_last", "validation", "vetro", 0.01, 0.01),
            ("seasonal_naive_7d", "validation", "vetro", 0.02, 0.02),
            ("sarima", "validation", "vetro", 0.03, 0.03),
        ],
        columns=["model", "split", "frazione", "wape", "rmse"],
    ).to_csv(metrics_path, index=False)
    calendar_path = tmp_path / "calendar.csv"
    calendar_path.write_text(
        "zona_raccolta,frazione,weekday,raccolta\n"
        "1,vetro,lunedi,1\n",
        encoding="utf-8",
    )
    output = tmp_path / "future.parquet"

    benchmark.forecast_arima_benchmark(
        withdrawals_path,
        civic_path,
        model_path,
        metrics_path,
        model_path,
        metrics_path,
        calendar_path,
        output,
        date(2024, 1, 22),
        date(2024, 1, 22),
    )

    result = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?) ORDER BY id_civico", [str(output)]
    ).df()
    assert result["prediction_t"].tolist() == pytest.approx([1.0, 2.0])
    assert result["prediction_t"].sum() == pytest.approx(3.0)
    assert set(result["model"]) == {"baseline_last"}
    assert set(result["pipeline"]) == {"zone-fraction"}


def test_non_finite_arima_forecasts_use_the_seasonal_naive_fallback():
    predicted, methods = benchmark.apply_forecast_fallback(
        np.array([1.0, np.nan, np.inf]),
        pd.Series([10.0, 20.0, 30.0]),
        "sarimax",
    )

    assert predicted.tolist() == [1.0, 20.0, 30.0]
    assert methods.tolist() == ["sarimax", "seasonal_naive_7d", "seasonal_naive_7d"]


def test_arima_aggregates_clusters_to_zone_fraction_and_recomputes_lag():
    frame = pd.DataFrame(
        {
            "scenario_seed": [42, 42, 42, 42],
            "anno_produzione": [2024] * 4,
            "data": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-08"]
            ),
            "zona_raccolta": ["1"] * 4,
            "frazione": ["vetro"] * 4,
            "cluster_label": [0, 1, 0, 1],
            "conferimento_t": [1.0, 2.0, 3.0, 4.0],
            "giorni_dal_ritiro_precedente": [np.nan, np.nan, 7.0, 7.0],
            "lag_7d_collection_t": [np.nan, np.nan, 1.0, 2.0],
        }
    )

    aggregated = benchmark.aggregate_arima_series(frame)

    assert "cluster_label" not in aggregated
    assert aggregated["conferimento_t"].tolist() == [3.0, 7.0]
    assert np.isnan(aggregated.loc[0, "lag_7d_collection_t"])
    assert aggregated.loc[1, "lag_7d_collection_t"] == 3.0


def test_civic_allocator_selection_requires_coverage_before_wape():
    metrics = pd.DataFrame(
        [
            ("historical", 0.9, "validation", "sarima", "vetro", 0.10, 0.91, 1.0),
            ("historical", 0.9, "validation", "sarimax", "vetro", 0.09, 0.92, 1.0),
            (
                "bayesian_ridge",
                1.0,
                "validation",
                "sarima",
                "vetro",
                0.05,
                0.89,
                0.5,
            ),
            (
                "bayesian_ridge",
                1.0,
                "validation",
                "sarimax",
                "vetro",
                0.04,
                0.95,
                0.5,
            ),
        ],
        columns=[
            "allocator",
            "setting",
            "split",
            "forecast_model",
            "frazione",
            "wape",
            "coverage",
            "mean_interval_width_t",
        ],
    )

    selected = benchmark.select_civic_allocators(metrics)

    assert selected["vetro"]["allocator"] == "historical"


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    con = duckdb.connect()
    con.register("frame", frame)
    con.execute("COPY frame TO ? (FORMAT parquet)", [str(path)])


def complete_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    annual_rows = []
    for civic_id, vetro, prg in ((1, 1.0, "A"), (2, 2.0, None)):
        row = {
            "anno": 2024,
            "id_civico": civic_id,
            "zona_raccolta": 1,
            "zona_costruzione": prg,
            "prg_error": None if prg else "missing",
            "zona_raccolta_inferita": False,
            "zona_costruzione_riassegnata": False,
            "distanza_inferenza_zona_raccolta_gradi": 0.0,
            "distanza_riassegnazione_prg": 0.0,
            "totale_t": vetro,
        }
        row.update({column: 0.0 for column in benchmark.ANNUAL_QUANTITY_COLS})
        row["vetro_t"] = vetro
        annual_rows.append(row)
    annual = pd.DataFrame(annual_rows)
    production = pd.DataFrame(
        {
            "anno_produzione": [2024, 2024],
            "data": ["2024-01-01", "2024-01-02"],
            "id_civico": ["1", "2"],
            "zona_raccolta": ["1", "1"],
            "frazione": ["vetro", "vetro"],
            "produzione_teorica_t": [1.0, 2.0],
            "conferimento_data": ["2024-01-01", "2024-01-08"],
            "raccolta_prevista": [1, 0],
        }
    )
    collections = pd.DataFrame(
        {
            "anno_produzione": [2024, 2024],
            "data": ["2024-01-01", "2024-01-08"],
            "id_civico": ["1", "2"],
            "zona_raccolta": ["1", "1"],
            "frazione": ["vetro", "vetro"],
            "conferimento_t": [1.0, 2.0],
            "giorni_accumulati": [1, 7],
            "cluster_label": [0, 0],
        }
    )
    calendar = tmp_path / "calendar.csv"
    calendar.write_text(
        "zona_raccolta,frazione,data\n"
        "1,vetro,2024-01-01\n"
        "1,vetro,2024-01-08\n"
        "1,vetro,2024-01-15\n",
        encoding="utf-8",
    )
    paths = (
        tmp_path / "annual.parquet",
        tmp_path / "production.parquet",
        tmp_path / "collections.parquet",
        calendar,
    )
    write_parquet(paths[0], annual)
    write_parquet(paths[1], production)
    write_parquet(paths[2], collections)
    return paths


def test_classify_collections_adds_civic_cluster_label(tmp_path):
    collections = pd.DataFrame(
        {
            "anno_produzione": [2024, 2024],
            "id_civico": ["1", "2"],
            "conferimento_t": [1.0, 2.0],
        }
    )
    assignments = pd.DataFrame(
        {
            "id_civico": ["1", "2"],
            "anno_from_col": [2024, 2024],
            "cluster_label": [4, 1],
        }
    )
    collections_path = tmp_path / "collections.parquet"
    assignments_path = tmp_path / "assignments.parquet"
    output_path = tmp_path / "classified.parquet"
    write_parquet(collections_path, collections)
    write_parquet(assignments_path, assignments)

    benchmark.classify_collections(
        collections_path, assignments_path, output_path
    )

    classified = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?) ORDER BY id_civico", [str(output_path)]
    ).df()
    assert list(classified.columns) == [
        "anno_produzione",
        "id_civico",
        "conferimento_t",
        "cluster_label",
    ]
    assert classified["cluster_label"].tolist() == [4, 1]


def test_civic_arima_disaggregation_uses_only_pre_test_weights(tmp_path):
    civic_rows = pd.DataFrame(
        [
            (2024, "2024-11-01", "a", "1", "vetro", 0, 1.0),
            (2024, "2024-11-01", "b", "1", "vetro", 0, 3.0),
            (2024, "2024-12-01", "a", "1", "vetro", 0, 2.0),
            (2024, "2024-12-01", "b", "1", "vetro", 0, 6.0),
            (2024, "2024-12-01", "c", "1", "vetro", 1, 1.0),
            (2024, "2024-12-01", "a", "1", "raee", 0, 1.0),
            (2024, "2024-12-01", "b", "1", "raee", 0, 1.0),
            (2024, "2024-12-01", "c", "1", "raee", 1, 1.0),
        ],
        columns=[
            "anno_produzione",
            "data",
            "id_civico",
            "zona_raccolta",
            "frazione",
            "cluster_label",
            "conferimento_t",
        ],
    )
    civic_input = tmp_path / "classified.parquet"
    write_parquet(civic_input, civic_rows)
    group_predictions = pd.DataFrame(
        [
            (42, 2024, "2024-12-01", "1", "vetro", 9.0, 12.0, 8.0, 10.0),
            (42, 2024, "2024-12-01", "1", "raee", 3.0, 6.0, 3.0, 4.5),
        ],
        columns=[
            "scenario_seed",
            "anno_produzione",
            "data",
            "zona_raccolta",
            "frazione",
            "conferimento_t",
            "pred_sarima_t",
            "pred_sarimax_t",
            "pred_seasonal_naive_7d_t",
        ],
    )

    (
        predictions_path,
        metrics_path,
        metrics,
        allocator_metrics_path,
        allocator_bundle,
    ) = (
        benchmark.create_civic_arima_evaluation(
            civic_input,
            group_predictions,
            date(2024, 12, 1),
            date(2024, 12, 31),
            tmp_path / "output",
            2024,
        )
    )

    predictions = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?) ORDER BY frazione, id_civico",
        [str(predictions_path)],
    ).df()
    vetro = predictions[predictions["frazione"] == "vetro"]
    assert vetro["allocation_share"].tolist() == pytest.approx([0.25, 0.75, 0.0])
    assert {
        "pred_civic_sarima_lower_t",
        "pred_civic_sarima_upper_t",
        "allocation_method",
    } <= set(predictions)
    assert (
        predictions["pred_civic_sarima_lower_t"]
        <= predictions["pred_civic_sarima_t"]
    ).all()
    assert (
        predictions["pred_civic_sarima_t"]
        <= predictions["pred_civic_sarima_upper_t"]
    ).all()
    assert vetro["pred_civic_sarima_t"].tolist() == pytest.approx([3.0, 9.0, 0.0])
    assert vetro["actual_conferimento_t"].tolist() == [2.0, 6.0, 1.0]
    assert vetro["cluster_label"].tolist() == [0, 0, 1]
    raee = predictions[predictions["frazione"] == "raee"]
    assert raee["allocation_share"].tolist() == pytest.approx([1 / 3] * 3)
    for civic_col, group_col in (
        ("actual_conferimento_t", "actual_group_total_t"),
        ("pred_civic_sarima_t", "pred_group_sarima_t"),
        ("pred_civic_sarimax_t", "pred_group_sarimax_t"),
        (
            "pred_civic_seasonal_naive_7d_t",
            "pred_group_seasonal_naive_7d_t",
        ),
    ):
        assert predictions.groupby("frazione")[civic_col].sum().tolist() == (
            predictions.groupby("frazione")[group_col].first().tolist()
        )
    assert metrics_path.is_file()
    assert set(metrics["model"]) == {
        "sarima",
        "sarimax",
        "seasonal_naive_7d",
    }
    assert "__macro__" in set(metrics["frazione"])
    assert {"wape_group", "wape_civic", "wape_delta_pp"} <= set(metrics)
    assert {"interval_coverage", "mean_interval_width_t"} <= set(metrics)
    assert allocator_metrics_path.is_file()
    assert allocator_bundle["target_coverage"] == 0.90


def test_civic_allocator_benchmarks_bayesian_candidates(tmp_path):
    rows = []
    values = {
        "2024-06-01": (1.0, 3.0),
        "2024-07-01": (1.2, 2.8),
        "2024-08-01": (0.8, 3.2),
        "2024-09-01": (1.1, 2.9),
        "2024-10-01": (1.3, 2.7),
        "2024-11-01": (0.9, 3.1),
        "2024-12-01": (1.0, 3.0),
    }
    for day, amounts in values.items():
        for civic_id, amount, cluster in zip(
            ("a", "b"), amounts, (0, 1), strict=True
        ):
            rows.append((2024, day, civic_id, "1", "vetro", cluster, amount))
    civic_input = tmp_path / "classified.parquet"
    write_parquet(
        civic_input,
        pd.DataFrame(
            rows,
            columns=[
                "anno_produzione",
                "data",
                "id_civico",
                "zona_raccolta",
                "frazione",
                "cluster_label",
                "conferimento_t",
            ],
        ),
    )
    group_predictions = pd.DataFrame(
        [
            (42, 2024, "2024-10-01", "1", "vetro", 4.0, 4.1, 3.9, 4.0, "validation"),
            (42, 2024, "2024-11-01", "1", "vetro", 4.0, 4.0, 4.1, 4.0, "validation"),
            (42, 2024, "2024-12-01", "1", "vetro", 4.0, 4.2, 4.0, 4.0, "test"),
        ],
        columns=[
            "scenario_seed",
            "anno_produzione",
            "data",
            "zona_raccolta",
            "frazione",
            "conferimento_t",
            "pred_sarima_t",
            "pred_sarimax_t",
            "pred_seasonal_naive_7d_t",
            "split",
        ],
    )

    outputs = benchmark.create_civic_arima_evaluation(
        civic_input,
        group_predictions,
        date(2024, 12, 1),
        date(2024, 12, 31),
        tmp_path / "output",
        2024,
    )

    allocator_metrics = pd.read_csv(outputs[3])
    assert {"historical", "dirichlet", "bayesian_ridge"} <= set(
        allocator_metrics["allocator"]
    )
    assert "selected" in set(allocator_metrics["allocator"])
    assert outputs[4]["selected_by_fraction"]["vetro"]["allocator"] in {
        "historical",
        "dirichlet",
        "bayesian_ridge",
    }
    first_predictions = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?) ORDER BY id_civico",
        [str(outputs[0])],
    ).df()
    changed = pd.DataFrame(
        rows,
        columns=[
            "anno_produzione",
            "data",
            "id_civico",
            "zona_raccolta",
            "frazione",
            "cluster_label",
            "conferimento_t",
        ],
    )
    changed.loc[changed["data"] == "2024-12-01", "conferimento_t"] = 2.0
    changed_input = tmp_path / "changed-classified.parquet"
    write_parquet(changed_input, changed)
    changed_outputs = benchmark.create_civic_arima_evaluation(
        changed_input,
        group_predictions,
        date(2024, 12, 1),
        date(2024, 12, 31),
        tmp_path / "changed-output",
        2024,
    )
    changed_predictions = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?) ORDER BY id_civico",
        [str(changed_outputs[0])],
    ).df()
    pd.testing.assert_frame_equal(
        first_predictions[
            [
                "allocation_share",
                "allocation_share_lower",
                "allocation_share_upper",
                "allocation_method",
            ]
        ],
        changed_predictions[
            [
                "allocation_share",
                "allocation_share_lower",
                "allocation_share_upper",
                "allocation_method",
            ]
        ],
    )


def test_prepare_validates_and_builds_zero_withdrawals(tmp_path):
    annual, production, collections, calendar = complete_fixture(tmp_path)
    outputs = benchmark.prepare_complete_dataset(
        annual_path=annual,
        production_path=production,
        collections_path=collections,
        calendar_path=calendar,
        output_dir=tmp_path / "output",
        year=2024,
        annual_seed=42,
        scenario_seed=42,
        spillover_days=0,
        missing_calendar_mode="error",
        generator_version="fixture",
    )

    prepared = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?) ORDER BY data", [str(outputs.withdrawals)]
    ).df()
    assert prepared["conferimento_t"].tolist() == [1.0, 2.0, 0.0]
    assert prepared["cluster_label"].tolist() == [0, 0, 0]
    assert np.isnan(prepared.loc[0, "lag_1_collection_t"])
    assert prepared["lag_1_collection_t"].iloc[1:].tolist() == [1.0, 2.0]
    assert np.isnan(prepared.loc[0, "lag_7d_collection_t"])
    assert prepared["lag_7d_collection_t"].iloc[1:].tolist() == [1.0, 2.0]
    assert prepared["rolling_7d_mean_t"].iloc[1:].tolist() == [1.0, 2.0]
    assert prepared["rolling_7d_count"].tolist() == [0, 1, 1]
    assert prepared["giorni_dal_ritiro_precedente"].iloc[1:].tolist() == [7, 7]
    assert not prepared.duplicated(
        ["scenario_seed", "data", "zona_raccolta", "frazione"]
    ).any()

    static = duckdb.connect().execute(
        "SELECT id_civico, zona_raccolta, zona_costruzione "
        "FROM read_parquet(?) ORDER BY id_civico",
        [str(outputs.static_civics)],
    ).fetchall()
    assert static == [("1", "1", "A"), ("2", "1", None)]
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["validation"]["unique_civics"] == 2
    assert manifest["validation"]["missing_prg"] == 1
    assert manifest["validation"]["scheduled_withdrawals"] == 3
    assert manifest["validation"]["zero_withdrawals"] == 1
    assert manifest["validation"]["total_t"] == 3.0


def test_prepare_can_skip_production_validation(tmp_path):
    annual, _, collections, calendar = complete_fixture(tmp_path)
    annual_frame = duckdb.connect().execute(
        "SELECT * EXCLUDE (prg_error, zona_raccolta_inferita, "
        "zona_costruzione_riassegnata, "
        "distanza_inferenza_zona_raccolta_gradi, "
        "distanza_riassegnazione_prg) FROM read_parquet(?)",
        [str(annual)],
    ).df()
    write_parquet(annual, annual_frame)

    outputs = benchmark.prepare_complete_dataset(
        annual_path=annual,
        production_path=None,
        collections_path=collections,
        calendar_path=calendar,
        output_dir=tmp_path / "output",
        year=2024,
        annual_seed=42,
        scenario_seed=42,
        spillover_days=0,
        missing_calendar_mode="error",
        generator_version="collections-only-fixture",
    )

    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert "production" not in manifest["inputs"]
    assert manifest["validation"]["total_t"] == 3.0


def test_prepare_rejects_unknown_fraction(tmp_path):
    annual, production, collections, calendar = complete_fixture(tmp_path)
    con = duckdb.connect()
    unknown = con.execute("SELECT * FROM read_parquet(?)", [str(production)]).df()
    unknown.loc[0, "frazione"] = "mystery"
    write_parquet(production, unknown)

    with pytest.raises(ValueError, match="unknown fractions"):
        benchmark.prepare_complete_dataset(
            annual_path=annual,
            production_path=production,
            collections_path=collections,
            calendar_path=calendar,
            output_dir=tmp_path / "output",
            year=2024,
            annual_seed=42,
            scenario_seed=42,
            spillover_days=0,
            missing_calendar_mode="error",
            generator_version="fixture",
        )


def test_training_uses_chronological_splits_and_writes_artifacts(
    tmp_path, capsys, monkeypatch
):
    dates = pd.date_range("2024-01-01", "2025-01-13", freq="7D")
    rows = []
    for fraction in ("vetro", "raee"):
        for index, day in enumerate(dates):
            rows.append(
                {
                    "scenario_seed": 42,
                    "anno_produzione": 2024,
                    "data": day,
                    "zona_raccolta": "1",
                    "frazione": fraction,
                    "cluster_label": 0,
                    "conferimento_t": (
                        10.0 + np.sin(index / 3)
                        if fraction == "vetro"
                        else float(index % 4 == 0)
                    ),
                    "giorni_dal_ritiro_precedente": (
                        pd.NA if index == 0 else 7
                    ),
                }
            )
    prepared = benchmark.add_features(pd.DataFrame(rows))
    input_path = tmp_path / "withdrawals.parquet"
    write_parquet(input_path, prepared)

    outputs = benchmark.train_benchmark(input_path, tmp_path / "output", seed=42)
    assert outputs["selected_model"] in {
        "ridge",
        "sgd",
        "hist_gradient_boosting",
    }
    assert outputs["model"].is_file()
    assert outputs["metrics"].is_file()
    assert outputs["predictions"].is_file()
    bundle = joblib.load(outputs["model"])
    assert bundle["split_strategy"] == "Jan-Sep train, Oct-Nov validation, Dec test"
    assert (
        bundle["models"]["hist_gradient_boosting"].named_steps["model"].loss
        == "poisson"
    )
    assert (
        bundle["models"]["sgd"]
        .named_steps["model"]
        .regressor.max_iter
        == 3
    )

    metrics = pd.read_csv(outputs["metrics"])
    assert {
        "baseline_last",
        "ridge",
        "sgd",
        "hist_gradient_boosting",
        "hist_quantiles",
    } <= set(metrics["model"])
    assert {"train", "validation", "test"} == set(metrics["split"])
    assert "__macro__" not in set(metrics["frazione"])
    predictions = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?)", [str(outputs["predictions"])]
    ).df()
    assert predictions["data"].min().date() >= date(2024, 12, 1)
    assert predictions["data"].max().date() <= date(2024, 12, 31)
    assert (predictions["selected_prediction_t"] >= 0).all()
    assert "pred_sgd_t" in predictions
    assert predictions.loc[
        predictions["frazione"] == "raee", "pred_q90_t"
    ].notna().all()
    episodic = predictions["frazione"] == "raee"
    assert (
        predictions.loc[episodic, "pred_q90_t"]
        >= predictions.loc[episodic, "pred_q50_t"]
    ).all()
    summary = capsys.readouterr().out
    assert "Training summary" in summary
    assert "elapsed:" in summary
    assert "validation fraction-average:" in summary
    assert "test fraction-average:" in summary
    assert "sgd validation:" in summary
    assert "sgd test:" in summary
    assert "hyperparameters:" in summary

    excluded = benchmark.train_benchmark(
        input_path,
        tmp_path / "continuous-output",
        seed=42,
        exclude_episodic_fractions=True,
    )
    excluded_metrics = pd.read_csv(excluded["metrics"])
    assert set(excluded_metrics["frazione"]) == {"vetro"}
    assert "hist_quantiles" not in set(excluded_metrics["model"])
    excluded_predictions = duckdb.connect().execute(
        "SELECT DISTINCT frazione FROM read_parquet(?)",
        [str(excluded["predictions"])],
    ).fetchall()
    assert excluded_predictions == [("vetro",)]

    cross_validated = benchmark.train_benchmark(
        input_path,
        tmp_path / "cross-validated-output",
        seed=42,
        exclude_episodic_fractions=True,
        validation_strategy="k-fold",
        validation_folds=2,
    )
    cross_validated_bundle = joblib.load(cross_validated["model"])
    assert cross_validated_bundle["validation_strategy"] == "k-fold"
    assert len(cross_validated_bundle["validation_folds"]) == 2
    assert all(
        fold["train_to"] < fold["validation_from"]
        for fold in cross_validated_bundle["validation_folds"]
    )

    monkeypatch.setattr(benchmark, "ARIMA_ORDERS", ((0, 0, 0),))
    monkeypatch.setattr(benchmark, "ARIMA_SEASONAL_ORDERS", ((0, 0, 0, 0),))
    arima = benchmark.train_arima_benchmark(
        input_path,
        tmp_path / "arima-output",
        exclude_episodic_fractions=True,
    )
    arima_metrics = pd.read_csv(arima["metrics"])
    assert set(arima_metrics["model"]) == {
        "sarima",
        "sarimax",
        "seasonal_naive_7d",
    }
    assert set(arima_metrics["frazione"]) == {"vetro"}
    assert "__macro__" not in set(arima_metrics["frazione"])
    arima_bundle = joblib.load(arima["model"])
    assert (
        arima_bundle["split_strategy"]
        == "Jan-Sep train, Oct-Nov validation, Dec test"
    )
    assert arima_bundle["model_name"] == "zone_fraction_sarima"
    assert arima_bundle["model_variants"] == ("sarima", "sarimax")
    assert arima_bundle["group_columns"] == ["zona_raccolta", "frazione"]
    assert arima_bundle["sarimax_exog_columns"] == benchmark.SARIMAX_EXOG_COLUMNS
    assert set(arima_bundle["models"]) == set(arima_bundle["sarimax_models"])
    assert (
        arima_bundle["sarimax_test_metrics"] == arima["sarimax_test_metrics"]
    )
    assert arima["predictions"].is_file()
    arima_predictions = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?)", [str(arima["predictions"])]
    ).df()
    assert arima_predictions["data"].max().date() <= date(2024, 12, 31)
    assert "pred_sarimax_t" in arima_predictions
    assert "cluster_label" not in arima_predictions

    cross_validated_arima = benchmark.train_arima_benchmark(
        input_path,
        tmp_path / "cross-validated-arima-output",
        exclude_episodic_fractions=True,
        validation_strategy="k-fold",
        validation_folds=2,
    )
    cross_validated_arima_bundle = joblib.load(cross_validated_arima["model"])
    assert cross_validated_arima_bundle["validation_strategy"] == "k-fold"
    assert (
        cross_validated_arima_bundle["split_strategy"]
        == "2-fold expanding-window validation, Dec test"
    )


def test_arima_parser_accepts_validation_strategy_flags():
    args = benchmark.build_parser().parse_args(
        [
            "arima",
            "--input",
            "withdrawals.parquet",
            "--civic-input",
            "classified.parquet",
            "--output-dir",
            "artifacts",
            "--validation-strategy",
            "k-fold",
            "--validation-folds",
            "5",
        ]
    )

    assert args.validation_strategy == "k-fold"
    assert args.validation_folds == 5
    assert args.civic_input == Path("classified.parquet")


def test_complete_parquet_graph_is_separate_from_legacy(tmp_path):
    graph = pipeline.build_complete_parquet_graph(
        "python",
        "7",
        "2024",
        tmp_path / "out",
        Path("annual.parquet"),
        Path("production.parquet"),
        Path("collections.parquet"),
        Path("calendar.csv"),
        scenario_seed=99,
        exclude_episodic_fractions=True,
        hgb_loss="squared_error",
        include_arima=True,
        validation_strategy="k-fold",
        validation_folds=4,
    )
    assert list(graph) == [
        "kmeans",
        "classify_collections",
        "prepare_zone_fraction",
        "train_zone_fraction",
        "train_zone_fraction_arima",
        "finalize_run",
    ]
    assert graph["classify_collections"].needs == ("kmeans",)
    assert graph["prepare_zone_fraction"].needs == ("classify_collections",)
    assert graph["train_zone_fraction"].needs == ("prepare_zone_fraction",)
    prepare = graph["prepare_zone_fraction"].command
    assert prepare[prepare.index("--scenario-seed") + 1] == "99"
    train = graph["train_zone_fraction"].command
    assert train[train.index("--seed") + 1] == "7"
    assert train[train.index("--hgb-loss") + 1] == "squared_error"
    assert train[train.index("--validation-strategy") + 1] == "k-fold"
    assert train[train.index("--validation-folds") + 1] == "4"
    assert "--exclude-episodic-fractions" in train
    arima = graph["train_zone_fraction_arima"]
    assert arima.needs == ("prepare_zone_fraction",)
    assert "--civic-input" in arima.command
    assert arima.command[arima.command.index("--civic-input") + 1].endswith(
        "conferimenti_giornalieri_classificati.parquet"
    )
    assert any(
        path.name == "zone_fraction_arima_test_predictions_per_civic.parquet"
        for path in arima.outputs
    )
    assert any(
        path.name == "zone_fraction_arima_civic_metrics.csv"
        for path in arima.outputs
    )
    assert any(
        path.name == "zone_fraction_arima_allocator_benchmark.csv"
        for path in arima.outputs
    )
    assert "--exclude-episodic-fractions" in arima.command
    assert arima.command[arima.command.index("--validation-strategy") + 1] == "k-fold"
    assert arima.command[arima.command.index("--validation-folds") + 1] == "4"
    assert list(pipeline.build_graph("python", "42", "2024")) == [
        "kmeans",
        "daily_distribution",
        "grouped_sgd",
        "finalize_run",
    ]
