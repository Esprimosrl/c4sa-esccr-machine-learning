from __future__ import annotations

from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from onnx_inference.app import (
    ModelPackage,
    PredictionInput,
    UnknownGroup,
    ZoneFractionForecastRequest,
    ZoneFractionPackage,
    ZoneFractionPredictionInput,
    feature_values,
)
from onnx_inference.export_models import export_bundle, export_zone_fraction_bundle
from zone_fraction_benchmark import fit_sarima


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = ROOT / "models/grouped_sgd_time_series_72modelli.joblib"


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("onnx") / "package"
    manifest = export_bundle(SOURCE_MODEL, output)
    assert len(manifest["groups"]) == 72
    assert output.stat().st_mode & 0o555 == 0o555
    return output


def test_calendar_features_match_training_conventions() -> None:
    instance = PredictionInput(
        data="2024-02-29",
        anno_produzione=2023,
        coord_x=16.0,
        coord_y=41.0,
        densita_locale=100.0,
        giorni_accumulati=2,
        zona_raccolta=3,
        frazione="altro_rd",
        cluster_label=0,
    )
    values = feature_values(instance)
    assert values["anno_ref"] == 2023
    assert values["day_of_year"] == 60
    assert values["day_of_week"] == float(instance.data.strftime("%w"))
    assert values["week_of_year"] == float(instance.data.strftime("%W"))
    assert values["quarter"] == 1
    assert values["season_sin"] == pytest.approx(
        math.sin(2 * math.pi * 60 / 366.0)
    )
    assert values["zona_raccolta=3"] == 1
    assert values["zona_raccolta=2"] == 0


def sklearn_prediction(bundle: dict, instance: PredictionInput) -> float:
    key = (
        f"frazione={instance.frazione}|cluster_label={instance.cluster_label}"
    )
    info = bundle["models"][key]
    raw = feature_values(instance)
    features = []
    for name in bundle["numeric_cols"]:
        mean, std = info["numeric_stats"][name]
        features.append((raw[name] - mean) / std)
    features.extend(raw[name] for name in bundle["feature_names"][len(features) :])
    pred_log = info["model"].predict(np.asarray([features], dtype=np.float32))
    return float(max(np.expm1(pred_log[0]), 0.0))


def test_package_routes_groups_and_matches_sklearn(model_dir: Path) -> None:
    bundle = joblib.load(SOURCE_MODEL)
    instances = [
        PredictionInput(
            data="2024-09-10",
            anno_produzione=2024,
            coord_x=16.2896597,
            coord_y=41.2407249,
            densita_locale=384.0,
            giorni_accumulati=3,
            zona_raccolta=1,
            frazione="altro_rd",
            cluster_label=5,
        ),
        PredictionInput(
            data="2025-01-03",
            anno_produzione=2024,
            coord_x=16.31,
            coord_y=41.23,
            densita_locale=950.0,
            giorni_accumulati=4,
            zona_raccolta=2,
            frazione="vetro",
            cluster_label=0,
        ),
    ]

    package = ModelPackage.load(model_dir)
    assert len(package.sessions) == 72
    actual = package.predict(instances)
    expected = [sklearn_prediction(bundle, item) for item in instances]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    unknown = instances[0].model_copy(update={"cluster_label": 999})
    with pytest.raises(UnknownGroup, match="cluster_label=999"):
        package.predict([unknown])


def test_missing_package_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ModelPackage.load(tmp_path)


def test_gpr_bundle_exports_to_onnx(tmp_path: Path) -> None:
    numeric_cols = [
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
    ]
    feature_names = numeric_cols + [f"zona_raccolta={level}" for level in range(1, 5)]
    rng = np.random.default_rng(42)
    features = rng.normal(size=(20, len(feature_names))).astype(np.float32)
    target = rng.normal(size=20)
    estimator = GaussianProcessRegressor(
        kernel=(
            ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(1.0, (1e-2, 1e2))
            + WhiteKernel(0.1, (1e-5, 1e1))
        ),
        alpha=1e-6,
        normalize_y=True,
        random_state=42,
    ).fit(features, target)
    source = tmp_path / "grouped_gpr.joblib"
    joblib.dump(
        {
            "model_name": "grouped_gpr",
            "numeric_cols": numeric_cols,
            "feature_names": feature_names,
            "models": {
                "frazione=vetro|cluster_label=0": {
                    "model": estimator,
                    "numeric_stats": {name: (0.0, 1.0) for name in numeric_cols},
                    "group_values": {"frazione": "vetro", "cluster_label": "0"},
                    "prediction_upper_bound_t": 0.05,
                }
            },
        },
        source,
    )

    output = tmp_path / "package"
    manifest = export_bundle(source, output)

    assert len(manifest["groups"]) == 1
    assert manifest["groups"][0]["prediction_upper_bound_t"] == 0.05
    assert (output / "models/000.onnx").exists()


def test_hist_gradient_boosting_bundle_exports_to_onnx(tmp_path: Path) -> None:
    numeric_cols = [
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
    ]
    feature_names = numeric_cols + [f"zona_raccolta={level}" for level in range(1, 5)]
    rng = np.random.default_rng(42)
    features = rng.normal(size=(100, len(feature_names))).astype(np.float32)
    target = np.exp(rng.normal(size=100))
    estimator = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=5,
        min_samples_leaf=2,
        random_state=42,
    ).fit(features, target)
    source = tmp_path / "grouped_hist.joblib"
    joblib.dump(
        {
            "model_name": "grouped_hist_gradient_boosting",
            "target_transform": "identity",
            "numeric_cols": numeric_cols,
            "feature_names": feature_names,
            "models": {
                "frazione=vetro|cluster_label=0": {
                    "model": estimator,
                    "numeric_stats": {name: (0.0, 1.0) for name in numeric_cols},
                    "group_values": {"frazione": "vetro", "cluster_label": "0"},
                }
            },
        },
        source,
    )

    output = tmp_path / "hist-package"
    manifest = export_bundle(source, output)

    assert len(manifest["groups"]) == 1
    assert (output / "models/000.onnx").exists()


@pytest.mark.parametrize("selected_model", ["hist_gradient_boosting", "ridge"])
def test_zone_fraction_package_serves_onnx_and_native_sarimax(
    tmp_path: Path,
    selected_model: str,
) -> None:
    numeric = ["giorni_dal_ritiro_precedente"]
    categorical = ["zona_raccolta", "frazione", "cluster_label"]
    training = pd.DataFrame(
        {
            "giorni_dal_ritiro_precedente": np.tile(
                [2.0, 3.0, np.nan, 4.0], 10
            ),
            "zona_raccolta": np.tile(["1", "2"], 20),
            "frazione": np.tile(["vetro", "carta_cartone"], 20),
            "cluster_label": np.tile([0, 1], 20),
        }
    )
    target = np.linspace(0.1, 1.0, len(training))
    if selected_model == "hist_gradient_boosting":
        numeric_transformer = SimpleImputer(
            strategy="median",
            add_indicator=True,
        )
        categorical_transformer = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        estimator = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=5,
            min_samples_leaf=2,
            random_state=42,
        )
    else:
        numeric_transformer = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True),
                ),
                ("scale", StandardScaler()),
            ]
        )
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
        estimator = Ridge()
    tabular_model = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numeric", numeric_transformer, numeric),
                        ("categorical", categorical_transformer, categorical),
                    ],
                    sparse_threshold=0.0,
                ),
            ),
            ("model", estimator),
        ]
    ).fit(training, target)
    tabular_path = tmp_path / "zone_fraction_benchmark.joblib"
    joblib.dump(
        {
            "contract": "zone-fraction-scheduled-withdrawal-v1",
            "selected_model": selected_model,
            "models": {selected_model: tabular_model},
            "numeric_features": numeric,
            "categorical_features": categorical,
        },
        tabular_path,
    )

    dates = pd.date_range("2024-01-01", periods=21, freq="D")
    exog = pd.DataFrame(
        {
            "giorni_dal_ritiro_precedente": np.tile([6.0, 7.0, 8.0], 7),
            "is_holiday": 0.0,
            "is_after_holiday": 0.0,
        },
        index=dates,
    )
    sarimax_model = fit_sarima(
        pd.Series(np.linspace(0.1, 0.5, len(dates)), index=dates),
        (0, 0, 0),
        (0, 0, 0, 0),
        exog,
    )
    sarimax_path = tmp_path / "zone_fraction_arima.joblib"
    joblib.dump(
        {
            "contract": "zone-fraction-scheduled-withdrawal-v1",
            "model_name": "zone_fraction_sarima",
            "sarimax_exog_columns": tuple(exog.columns),
            "sarimax_models": {
                "scenario_seed=42|zona_raccolta=1|frazione=vetro|cluster_label=0": {
                    "model": sarimax_model
                }
            },
        },
        sarimax_path,
    )

    model_dir = tmp_path / "zone-package"
    manifest = export_zone_fraction_bundle(
        tabular_path, sarimax_path, model_dir
    )
    assert (model_dir / "tabular_preprocessor.joblib").exists()
    assert manifest["tabular"]["transformed_feature_count"] > len(
        numeric + categorical
    )
    package = ZoneFractionPackage.load(model_dir, manifest)
    instance = ZoneFractionPredictionInput(
        giorni_dal_ritiro_precedente=None,
        zona_raccolta="1",
        frazione="vetro",
        cluster_label=0,
    )
    expected = max(float(tabular_model.predict(pd.DataFrame([{
        "giorni_dal_ritiro_precedente": np.nan,
        "zona_raccolta": "1",
        "frazione": "vetro",
        "cluster_label": 0,
    }]))[0]), 0.0)
    assert package.predict([instance])[0] == pytest.approx(expected, rel=1e-5)

    calendar = tmp_path / "calendar.csv"
    calendar.write_text(
        "zona_raccolta,frazione,weekday\n1,vetro,lunedi\n",
        encoding="utf-8",
    )
    forecasts = package.forecast(
        ZoneFractionForecastRequest(
            forecast_from_date="2024-01-22",
            forecast_to_date="2024-01-28",
        ),
        calendar,
    )
    assert forecasts
    assert {row["frazione"] for row in forecasts} == {"vetro"}
    assert all(np.isfinite(row["prediction_t"]) for row in forecasts)
