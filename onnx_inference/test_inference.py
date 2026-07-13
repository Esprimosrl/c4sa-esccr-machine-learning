from __future__ import annotations

from pathlib import Path
import math

import joblib
import numpy as np
import pytest

from onnx_inference.app import (
    ModelPackage,
    PredictionInput,
    UnknownGroup,
    feature_values,
)
from onnx_inference.export_models import export_bundle


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
