from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from pydantic import ValidationError

from onnx_clustering.app import (
    ModelPackage,
    PredictionInput,
    UnknownCategory,
)
from onnx_clustering.export_model import export_bundle


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = ROOT / "models/kmeans_model_annuo_peso_coord.joblib"


def instance(**updates) -> PredictionInput:
    values = {
        "anno_produzione": 2024,
        "coord_x": 16.3149866,
        "coord_y": 41.2333986,
        "densita_locale": 397.0,
        "altro_rd_t": 4.47,
        "carta_cartone_t": 76.623,
        "frazione_organica_t": 266.098,
        "ingombranti_misti_t": 139.054,
        "legno_t": 0.0,
        "metallo_t": 2.527,
        "plastica_t": 22.206,
        "pulizia_stradale_recupero_t": 0.0,
        "raee_t": 0.0,
        "rifiuti_cd_t": 0.0,
        "selettiva_t": 0.0,
        "tessili_t": 0.0,
        "vetro_t": 28.849,
        "zona_raccolta": 2,
        "zona_costruzione": "B3R",
    }
    values.update(updates)
    return PredictionInput(**values)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("clustering-onnx") / "package"
    manifest = export_bundle(SOURCE_MODEL, output)
    assert manifest["n_clusters"] == 6
    assert output.stat().st_mode & 0o555 == 0o555
    return output


def sklearn_prediction(bundle: dict, item: PredictionInput) -> tuple[int, float]:
    raw = item.model_dump()
    numeric = []
    for name in bundle["numeric_cols"]:
        if name == "anno_ref":
            value = item.anno_produzione
        elif name == "giorni_accumulati":
            value = bundle["numeric_stats"][name][0]
        else:
            value = raw[name]
        mean, scale = bundle["numeric_stats"][name]
        numeric.append((float(value) - mean) / scale)

    categories = {
        "zona_raccolta": str(item.zona_raccolta),
        "zona_costruzione": (
            item.zona_costruzione.strip()
            if item.zona_costruzione is not None
            else None
        ),
    }
    encoded = [
        float(categories[name] == level)
        for name in bundle["categorical_cols"]
        for level in bundle["categorical_levels"][name]
    ]
    features = np.asarray([numeric + encoded], dtype=np.float32)
    features *= bundle["feature_weights"]
    model = bundle["model"]
    return int(model.predict(features)[0]), float(model.transform(features).min())


def test_package_matches_sklearn_and_preserves_order(model_dir: Path) -> None:
    bundle = joblib.load(SOURCE_MODEL)
    instances = [
        instance(),
        instance(
            coord_x=16.29,
            coord_y=41.22,
            zona_raccolta=1,
            zona_costruzione=None,
            altro_rd_t=15.0,
            vetro_t=55.0,
        ),
    ]
    package = ModelPackage.load(model_dir)
    actual = package.predict(instances)
    expected = [sklearn_prediction(bundle, item) for item in instances]

    assert [value[0] for value in actual] == [value[0] for value in expected]
    np.testing.assert_allclose(
        [value[1] for value in actual],
        [value[1] for value in expected],
        rtol=1e-5,
        atol=1e-4,
    )


def test_invalid_inputs_and_unknown_category(model_dir: Path) -> None:
    with pytest.raises(ValidationError):
        instance(vetro_t=-1)

    package = ModelPackage.load(model_dir)
    with pytest.raises(UnknownCategory, match="zona_costruzione") as exc_info:
        package.predict([instance(zona_costruzione="unknown")])
    assert exc_info.value.index == 0


def test_export_refuses_overwrite_and_missing_package_fails(
    model_dir: Path, tmp_path: Path
) -> None:
    with pytest.raises(FileExistsError):
        export_bundle(SOURCE_MODEL, model_dir)
    with pytest.raises(FileNotFoundError):
        ModelPackage.load(tmp_path)
