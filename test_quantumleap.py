from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
import json
import socket
import subprocess
import sys

import duckdb
import joblib
import pandas as pd
import pytest
from fastapi import HTTPException

import quantumleap
import onnx_clustering.app as clustering_app
import onnx_inference.app as inference_app
from quantumleap import (
    QuantumLeapClient,
    QuantumLeapDataError,
    QuantumLeapError,
    QuantumLeapTimeout,
)
from onnx_clustering.app import (
    ModelPackage as ClusteringModelPackage,
    PredictionInput as ClusteringInput,
)
from onnx_clustering.export_model import export_bundle as export_clustering_bundle
from onnx_inference.app import (
    ModelPackage as RegressionModelPackage,
    PredictionInput as RegressionInput,
    QuantumLeapPredictionRequest,
    UnknownGroup,
    predict_quantumleap,
)
from onnx_inference.export_models import export_bundle as export_regression_bundle


ROOT = Path(__file__).parent


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_quantumleap", "model-training/prepare_quantumleap.py")
pipeline = load_module("ml_pipeline_graph", "model-training/ml_pipeline_graph.py")
publisher = load_module(
    "publish_digital_twins", "model-training/publish_digital_twins.py"
)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


def entity(entity_id="a/b", indexes=None, **attrs):
    indexes = indexes or ["2024-01-01T00:00:00Z"]
    return {
        "id": entity_id,
        "index": indexes,
        "attributes": [
            {"attrName": name, "values": values}
            for name, values in attrs.items()
        ],
    }


def client():
    return QuantumLeapClient("http://ql/base/", "Waste Type/1", "tenant", "/andria", 2)


def test_client_encodes_query_headers_and_paginates(monkeypatch):
    monkeypatch.setattr(quantumleap, "PAGE_SIZE", 2)
    requests = []
    pages = [
        {"entities": [entity(indexes=["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"], x=[1, 2], y=[3, 4])]},
        {"entities": [entity(indexes=["2024-01-03T00:00:00Z"], x=[5], y=[6])]},
    ]

    def fake_open(request, timeout):
        requests.append((request, timeout))
        return Response(pages.pop(0))

    monkeypatch.setattr(quantumleap, "urlopen", fake_open)
    rows = client().fetch(
        ["x", "y"],
        "2024-01-01T00:00:00Z",
        "2024-01-31T00:00:00Z",
        entity_ids=["a/b", "urn:x"],
    )
    assert [row["x"] for row in rows] == [1, 2, 5]
    assert len(requests) == 2
    parsed = urlparse(requests[0][0].full_url)
    assert parsed.path.endswith("/v2/types/Waste%20Type%2F1")
    assert parse_qs(parsed.query) == {
        "attrs": ["x,y"],
        "fromDate": ["2024-01-01T00:00:00Z"],
        "toDate": ["2024-01-31T00:00:00Z"],
        "id": ["a/b,urn:x"],
        "limit": ["2"],
        "offset": ["0"],
    }
    assert parse_qs(urlparse(requests[1][0].full_url).query)["offset"] == ["2"]
    headers = {key.lower(): value for key, value in requests[0][0].headers.items()}
    assert headers["fiware-service"] == "tenant"
    assert headers["fiware-servicepath"] == "/andria"


def test_client_accepts_quantumleap_entity_id_schema_key(monkeypatch):
    payload = {"entities": [{**entity(x=[1]), "entityId": "live-id"}]}
    payload["entities"][0].pop("id")
    monkeypatch.setattr(
        quantumleap, "urlopen", lambda *args, **kwargs: Response(payload)
    )
    rows = client().fetch(
        ["x"], "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"
    )
    assert rows[0]["entity_id"] == "live-id"


def test_client_fetch_latest_uses_last_n(monkeypatch):
    requests = []
    monkeypatch.setattr(
        quantumleap,
        "urlopen",
        lambda request, timeout: requests.append(request) or Response({"entities": []}),
    )
    assert client().fetch_latest(["cluster_label"], entity_ids=["urn:twin:1"]) == []
    assert parse_qs(urlparse(requests[0].full_url).query) == {
        "attrs": ["cluster_label"],
        "id": ["urn:twin:1"],
        "lastN": ["1"],
        "limit": ["10000"],
        "offset": ["0"],
    }


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "entities array"),
        ({"entities": [entity(x=[1])]}, "missing attributes"),
        ({"entities": [entity(indexes=["2024-01-01T00:00:00Z"], x=[1, 2], y=[3])]}, "length mismatch"),
        ({"entities": [entity(x=[None], y=[3])]}, "Null required"),
        ({"entities": [entity(indexes=["2024-01-02T00:00:00Z", "2024-01-01T00:00:00Z"], x=[1, 2], y=[3, 4])]}, "out of order"),
    ],
)
def test_client_rejects_malformed_parallel_history(monkeypatch, payload, message):
    monkeypatch.setattr(quantumleap, "urlopen", lambda *args, **kwargs: Response(payload))
    with pytest.raises(QuantumLeapDataError, match=message):
        client().fetch(["x", "y"], "2024-01-01T00:00:00Z", "2024-01-31T00:00:00Z")


def test_client_maps_no_records_and_upstream_failures(monkeypatch):
    def not_found(*args, **kwargs):
        raise HTTPError("url", 404, "missing", {}, None)

    monkeypatch.setattr(quantumleap, "urlopen", not_found)
    assert client().fetch(["x"], "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z") == []
    monkeypatch.setattr(quantumleap, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(socket.timeout()))
    with pytest.raises(QuantumLeapTimeout):
        client().fetch(["x"], "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")


def daily_rows():
    base = {
        "timestamp": "2024-01-01T00:00:00Z",
        "id_civico": "10",
        "anno_produzione": 2024,
        "coord_x": 16.3,
        "coord_y": 41.2,
        "densita_locale": 100,
        "zona_raccolta": 1,
        "zona_costruzione": "B3R",
        "giorni_accumulati": 2,
        "cluster_label": 99,
    }
    return pd.DataFrame([
        {**base, "entity_id": "10:vetro", "frazione": "vetro", "conferimento_t": 1.5},
        {**base, "entity_id": "10:vetro", "frazione": "vetro", "conferimento_t": 2.0},
        {**base, "entity_id": "10:carta", "frazione": "carta_cartone", "conferimento_t": 4.0},
        {**base, "entity_id": "10:vetro:2025", "anno_produzione": 2025, "frazione": "vetro", "conferimento_t": 8.0},
    ])


def test_annual_pivot_zero_fill_multi_year_and_join():
    annual = prepare.annual_rows(daily_rows())
    assert len(annual) == 2
    row = annual[annual["anno_produzione"] == 2024].iloc[0]
    assert row["vetro_t"] == 3.5
    assert row["carta_cartone_t"] == 4
    assert row["legno_t"] == 0
    assert row["totale_t"] == 7.5

    assignments = annual[["id_civico", "anno_produzione"]].copy()
    assignments["cluster_label"] = [3, 4]
    joined = prepare.join_clusters(daily_rows(), assignments)
    assert list(joined["cluster_label"]) == [3, 3, 3, 4]

    joined_without_old_labels = prepare.join_clusters(
        daily_rows().drop(columns=["cluster_label"]), assignments
    )
    assert list(joined_without_old_labels["cluster_label"]) == [3, 3, 3, 4]

    numeric_daily_ids = daily_rows()
    numeric_daily_ids["id_civico"] = numeric_daily_ids["id_civico"].astype(int)
    joined_numeric_ids = prepare.join_clusters(numeric_daily_ids, assignments)
    assert list(joined_numeric_ids["cluster_label"]) == [3, 3, 3, 4]
    assert joined_numeric_ids["id_civico"].dtype.name == "string"


def test_annual_rejects_unknown_fraction_and_static_conflict():
    rows = daily_rows()
    rows.loc[0, "frazione"] = "mystery"
    with pytest.raises(ValueError, match="Unknown fractions"):
        prepare.annual_rows(rows)
    rows = daily_rows()
    rows.loc[0, "coord_x"] = 99
    with pytest.raises(ValueError, match="Conflicting coord_x"):
        prepare.annual_rows(rows)
    rows = daily_rows()
    rows.loc[2, "entity_id"] = "10:vetro"
    with pytest.raises(ValueError, match="Conflicting fraction"):
        prepare.annual_rows(rows)


def test_pipeline_modes_preserve_file_graph_and_order_quantumleap(tmp_path):
    assert list(pipeline.build_graph("python", "42", "2024")) == [
        "kmeans", "daily_distribution", "grouped_sgd"
    ]
    output_dir = tmp_path / "isolated"
    file_graph = pipeline.build_graph("python", "42", "2024", output_dir)
    assert file_graph["kmeans"].outputs == (
        output_dir / "data_csv_parquet/kmeans_assignments_annuo_peso_coord.csv",
        output_dir / "models/kmeans_model_annuo_peso_coord.joblib",
    )
    assert file_graph["grouped_sgd"].outputs == (
        output_dir / "models/grouped_sgd_time_series.joblib",
        output_dir / "grouped_sgd_time_series_metrics.csv",
        output_dir / "predictions/sgd_test_predictions_grouped.parquet",
    )
    args = SimpleNamespace(
        from_date="2024-01-01T00:00:00Z",
        to_date="2024-12-31T23:59:59Z",
        ql_url="http://ql",
        ql_entity_type="Waste",
        fiware_service="tenant",
        fiware_service_path="/",
        ql_timeout_seconds=30,
        ql_id_pattern=None,
    )
    graph = pipeline.build_quantumleap_graph("python", "42", tmp_path, args)
    assert [node.name for node in pipeline.topo_sort(graph)] == [
        "quantumleap_fetch", "annual_aggregation", "kmeans", "cluster_join", "grouped_sgd"
    ]

    args.publish_orion = True
    args.forecast_calendar = "data_csv_parquet/calendario_raccolta.csv"
    args.forecast_from_date = "2025-01-01"
    args.forecast_to_date = "2025-01-31"
    args.orion_url = "http://orion:1026"
    args.orion_entity_type = "RifiutiDigitalTwin"
    args.orion_timeout_seconds = 10
    args.orion_batch_size = 20
    args.output_dir = output_dir
    graph = pipeline.build_quantumleap_graph("python", "42", tmp_path, args)
    assert [node.name for node in pipeline.topo_sort(graph)] == [
        "quantumleap_fetch",
        "annual_aggregation",
        "kmeans",
        "cluster_join",
        "grouped_sgd",
        "publish_digital_twins",
    ]
    publish = graph["publish_digital_twins"]
    assert publish.needs == ("grouped_sgd",)
    assert "--forecast-from-date" in publish.command
    assert "http://orion:1026" in publish.command
    assert (
        str(output_dir / "models/kmeans_model_annuo_peso_coord.joblib")
        in publish.command
    )
    assert str(output_dir / "models/grouped_sgd_time_series.joblib") in publish.command


def test_pipeline_dry_run_allows_uncreated_intermediate_outputs(tmp_path, capsys):
    middle = tmp_path / "middle"
    final = tmp_path / "final"
    final.touch()
    graph = {
        "middle": pipeline.Node("middle", ["make-middle"], (), (middle,)),
        "final": pipeline.Node(
            "final", ["make-final"], (middle,), (final,), ("middle",)
        ),
    }

    pipeline.run_graph(graph, force=False, dry_run=True)

    assert capsys.readouterr().out.splitlines() == [
        "run middle: make-middle",
        "run final: make-final",
    ]


def test_pipeline_normalizes_date_only_training_ranges():
    assert pipeline.normalize_training_range("2024-01-01", "2024-12-31") == (
        "2024-01-01T00:00:00Z",
        "2024-12-31T23:59:59Z",
    )
    with pytest.raises(ValueError, match="timezone"):
        pipeline.normalize_training_range(
            "2024-01-01T12:00:00", "2024-12-31T12:00:00"
        )
    with pytest.raises(ValueError, match="must not be after"):
        pipeline.normalize_training_range("2025-01-01", "2024-12-31")


def test_digital_twin_uses_latest_available_classification():
    assignments = pd.DataFrame(
        [
            {
                "id_civico": "10",
                "anno_produzione": 2023,
                "coord_x": 16.1,
                "coord_y": 41.1,
                "densita_locale": 100,
                "zona_raccolta": 1,
                "cluster_label": 2,
                "cluster_distance": 0.3,
            },
            {
                "id_civico": "10",
                "anno_produzione": 2024,
                "coord_x": 16.2,
                "coord_y": 41.2,
                "densita_locale": 110,
                "zona_raccolta": 2,
                "cluster_label": 4,
                "cluster_distance": 0.1,
            },
        ]
    )
    selected = publisher.latest_assignments(assignments, 2025)
    assert selected.loc[0, "classification_year"] == 2024
    assert selected.loc[0, "cluster_label"] == 4

    entity = publisher.build_entity(
        selected.iloc[0],
        2025,
        publisher.parse_day("2025-01-01"),
        publisher.parse_day("2025-01-31"),
        {publisher.parse_day("2025-01-07"): {"vetro": 0.25}},
        "RifiutiDigitalTwin",
        "classifier-sha",
        "regressor-sha",
        "2025-01-01T00:00:00Z",
    )
    assert entity["id"] == "urn:ngsi-v2:RifiutiDigitalTwin:10:2025"
    assert entity["cluster_label"] == {"type": "Integer", "value": 4}
    assert entity["future_values"]["value"] == [
        {"date": "2025-01-07", "values_t": {"vetro": 0.25}}
    ]


def test_orion_publication_batches_and_sends_tenant_headers(tmp_path, monkeypatch):
    stage = tmp_path / "twins.jsonl"
    entities = [{"id": f"urn:test:{i}", "type": "Twin"} for i in range(3)]
    stage.write_text("".join(json.dumps(item) + "\n" for item in entities))
    requests = []

    class OrionResponse:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def fake_open(request, timeout):
        requests.append((request, timeout))
        return OrionResponse()

    monkeypatch.setattr(publisher, "urlopen", fake_open)
    count = publisher.publish_twins(
        stage, "http://orion:1026/", "tenant", "/andria", 7, 2
    )
    assert count == 3
    assert [len(json.loads(item.data)["entities"]) for item, _ in requests] == [2, 1]
    assert json.loads(requests[0][0].data)["actionType"] == "append"
    headers = {key.lower(): value for key, value in requests[0][0].headers.items()}
    assert headers["fiware-service"] == "tenant"
    assert headers["fiware-servicepath"] == "/andria"
    assert requests[0][1] == 7


def test_existing_models_stage_digital_twin_predictions(tmp_path):
    assignments = pd.DataFrame(
        [
            {
                "id_civico": "10",
                "anno_produzione": 2024,
                "coord_x": 16.2896597,
                "coord_y": 41.2407249,
                "densita_locale": 384.0,
                "zona_raccolta": "1",
                "cluster_label": 0,
                "cluster_distance": 0.2,
            }
        ]
    )
    assignments_path = tmp_path / "assignments.parquet"
    prepare.write_parquet(assignments, assignments_path)
    stage_path = tmp_path / "twins.jsonl"
    count = publisher.stage_twins(
        assignments_path,
        ROOT / "models/kmeans_model_annuo_peso_coord.joblib",
        ROOT / "models/grouped_sgd_time_series_72modelli.joblib",
        ROOT / "data_csv_parquet/calendario_raccolta.csv",
        publisher.parse_day("2025-01-01"),
        publisher.parse_day("2025-01-07"),
        "RifiutiDigitalTwin",
        stage_path,
    )
    entity = json.loads(stage_path.read_text())
    assert count == 1
    assert entity["cluster_label"]["value"] == 0
    assert entity["future_values"]["value"]
    assert all(
        prediction >= 0
        for event in entity["future_values"]["value"]
        for prediction in event["values_t"].values()
    )


def test_tiny_training_pipeline_classifies_then_trains_regressor(tmp_path):
    rows = []
    for day in range(1, 11):
        timestamp = f"2024-01-{day:02d}T00:00:00Z"
        for civic in range(12):
            rows.append(
                {
                    "entity_id": f"{civic}:vetro",
                    "timestamp": timestamp,
                    "data": timestamp,
                    "id_civico": str(civic),
                    "frazione": "vetro",
                    "anno_produzione": 2024,
                    "coord_x": 16.25 + civic * 0.01,
                    "coord_y": 41.20 + civic * 0.005,
                    "densita_locale": 100 + civic * 20,
                    "zona_raccolta": civic % 4 + 1,
                    "zona_costruzione": "B3R",
                    "giorni_accumulati": day % 3 + 1,
                    "conferimento_t": (civic + 1) * (day + 1) / 100,
                }
            )
    daily = pd.DataFrame(rows)
    annual = prepare.annual_rows(daily)

    annual_path = tmp_path / "annual.parquet"
    assignments_path = tmp_path / "assignments.parquet"
    kmeans_model = tmp_path / "kmeans.joblib"
    prepare.write_parquet(annual, annual_path)
    subprocess.run(
        [
            sys.executable,
            "model-training/classification/train_kmeans.py",
            "--input",
            str(annual_path),
            "--n-clusters",
            "2",
            "--output",
            str(assignments_path),
            "--model-output",
            str(kmeans_model),
            "--coord-weight",
            "8",
            "--annual-kg-weight",
            "0.5",
        ],
        cwd=ROOT,
        check=True,
    )

    assignments = prepare.read_parquet(assignments_path)
    classified = prepare.join_clusters(daily, assignments)
    assert len(classified) == len(daily)
    assert classified["cluster_label"].notna().all()
    assert classified["cluster_label"].nunique() == 2

    classified_path = tmp_path / "classified.parquet"
    regressor_model = tmp_path / "regressor.joblib"
    metrics_path = tmp_path / "metrics.csv"
    predictions_path = tmp_path / "predictions.parquet"
    prepare.write_parquet(classified, classified_path)
    subprocess.run(
        [
            sys.executable,
            "model-training/regression/train_grouped_sgd_time_series_regressor.py",
            "--input",
            str(classified_path),
            "--split-ratios",
            "60,20,20",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--model-output",
            str(regressor_model),
            "--metrics-output",
            str(metrics_path),
            "--predictions-output",
            str(predictions_path),
        ],
        cwd=ROOT,
        check=True,
    )

    bundle = joblib.load(regressor_model)
    assert len(bundle["models"]) == 2
    assert metrics_path.exists()
    prediction_count = duckdb.connect().execute(
        "SELECT count(*) FROM read_parquet(?)", [str(predictions_path)]
    ).fetchone()[0]
    assert prediction_count > 0


def test_existing_models_classify_then_regress(tmp_path):
    clustering_dir = tmp_path / "clustering"
    regression_dir = tmp_path / "regression"
    export_clustering_bundle(
        ROOT / "models/kmeans_model_annuo_peso_coord.joblib",
        clustering_dir,
    )
    export_regression_bundle(
        ROOT / "models/grouped_sgd_time_series_72modelli.joblib",
        regression_dir,
    )
    clustering = ClusteringModelPackage.load(clustering_dir)
    regression = RegressionModelPackage.load(regression_dir)

    annual = ClusteringInput(
        anno_produzione=2024,
        coord_x=16.2896597,
        coord_y=41.2407249,
        densita_locale=384,
        altro_rd_t=4.47,
        carta_cartone_t=76.623,
        frazione_organica_t=266.098,
        ingombranti_misti_t=139.054,
        legno_t=0,
        metallo_t=2.527,
        plastica_t=22.206,
        pulizia_stradale_recupero_t=0,
        raee_t=0,
        rifiuti_cd_t=0,
        selettiva_t=0,
        tessili_t=0,
        vetro_t=28.849,
        zona_raccolta=1,
        zona_costruzione="C3PdF",
    )
    cluster_label, cluster_distance = clustering.predict([annual])[0]
    prediction = regression.predict(
        [
            RegressionInput(
                data="2024-09-10",
                anno_produzione=2024,
                coord_x=annual.coord_x,
                coord_y=annual.coord_y,
                densita_locale=annual.densita_locale,
                giorni_accumulati=3,
                zona_raccolta=annual.zona_raccolta,
                frazione="altro_rd",
                cluster_label=cluster_label,
            )
        ]
    )[0]

    assert 0 <= cluster_label < clustering.n_clusters
    assert cluster_distance >= 0
    assert prediction >= 0


class FakePackage:
    model_version = "v1"

    def predict(self, instances):
        if instances[0].cluster_label == 999:
            raise UnknownGroup(0, instances[0].frazione, 999)
        return [float(item.coord_x) for item in instances]


class FakeQL:
    def __init__(self, result):
        self.result = result
        self.call = None

    def fetch(self, *args, **kwargs):
        self.call = (args, kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def api_call(result, entity_ids=None):
    ql = FakeQL(result)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        model_package=FakePackage(), quantumleap=ql
    )))
    payload = QuantumLeapPredictionRequest(
        from_date="2024-01-01T00:00:00Z",
        to_date="2024-01-31T23:59:59Z",
        entity_ids=entity_ids,
    )
    return predict_quantumleap(payload, request), ql


def feature_row(entity_id, timestamp, cluster=1):
    return {
        "entity_id": entity_id,
        "timestamp": timestamp,
        "id_civico": "10",
        "frazione": "vetro",
        "anno_produzione": 2024,
        "coord_x": 16.3,
        "coord_y": 41.2,
        "densita_locale": 100,
        "zona_raccolta": 1,
        "giorni_accumulati": 2,
        "cluster_label": cluster,
    }


def complete_daily_row(**updates):
    row = {
        "entity_id": "10:vetro",
        "timestamp": "2024-01-01T00:00:00Z",
        "id_civico": "10",
        "frazione": "vetro",
        "anno_produzione": 2024,
        "coord_x": 16.3,
        "coord_y": 41.2,
        "densita_locale": 100,
        "zona_raccolta": 1,
        "zona_costruzione": "B3R",
        "giorni_accumulati": 2,
        "conferimento_t": 1.5,
    }
    row.update(updates)
    return row


def test_onnx_clustering_fetches_aggregates_and_publishes_labels(monkeypatch):
    class Package:
        model_version = "classifier-v1"

        def predict(self, instances):
            assert instances[0].vetro_t == 1.5
            assert instances[0].carta_cartone_t == 0
            return [(3, 0.25)]

    published = []
    monkeypatch.setattr(
        clustering_app,
        "publish_entities",
        lambda entities, *args: published.extend(entities) or len(entities),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        model_package=Package(),
        quantumleap=FakeQL([complete_daily_row()]),
        orion={
            "url": "http://orion",
            "service": "tenant",
            "service_path": "/",
            "timeout": 5,
            "entity_type": "RifiutiDigitalTwin",
        },
    )))
    result = clustering_app.predict_quantumleap(
        clustering_app.QuantumLeapPredictionRequest(
            from_date="2024-01-01T00:00:00Z",
            to_date="2024-12-31T23:59:59Z",
            forecast_year=2025,
            civic_ids=["10"],
        ),
        request,
    )
    assert result["predictions"][0]["cluster_label"] == 3
    assert published[0]["id"] == "urn:ngsi-v2:RifiutiDigitalTwin:10:2025"
    assert published[0]["classification_year"]["value"] == 2024
    assert "forecast_from" not in published[0]
    assert "forecast_to" not in published[0]


def test_onnx_regression_reads_twin_label_forecasts_and_publishes(
    tmp_path, monkeypatch
):
    calendar = tmp_path / "calendar.csv"
    calendar.write_text("zona_raccolta,frazione,weekday\n1,vetro,martedi\n")

    class Package:
        model_version = "regressor-v1"
        sessions = {("vetro", 3): object()}

        def predict(self, instances):
            assert all(item.cluster_label == 3 for item in instances)
            return [0.5] * len(instances)

    class TwinQL:
        def fetch_latest(self, attrs, entity_ids=None, id_pattern=None):
            return [{
                "entity_id": "urn:ngsi-v2:RifiutiDigitalTwin:10:2025",
                "timestamp": "2025-01-01T00:00:00Z",
                "id_civico": "10",
                "forecast_year": 2025,
                "classification_year": 2024,
                "cluster_label": 3,
                "cluster_distance": 0.25,
                "classifier_model_version": "classifier-v1",
            }]

    published = []
    monkeypatch.setattr(
        inference_app,
        "publish_entities",
        lambda entities, *args: published.extend(entities) or len(entities),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        model_package=Package(),
        quantumleap=FakeQL([complete_daily_row()]),
        twin_quantumleap=TwinQL(),
        forecast={
            "calendar": calendar,
            "label_wait": 0,
            "orion_url": "http://orion",
            "orion_timeout": 5,
            "service": "tenant",
            "service_path": "/",
            "entity_type": "RifiutiDigitalTwin",
        },
    )))
    result = inference_app.forecast_quantumleap(
        inference_app.QuantumLeapForecastRequest(
            from_date="2024-01-01T00:00:00Z",
            to_date="2024-12-31T23:59:59Z",
            forecast_from_date="2025-01-01",
            forecast_to_date="2025-01-31",
            civic_ids=["10"],
        ),
        request,
    )
    assert result["predictions"][0]["future_values"]
    assert published[0]["cluster_label"]["value"] == 3
    assert published[0]["regressor_model_version"]["value"] == "regressor-v1"


def test_ranged_inference_filters_orders_and_returns_identity():
    result, ql = api_call([
        feature_row("b", "2024-01-02T00:00:00Z"),
        feature_row("a", "2024-01-01T00:00:00Z"),
    ], ["a", "b"])
    assert result["model_version"] == "v1"
    assert [item["entity_id"] for item in result["predictions"]] == ["a", "b"]
    assert result["predictions"][0]["timestamp"] == "2024-01-01T00:00:00Z"
    assert ql.call[1]["entity_ids"] == ["a", "b"]


@pytest.mark.parametrize(
    "result,status",
    [
        ([], 404),
        (ValueError("bad dates"), 422),
        (QuantumLeapTimeout("slow"), 504),
        (QuantumLeapError("bad gateway"), 502),
        ([feature_row("a", "2024-01-01T00:00:00Z", 999)], 422),
        ([{**feature_row("a", "2024-01-01T00:00:00Z"), "coord_x": "bad"}], 502),
    ],
)
def test_ranged_inference_error_mapping(result, status):
    with pytest.raises(HTTPException) as exc:
        api_call(result)
    assert exc.value.status_code == status
