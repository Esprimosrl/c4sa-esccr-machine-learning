from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
from urllib.request import Request, urlopen
import importlib.util
import json
import os
import sys
import time
import uuid

import pytest

import onnx_clustering.app as clustering_app
import onnx_inference.app as inference_app
from inference_data import DAILY_ATTRS, FRACTIONS
from onnx_clustering.export_model import export_bundle as export_clustering
from onnx_inference.export_models import export_bundle as export_regression
from orion import publish_entities
from quantumleap import QuantumLeapClient


ROOT = Path(__file__).parent
RUN = os.environ.get("RUN_ORION_INTEGRATION") == "1"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set RUN_ORION_INTEGRATION=1 to use local Orion/QuantumLeap"
)


def load_publisher():
    spec = importlib.util.spec_from_file_location(
        "publish_digital_twins_integration",
        ROOT / "model-training/publish_digital_twins.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def request(method: str, url: str, headers: dict[str, str], body=None):
    payload = None if body is None else json.dumps(body).encode()
    request_headers = dict(headers)
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
    with urlopen(
        Request(url, data=payload, method=method, headers=request_headers), timeout=10
    ) as response:
        data = response.read()
        return response.status, response.headers, json.loads(data) if data else None


def test_orion_publisher_reaches_quantumleap(tmp_path):
    publisher = load_publisher()
    orion_url = os.environ.get("ORION_URL", "http://localhost:1026").rstrip("/")
    ql_url = os.environ.get("QL_URL", "http://localhost:8668").rstrip("/")
    notify_url = os.environ.get(
        "QL_NOTIFY_URL", "http://fiware-quantumleap:8668/v2/notify"
    )
    service = os.environ.get("FIWARE_SERVICE", "openiot")
    service_path = os.environ.get("FIWARE_SERVICE_PATH", "/")
    headers = {
        "Fiware-Service": service,
        "Fiware-ServicePath": service_path,
    }
    suffix = uuid.uuid4().hex
    entity_type = "RifiutiDigitalTwinIntegration"
    entity_id = f"urn:ngsi-v2:{entity_type}:{suffix}:2025"
    subscription_url = None

    try:
        _, response_headers, _ = request(
            "POST",
            f"{orion_url}/v2/subscriptions",
            headers,
            {
                "description": "Temporary digital twin integration test",
                "subject": {
                    "entities": [{"idPattern": ".*", "type": entity_type}]
                },
                "notification": {"http": {"url": notify_url}},
            },
        )
        subscription_url = orion_url + response_headers["Location"]

        stage = tmp_path / "twins.jsonl"
        stage.write_text(
            json.dumps(
                {
                    "id": entity_id,
                    "type": entity_type,
                    "cluster_label": {"type": "Integer", "value": 4},
                    "future_values": {
                        "type": "StructuredValue",
                        "value": [
                            {"date": "2025-01-07", "values_t": {"vetro": 0.25}}
                        ],
                    },
                }
            )
            + "\n"
        )
        assert publisher.publish_twins(
            stage, orion_url, service, service_path, 10, 20
        ) == 1

        _, _, stored = request(
            "GET",
            f"{orion_url}/v2/entities/{quote(entity_id, safe='')}?type={entity_type}",
            headers,
        )
        assert stored["cluster_label"]["value"] == 4
        assert stored["future_values"]["value"][0]["values_t"]["vetro"] == 0.25

        now = datetime.now(timezone.utc)
        ql = QuantumLeapClient(ql_url, entity_type, service, service_path, 10)
        for _ in range(20):
            history = ql.fetch(
                ["cluster_label", "future_values"],
                (now - timedelta(minutes=1)).isoformat(),
                (now + timedelta(minutes=1)).isoformat(),
                entity_ids=[entity_id],
            )
            if history:
                break
            time.sleep(0.25)
        assert history and history[-1]["cluster_label"] == 4
        assert history[-1]["future_values"][0]["values_t"]["vetro"] == 0.25
    finally:
        try:
            request(
                "DELETE",
                f"{orion_url}/v2/entities/{quote(entity_id, safe='')}?type={entity_type}",
                headers,
            )
        except Exception:
            pass
        try:
            request(
                "DELETE",
                f"{ql_url}/v2/entities/{quote(entity_id, safe='')}?type={entity_type}",
                headers,
            )
        except Exception:
            pass
        if subscription_url:
            request("DELETE", subscription_url, headers)


def test_onnx_services_classify_and_forecast_through_quantumleap(tmp_path):
    orion_url = os.environ.get("ORION_URL", "http://localhost:1026").rstrip("/")
    ql_url = os.environ.get("QL_URL", "http://localhost:8668").rstrip("/")
    notify_url = os.environ.get("QL_NOTIFY_URL", "http://fiware-quantumleap:8668/v2/notify")
    service = os.environ.get("FIWARE_SERVICE", "openiot")
    service_path = os.environ.get("FIWARE_SERVICE_PATH", "/")
    headers = {"Fiware-Service": service, "Fiware-ServicePath": service_path}
    suffix = uuid.uuid4().hex
    civic = f"integration-{suffix}"
    raw_type = "RifiutiConferimentoIntegration"
    twin_type = "RifiutiDigitalTwinIntegration"
    forecast_year = datetime.now(timezone.utc).year + 1
    subscriptions = []
    raw_ids = [f"urn:ngsi-v2:{raw_type}:{suffix}:{fraction}" for fraction in FRACTIONS]
    twin_id = f"urn:ngsi-v2:{twin_type}:{civic}:{forecast_year}"
    now = datetime.now(timezone.utc)
    from_date = (now - timedelta(minutes=1)).isoformat()
    to_date = (now + timedelta(minutes=1)).isoformat()

    try:
        for entity_type in (raw_type, twin_type):
            _, response_headers, _ = request(
                "POST",
                f"{orion_url}/v2/subscriptions",
                headers,
                {
                    "description": "Temporary ONNX system integration test",
                    "subject": {"entities": [{"idPattern": ".*", "type": entity_type}]},
                    "notification": {"http": {"url": notify_url}},
                },
            )
            subscriptions.append(orion_url + response_headers["Location"])

        raw_entities = []
        for entity_id, fraction in zip(raw_ids, FRACTIONS, strict=True):
            raw_entities.append({
                "id": entity_id,
                "type": raw_type,
                "id_civico": {"type": "Text", "value": civic},
                "frazione": {"type": "Text", "value": fraction},
                "anno_produzione": {"type": "Integer", "value": now.year},
                "coord_x": {"type": "Number", "value": 16.3},
                "coord_y": {"type": "Number", "value": 41.2},
                "densita_locale": {"type": "Number", "value": 100.0},
                "zona_raccolta": {"type": "Text", "value": "1"},
                "zona_costruzione": {"type": "Text", "value": "B3R"},
                "giorni_accumulati": {"type": "Integer", "value": 1},
                "conferimento_t": {"type": "Number", "value": 1.0},
            })
        assert publish_entities(raw_entities, orion_url, service, service_path, 10) == len(FRACTIONS)

        raw_ql = QuantumLeapClient(ql_url, raw_type, service, service_path, 10)
        for _ in range(20):
            if len(raw_ql.fetch(list(DAILY_ATTRS), from_date, to_date)) == len(FRACTIONS):
                break
            time.sleep(0.25)
        else:
            pytest.fail("Raw entities did not reach QuantumLeap.")

        clustering_dir = tmp_path / "clustering"
        regression_dir = tmp_path / "regression"
        export_clustering(ROOT / "models/kmeans_model_annuo_peso_coord.joblib", clustering_dir)
        export_regression(ROOT / "models/grouped_sgd_time_series_72modelli.joblib", regression_dir)
        clustering_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            model_package=clustering_app.ModelPackage.load(clustering_dir),
            quantumleap=raw_ql,
            orion={"url": orion_url, "service": service, "service_path": service_path, "timeout": 10, "entity_type": twin_type},
        )))
        clustered = clustering_app.predict_quantumleap(
            clustering_app.QuantumLeapPredictionRequest(
                from_date=from_date,
                to_date=to_date,
                forecast_year=forecast_year,
                civic_ids=[civic],
            ),
            clustering_request,
        )
        assert clustered["predictions"][0]["entity_id"] == twin_id

        regression_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            model_package=inference_app.ModelPackage.load(regression_dir),
            quantumleap=raw_ql,
            twin_quantumleap=QuantumLeapClient(ql_url, twin_type, service, service_path, 10),
            forecast={
                "calendar": ROOT / "data_csv_parquet/calendario_raccolta.csv",
                "label_wait": 5,
                "orion_url": orion_url,
                "orion_timeout": 10,
                "service": service,
                "service_path": service_path,
                "entity_type": twin_type,
            },
        )))
        forecast = inference_app.forecast_quantumleap(
            inference_app.QuantumLeapForecastRequest(
                from_date=from_date,
                to_date=to_date,
                forecast_from_date=f"{forecast_year}-01-01",
                forecast_to_date=f"{forecast_year}-01-07",
                civic_ids=[civic],
            ),
            regression_request,
        )
        assert forecast["predictions"][0]["future_values"]
        _, _, stored = request("GET", f"{orion_url}/v2/entities/{quote(twin_id, safe='')}?type={twin_type}", headers)
        assert stored["future_values"]["value"]
    finally:
        for entity_id, entity_type in [(item, raw_type) for item in raw_ids] + [(twin_id, twin_type)]:
            for base in (orion_url, ql_url):
                try:
                    request("DELETE", f"{base}/v2/entities/{quote(entity_id, safe='')}?type={entity_type}", headers)
                except Exception:
                    pass
        for subscription in subscriptions:
            request("DELETE", subscription, headers)
