#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import argparse
import json
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion import publish_entities
from quantumleap import QuantumLeapClient


def request(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict | None = None,
) -> tuple[int, object]:
    data = json.dumps(payload).encode() if payload is not None else None
    request_headers = {
        **headers,
        **({"Content-Type": "application/json"} if data is not None else {}),
    }
    try:
        with urlopen(
            Request(url, data=data, method=method, headers=request_headers),
            timeout=30,
        ) as response:
            return response.status, response.headers
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc


def entities(
    entity_type: str,
    count: int,
    run_label: str,
) -> list[dict[str, object]]:
    observed = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return [
        {
            "id": f"urn:ngsi-v2:{entity_type}:{index}",
            "type": entity_type,
            "benchmark_run": {"type": "Text", "value": run_label},
            "benchmark_value": {"type": "Integer", "value": index},
            "dateObserved": {"type": "DateTime", "value": observed},
        }
        for index in range(count)
    ]


def wait_for_quantumleap(
    client: QuantumLeapClient,
    run_label: str,
    expected: int,
    from_date: str,
    to_date: str,
    timeout_seconds: float,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    last_progress = time.monotonic()
    previous_visible = -1
    while True:
        rows = client.fetch(["benchmark_run"], from_date, to_date)
        visible = sum(row["benchmark_run"] == run_label for row in rows)
        if visible >= expected:
            return visible
        now = time.monotonic()
        if visible > previous_visible:
            previous_visible = visible
            last_progress = now
        if now >= deadline or now - last_progress >= 10:
            return visible
        time.sleep(0.5)


def delete_orion_entities(
    orion_url: str,
    headers: dict[str, str],
    entity_type: str,
    count: int,
) -> None:
    pending = (
        {"id": f"urn:ngsi-v2:{entity_type}:{index}", "type": entity_type}
        for index in range(count)
    )
    while batch := list(islice(pending, 500)):
        request(
            "POST",
            f"{orion_url}/v2/op/update",
            headers,
            {"actionType": "delete", "entities": batch},
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Orion batch publication and QuantumLeap propagation."
    )
    parser.add_argument("--orion-url", default="http://localhost:1026")
    parser.add_argument("--ql-url", default="http://localhost:8668")
    parser.add_argument(
        "--notify-url", default="http://fiware-quantumleap:8668/v2/notify"
    )
    parser.add_argument("--fiware-service", default="openiot")
    parser.add_argument("--fiware-service-path", default="/")
    parser.add_argument("--entities", type=int, default=1_000)
    parser.add_argument("--batch-sizes", default="20,50,100,250")
    parser.add_argument("--propagation-timeout-seconds", type=float, default=180)
    args = parser.parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if args.entities <= 0 or any(size <= 0 for size in batch_sizes):
        parser.error("--entities and every --batch-sizes value must be positive")

    orion_url = args.orion_url.rstrip("/")
    ql_url = args.ql_url.rstrip("/")
    headers = {
        "Fiware-Service": args.fiware_service,
        "Fiware-ServicePath": args.fiware_service_path,
    }
    entity_type = f"RifiutiDigitalTwinBenchmark{uuid.uuid4().hex[:10]}"
    subscription_url = None
    started = datetime.now(timezone.utc) - timedelta(minutes=1)
    from_date = started.isoformat()
    to_date = (started + timedelta(minutes=10)).isoformat()

    try:
        status, response_headers = request(
            "POST",
            f"{orion_url}/v2/subscriptions",
            headers,
            {
                "description": "Temporary Orion QuantumLeap publication benchmark",
                "subject": {
                    "entities": [{"idPattern": ".*", "type": entity_type}]
                },
                "notification": {"http": {"url": args.notify_url}},
            },
        )
        if status != 201:
            raise RuntimeError(f"Subscription creation returned HTTP {status}.")
        subscription_url = orion_url + response_headers["Location"]
        ql = QuantumLeapClient(
            ql_url,
            entity_type,
            args.fiware_service,
            args.fiware_service_path,
            30,
        )

        initialize = f"initialize-{uuid.uuid4().hex}"
        publish_entities(
            entities(entity_type, 1, initialize),
            orion_url,
            args.fiware_service,
            args.fiware_service_path,
            30,
            batch_size=1,
            concurrency=1,
        )
        initialized = wait_for_quantumleap(
            ql,
            initialize,
            1,
            from_date,
            to_date,
            args.propagation_timeout_seconds,
        )
        if initialized != 1:
            raise RuntimeError("QuantumLeap entity type initialization failed.")

        warmup = f"warmup-{uuid.uuid4().hex}"
        warmup_entities = entities(entity_type, args.entities, warmup)
        for offset in range(0, args.entities, 300):
            chunk = warmup_entities[offset : offset + 300]
            publish_entities(
                chunk,
                orion_url,
                args.fiware_service,
                args.fiware_service_path,
                30,
                batch_size=20,
                concurrency=1,
            )
            expected = offset + len(chunk)
            warmed = wait_for_quantumleap(
                ql,
                warmup,
                expected,
                from_date,
                to_date,
                args.propagation_timeout_seconds,
            )
            if warmed != expected:
                raise RuntimeError(
                    f"Only {warmed} of {expected} warm-up rows reached QuantumLeap."
                )

        cases = [(20, 1), *[(size, 2) for size in batch_sizes]]
        print(
            "batch_size concurrency published visible "
            "orion_seconds visible_seconds entities/s",
            flush=True,
        )
        for batch_size, concurrency in cases:
            run_label = f"run-{batch_size}-{concurrency}-{uuid.uuid4().hex}"
            batch = entities(entity_type, args.entities, run_label)
            start = time.perf_counter()
            published = publish_entities(
                iter(batch),
                orion_url,
                args.fiware_service,
                args.fiware_service_path,
                30,
                batch_size=batch_size,
                concurrency=concurrency,
            )
            orion_elapsed = time.perf_counter() - start
            visible = wait_for_quantumleap(
                ql,
                run_label,
                args.entities,
                from_date,
                to_date,
                args.propagation_timeout_seconds,
            )
            visible_elapsed = time.perf_counter() - start
            print(
                f"{batch_size} {concurrency} {published} {visible} "
                f"{orion_elapsed:.3f} {visible_elapsed:.3f} "
                f"{published / visible_elapsed:.1f}",
                flush=True,
            )
    finally:
        if subscription_url:
            try:
                request("DELETE", subscription_url, headers)
            except Exception as exc:
                print(f"warning: subscription cleanup failed: {exc}", file=sys.stderr)
        try:
            delete_orion_entities(orion_url, headers, entity_type, args.entities)
        except Exception as exc:
            print(f"warning: Orion cleanup failed: {exc}", file=sys.stderr)
        try:
            request(
                "DELETE",
                f"{ql_url}/v2/types/{quote(entity_type, safe='')}?dropTable=true",
                headers,
            )
        except Exception as exc:
            print(f"warning: QuantumLeap cleanup failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
