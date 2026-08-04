#!/usr/bin/env python3

import csv
import os
import sys
from typing import Any

import requests


QL_URL = os.getenv("QL_URL", "http://localhost:8668")
ENTITY_TYPE = os.getenv("ENTITY_TYPE", "Room")

FIWARE_SERVICE = os.getenv("FIWARE_SERVICE", "openiot")
FIWARE_SERVICE_PATH = os.getenv("FIWARE_SERVICE_PATH", "/")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))

# CSV column -> NGSI attribute type
ATTRIBUTE_TYPES = {
    "temperature": "Number",
    "humidity": "Number",
}


def convert_value(value: str, ngsi_type: str) -> Any:
    """Convert a CSV string into the appropriate Python/JSON value."""
    value = value.strip()

    if value == "":
        return None

    if ngsi_type == "Number":
        return float(value)

    if ngsi_type == "Integer":
        return int(value)

    if ngsi_type == "Boolean":
        normalized = value.lower()

        if normalized in {"true", "1", "yes"}:
            return True

        if normalized in {"false", "0", "no"}:
            return False

        raise ValueError(f"Invalid Boolean value: {value}")

    return value


def make_entity(row: dict[str, str]) -> dict[str, Any]:
    entity: dict[str, Any] = {
        "id": row["entity_id"],
        "type": ENTITY_TYPE,
        "timestamp": {
            "type": "DateTime",
            "value": row["timestamp"],
        },
    }

    for column, ngsi_type in ATTRIBUTE_TYPES.items():
        raw_value = row.get(column, "")
        value = convert_value(raw_value, ngsi_type)

        if value is not None:
            entity[column] = {
                "type": ngsi_type,
                "value": value,
            }

    return entity


def send_batch(entities: list[dict[str, Any]]) -> None:
    payload = {
        "subscriptionId": "csv-import",
        "data": entities,
    }

    headers = {
        "Content-Type": "application/json",
        "Fiware-Service": FIWARE_SERVICE,
        "Fiware-ServicePath": FIWARE_SERVICE_PATH,
        # Instruct QuantumLeap to use the timestamp attribute as time_index.
        "Fiware-TimeIndex-Attribute": "timestamp",
    }

    response = requests.post(
        f"{QL_URL}/v2/notify",
        json=payload,
        headers=headers,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"QuantumLeap returned HTTP {response.status_code}: {response.text}"
        )


def import_csv(filename: str) -> None:
    batch: list[dict[str, Any]] = []
    imported = 0

    with open(filename, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {
            "timestamp",
            "entity_id",
            *ATTRIBUTE_TYPES.keys(),
        }

        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing CSV columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                batch.append(make_entity(row))
            except Exception as exc:
                raise ValueError(
                    f"Invalid data at CSV row {row_number}: {exc}"
                ) from exc

            if len(batch) >= BATCH_SIZE:
                send_batch(batch)
                imported += len(batch)
                print(f"Imported {imported} rows")
                batch.clear()

        if batch:
            send_batch(batch)
            imported += len(batch)

    print(f"Import completed: {imported} rows")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} FILE.csv", file=sys.stderr)
        raise SystemExit(2)

    import_csv(sys.argv[1])


if __name__ == "__main__":
    main()
