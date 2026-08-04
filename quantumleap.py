"""Small QuantumLeap NGSIv2 history client shared by training and inference."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import socket
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


PAGE_SIZE = 10_000


class QuantumLeapError(RuntimeError):
    pass


class QuantumLeapTimeout(QuantumLeapError):
    pass


class QuantumLeapDataError(QuantumLeapError):
    pass


def parse_datetime(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Dates must be ISO 8601 strings.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO 8601 date: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Date must include a timezone: {value}")
    return parsed


class QuantumLeapClient:
    def __init__(
        self,
        url: str,
        entity_type: str,
        fiware_service: str,
        fiware_service_path: str,
        timeout_seconds: float = 30,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        self.url = url.rstrip("/")
        self.entity_type = entity_type
        self.headers = {
            "Accept": "application/json",
            "Fiware-Service": fiware_service,
            "Fiware-ServicePath": fiware_service_path,
        }
        self.timeout_seconds = timeout_seconds

    def fetch(
        self,
        attrs: list[str],
        from_date: str,
        to_date: str,
        *,
        entity_ids: list[str] | None = None,
        id_pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for page in self.fetch_pages(
            attrs,
            from_date,
            to_date,
            entity_ids=entity_ids,
            id_pattern=id_pattern,
        ):
            rows.extend(page)
        return rows

    def fetch_pages(
        self,
        attrs: list[str],
        from_date: str,
        to_date: str,
        *,
        entity_ids: list[str] | None = None,
        id_pattern: str | None = None,
        concurrency: int = 1,
    ) -> Iterator[list[dict[str, Any]]]:
        if not attrs or len(attrs) != len(set(attrs)):
            raise ValueError("attrs must be a non-empty list without duplicates.")
        if entity_ids and id_pattern:
            raise ValueError("entity_ids and id_pattern are mutually exclusive.")
        if entity_ids is not None and not entity_ids:
            raise ValueError("entity_ids must not be empty.")
        if entity_ids and any(not isinstance(value, str) or not value for value in entity_ids):
            raise ValueError("entity_ids must contain non-empty strings.")
        if parse_datetime(from_date) > parse_datetime(to_date):
            raise ValueError("from_date must not be after to_date.")
        if concurrency <= 0:
            raise ValueError("concurrency must be greater than zero.")

        def get_page(offset: int) -> Any:
            params: dict[str, str | int] = {
                "attrs": ",".join(attrs),
                "fromDate": from_date,
                "toDate": to_date,
                "limit": PAGE_SIZE,
                "offset": offset,
            }
            if entity_ids:
                params["id"] = ",".join(entity_ids)
            if id_pattern:
                params["idPattern"] = id_pattern

            path = quote(self.entity_type, safe="")
            request = Request(
                f"{self.url}/v2/types/{path}?{urlencode(params)}",
                headers=self.headers,
            )
            return self._get_json(request)

        last_timestamp: dict[str, datetime] = {}
        offset = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            while True:
                offsets = [
                    offset + worker * PAGE_SIZE for worker in range(concurrency)
                ]
                for payload in executor.map(get_page, offsets):
                    page = self._normalize(payload, attrs, last_timestamp)
                    if page:
                        yield page
                    if len(page) < PAGE_SIZE:
                        return
                offset += concurrency * PAGE_SIZE

    def fetch_latest(
        self,
        attrs: list[str],
        *,
        entity_ids: list[str] | None = None,
        id_pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._fetch(
            attrs,
            entity_ids=entity_ids,
            id_pattern=id_pattern,
            extra_params={"lastN": 1},
        )

    def _fetch(
        self,
        attrs: list[str],
        *,
        entity_ids: list[str] | None = None,
        id_pattern: str | None = None,
        extra_params: dict[str, str | int] | None = None,
    ) -> list[dict[str, Any]]:
        if not attrs or len(attrs) != len(set(attrs)):
            raise ValueError("attrs must be a non-empty list without duplicates.")
        if entity_ids and id_pattern:
            raise ValueError("entity_ids and id_pattern are mutually exclusive.")
        if entity_ids is not None and not entity_ids:
            raise ValueError("entity_ids must not be empty.")
        if entity_ids and any(not isinstance(value, str) or not value for value in entity_ids):
            raise ValueError("entity_ids must contain non-empty strings.")

        rows: list[dict[str, Any]] = []
        offset = 0
        last_timestamp: dict[str, datetime] = {}
        while True:
            params: dict[str, str | int] = {
                "attrs": ",".join(attrs),
                "limit": PAGE_SIZE,
                "offset": offset,
                **(extra_params or {}),
            }
            if entity_ids:
                params["id"] = ",".join(entity_ids)
            if id_pattern:
                params["idPattern"] = id_pattern
            path = quote(self.entity_type, safe="")
            payload = self._get_json(
                Request(
                    f"{self.url}/v2/types/{path}?{urlencode(params)}",
                    headers=self.headers,
                )
            )
            page = self._normalize(payload, attrs, last_timestamp)
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                return rows
            offset += PAGE_SIZE

    def _get_json(self, request: Request) -> Any:
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code == 404:
                return {"entities": []}
            raise QuantumLeapError(f"QuantumLeap returned HTTP {exc.code}.") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise QuantumLeapTimeout("QuantumLeap request timed out.") from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise QuantumLeapTimeout("QuantumLeap request timed out.") from exc
            raise QuantumLeapError(f"QuantumLeap request failed: {exc.reason}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise QuantumLeapDataError("QuantumLeap returned invalid JSON.") from exc

    @staticmethod
    def _normalize(
        payload: Any,
        attrs: list[str],
        last_timestamp: dict[str, datetime],
    ) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
            raise QuantumLeapDataError(
                "QuantumLeap response must contain an entities array."
            )

        rows: list[dict[str, Any]] = []
        for entity in payload["entities"]:
            if not isinstance(entity, dict):
                raise QuantumLeapDataError("Each QuantumLeap entity must have a string id.")
            entity_id = entity.get("entityId", entity.get("id"))
            if not isinstance(entity_id, str):
                raise QuantumLeapDataError(
                    "Each QuantumLeap entity must have a string id."
                )
            indexes = entity.get("index")
            attributes = entity.get("attributes")
            if not isinstance(indexes, list) or not isinstance(attributes, list):
                raise QuantumLeapDataError(f"Malformed history for entity {entity_id}.")

            values_by_attr: dict[str, list[Any]] = {}
            for attribute in attributes:
                if not isinstance(attribute, dict):
                    raise QuantumLeapDataError(
                        f"Malformed attributes for entity {entity_id}."
                    )
                name, values = attribute.get("attrName"), attribute.get("values")
                if not isinstance(name, str) or not isinstance(values, list):
                    raise QuantumLeapDataError(f"Malformed attribute for entity {entity_id}.")
                if name in values_by_attr:
                    raise QuantumLeapDataError(
                        f"Duplicate attribute {name} for entity {entity_id}."
                    )
                values_by_attr[name] = values

            missing = set(attrs) - values_by_attr.keys()
            if missing:
                raise QuantumLeapDataError(
                    f"Entity {entity_id} is missing attributes: {', '.join(sorted(missing))}."
                )
            for name in attrs:
                if len(values_by_attr[name]) != len(indexes):
                    raise QuantumLeapDataError(
                        f"Parallel array length mismatch for {entity_id}.{name}."
                    )

            previous = last_timestamp.get(entity_id)
            for position, timestamp in enumerate(indexes):
                try:
                    parsed = parse_datetime(timestamp)
                except ValueError as exc:
                    raise QuantumLeapDataError(str(exc)) from exc
                if previous is not None and parsed < previous:
                    raise QuantumLeapDataError(
                        f"Timestamps are out of order for entity {entity_id}."
                    )
                previous = parsed
                values = {name: values_by_attr[name][position] for name in attrs}
                nulls = [name for name, value in values.items() if value is None]
                if nulls:
                    raise QuantumLeapDataError(
                        f"Null required values for {entity_id}: {', '.join(nulls)}."
                    )
                rows.append({"entity_id": entity_id, "timestamp": timestamp, **values})
            if previous is not None:
                last_timestamp[entity_id] = previous
        return rows
