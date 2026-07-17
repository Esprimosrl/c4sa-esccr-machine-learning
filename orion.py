"""Small NGSIv2 batch publisher shared by training and inference."""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
import socket


class OrionError(RuntimeError):
    pass


class OrionTimeout(OrionError):
    pass


def publish_entities(
    entities: list[dict],
    url: str,
    fiware_service: str,
    fiware_service_path: str,
    timeout_seconds: float,
    batch_size: int = 20,
    opener=urlopen,
) -> int:
    if timeout_seconds <= 0 or batch_size <= 0:
        raise ValueError("Orion timeout and batch size must be greater than zero.")
    endpoint = f"{url.rstrip('/')}/v2/op/update"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Fiware-Service": fiware_service,
        "Fiware-ServicePath": fiware_service_path,
    }
    published = 0
    for offset in range(0, len(entities), batch_size):
        batch = entities[offset : offset + batch_size]
        request = Request(
            endpoint,
            data=json.dumps({"actionType": "append", "entities": batch}).encode(),
            method="POST",
            headers=headers,
        )
        try:
            with opener(request, timeout=timeout_seconds) as response:
                if not 200 <= response.status < 300:
                    raise OrionError(f"Orion returned HTTP {response.status}.")
        except HTTPError as exc:
            raise OrionError(f"Orion returned HTTP {exc.code}.") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise OrionTimeout("Orion request timed out.") from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise OrionTimeout("Orion request timed out.") from exc
            raise OrionError(f"Orion request failed: {exc.reason}") from exc
        published += len(batch)
    return published
