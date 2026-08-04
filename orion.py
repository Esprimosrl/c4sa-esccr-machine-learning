"""Small NGSIv2 batch publisher shared by training and inference."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
import socket


class OrionError(RuntimeError):
    pass


class OrionTimeout(OrionError):
    pass


def publish_entities(
    entities: Iterable[dict],
    url: str,
    fiware_service: str,
    fiware_service_path: str,
    timeout_seconds: float,
    batch_size: int = 50,
    opener=urlopen,
    concurrency: int = 2,
) -> int:
    if timeout_seconds <= 0 or batch_size <= 0 or concurrency <= 0:
        raise ValueError(
            "Orion timeout, batch size, and concurrency must be greater than zero."
        )
    endpoint = f"{url.rstrip('/')}/v2/op/update"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Fiware-Service": fiware_service,
        "Fiware-ServicePath": fiware_service_path,
    }

    def batches() -> Iterator[list[dict]]:
        iterator = iter(entities)
        while batch := list(islice(iterator, batch_size)):
            yield batch

    def send(batch: list[dict]) -> int:
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
        return len(batch)

    published = 0
    batch_iterator = iter(batches())
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while pending := list(islice(batch_iterator, concurrency)):
            for count in executor.map(send, pending):
                published += count
    return published
