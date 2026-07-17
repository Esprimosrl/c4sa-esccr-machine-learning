#!/usr/bin/env python3

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import argparse
import json
import os
import socket


ENTITY_TYPES = {
    "RifiutiConferimento": "Storico conferimenti in QuantumLeap",
    "RifiutiDigitalTwin": "Storico digital twin in QuantumLeap",
}


def request_json(method, url, headers, timeout, payload=None, opener=urlopen):
    request = Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={
            **headers,
            **({"Content-Type": "application/json"} if payload is not None else {}),
        },
    )
    try:
        with opener(request, timeout=timeout) as response:
            body = response.read()
            return json.loads(body) if body else None
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Orion returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"Orion request failed: {exc}") from exc


def entity_types(subscription):
    return {
        entity.get("type")
        for entity in subscription.get("subject", {}).get("entities", [])
        if entity.get("type")
    }


def notification_url(subscription):
    return subscription.get("notification", {}).get("http", {}).get("url")


def manage_subscriptions(
    orion_url,
    notify_url,
    service,
    service_path,
    timeout,
    remove=False,
    opener=urlopen,
):
    base_url = orion_url.rstrip("/") + "/v2/subscriptions"
    headers = {
        "Accept": "application/json",
        "Fiware-Service": service,
        "Fiware-ServicePath": service_path,
    }
    subscriptions = request_json(
        "GET", f"{base_url}?limit=1000", headers, timeout, opener=opener
    )
    if not isinstance(subscriptions, list):
        raise RuntimeError("Orion returned an invalid subscription list.")

    actions = []
    if remove:
        needed = set(ENTITY_TYPES)
        for subscription in subscriptions:
            types = entity_types(subscription)
            if (
                types
                and types <= needed
                and notification_url(subscription) == notify_url
            ):
                subscription_id = subscription.get("id")
                if not subscription_id:
                    raise RuntimeError("Orion returned a subscription without an id.")
                request_json(
                    "DELETE",
                    f"{base_url}/{quote(subscription_id, safe='')}",
                    headers,
                    timeout,
                    opener=opener,
                )
                actions.append(f"removed {subscription_id} ({', '.join(sorted(types))})")
        return actions

    for entity_type, description in ENTITY_TYPES.items():
        exists = any(
            entity_type in entity_types(subscription)
            and notification_url(subscription) == notify_url
            and subscription.get("status", "active") == "active"
            for subscription in subscriptions
        )
        if exists:
            actions.append(f"already present ({entity_type})")
            continue
        request_json(
            "POST",
            base_url,
            headers,
            timeout,
            {
                "description": description,
                "subject": {
                    "entities": [{"idPattern": ".*", "type": entity_type}]
                },
                "notification": {"http": {"url": notify_url}},
            },
            opener,
        )
        actions.append(f"created ({entity_type})")
    return actions


def main():
    parser = argparse.ArgumentParser(
        description="Create or remove the Orion subscriptions required by this project."
    )
    parser.add_argument("--remove", action="store_true")
    parser.add_argument(
        "--orion-url", default=os.environ.get("ORION_URL", "http://localhost:1026")
    )
    parser.add_argument(
        "--notify-url",
        default=os.environ.get(
            "QL_NOTIFY_URL", "http://fiware-quantumleap:8668/v2/notify"
        ),
    )
    parser.add_argument(
        "--service", default=os.environ.get("FIWARE_SERVICE", "openiot")
    )
    parser.add_argument(
        "--service-path", default=os.environ.get("FIWARE_SERVICE_PATH", "/")
    )
    parser.add_argument(
        "--timeout", type=float, default=float(os.environ.get("ORION_TIMEOUT_SECONDS", "30"))
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    for action in manage_subscriptions(
        args.orion_url,
        args.notify_url,
        args.service,
        args.service_path,
        args.timeout,
        args.remove,
    ):
        print(action)


if __name__ == "__main__":
    main()
