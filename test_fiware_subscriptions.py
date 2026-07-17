from urllib.parse import urlparse
import json

import manage_fiware_subscriptions as subscriptions


class Response:
    def __init__(self, payload=None):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return b"" if self.payload is None else json.dumps(self.payload).encode()


def item(subscription_id, entity_types, url="http://ql/v2/notify", status="active"):
    return {
        "id": subscription_id,
        "status": status,
        "subject": {
            "entities": [{"idPattern": ".*", "type": value} for value in entity_types]
        },
        "notification": {"http": {"url": url}},
    }


def test_create_only_missing_subscription():
    requests = []

    def opener(request, timeout):
        requests.append(request)
        if request.method == "GET":
            return Response([item("source", ["RifiutiConferimento"])])
        return Response({})

    actions = subscriptions.manage_subscriptions(
        "http://orion/", "http://ql/v2/notify", "openiot", "/", 2, opener=opener
    )

    assert actions == [
        "already present (RifiutiConferimento)",
        "created (RifiutiDigitalTwin)",
    ]
    posts = [request for request in requests if request.method == "POST"]
    assert len(posts) == 1
    assert json.loads(posts[0].data)["subject"]["entities"][0]["type"] == "RifiutiDigitalTwin"
    assert posts[0].get_header("Fiware-service") == "openiot"
    assert posts[0].get_header("Content-type") == "application/json"
    assert requests[0].get_header("Content-type") is None


def test_remove_only_matching_needed_subscriptions():
    existing = [
        item("source/id", ["RifiutiConferimento"]),
        item("twin", ["RifiutiDigitalTwin"]),
        item("mixed", ["RifiutiDigitalTwin", "UnrelatedType"]),
        item("other-url", ["RifiutiConferimento"], "http://other/v2/notify"),
    ]
    requests = []

    def opener(request, timeout):
        requests.append(request)
        return Response(existing if request.method == "GET" else None)

    actions = subscriptions.manage_subscriptions(
        "http://orion", "http://ql/v2/notify", "openiot", "/", 2, True, opener
    )

    assert actions == [
        "removed source/id (RifiutiConferimento)",
        "removed twin (RifiutiDigitalTwin)",
    ]
    assert [urlparse(request.full_url).path for request in requests if request.method == "DELETE"] == [
        "/v2/subscriptions/source%2Fid",
        "/v2/subscriptions/twin",
    ]
    assert all(request.get_header("Content-type") is None for request in requests)
