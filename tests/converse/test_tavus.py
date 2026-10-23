import json

import httpx
import pytest
from pydantic import SecretStr

from legible.api.settings import Settings
from legible.converse.tavus import TavusClient, TavusError


def client_with(handler) -> TavusClient:
    return TavusClient(
        "secret-key",
        face_id="r79e1c033f",
        pal_id="p5317866",
        api_base="https://tavus.test/v2/",
        max_call_seconds=600,
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.anyio
async def test_create_conversation_sends_an_echo_mode_request():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "conversation_id": "c123456",
                "conversation_name": "Legible talk",
                "conversation_url": "https://tavus.daily.co/c123456",
                "status": "active",
                "meeting_token": "eyJhbGciOi.token",
                "created_at": "2026-10-23T10:00:00Z",
            },
        )

    conversation = await client_with(handler).create_conversation("Legible talk abc")

    assert conversation.conversation_id == "c123456"
    assert conversation.conversation_url == "https://tavus.daily.co/c123456"
    assert conversation.meeting_token == "eyJhbGciOi.token" and conversation.status == "active"
    request = seen[0]
    assert request.method == "POST" and str(request.url) == "https://tavus.test/v2/conversations"
    assert request.headers["x-api-key"] == "secret-key"
    body = json.loads(request.content)
    assert body["face_id"] == "r79e1c033f" and body["pal_id"] == "p5317866"
    assert body["conversation_name"] == "Legible talk abc" and body["require_auth"] is True
    assert body["properties"]["max_call_duration"] == 600
    assert body["properties"]["enable_closed_captions"] is False


@pytest.mark.anyio
async def test_meeting_token_is_optional():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"conversation_id": "c1", "conversation_url": "https://tavus.daily.co/c1"})

    conversation = await client_with(handler).create_conversation("x")
    assert conversation.meeting_token is None and conversation.status is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "body", "detail"),
    [
        (400, {"message": "User has reached maximum concurrent conversations"}, "maximum concurrent"),
        (400, {"error": "Invalid face_id"}, "Invalid face_id"),
        (401, {"message": "Invalid access token"}, "Invalid access token"),
        (502, None, "Bad Gateway"),
    ],
)
async def test_error_responses_raise_tavus_error(status, body, detail):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body) if body is not None else httpx.Response(status)

    with pytest.raises(TavusError) as caught:
        await client_with(handler).create_conversation("x")
    assert caught.value.status_code == status and detail in str(caught.value)


@pytest.mark.anyio
async def test_network_failure_and_incomplete_responses_raise_tavus_error():
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(TavusError, match="Could not reach Tavus"):
        await client_with(unreachable).create_conversation("x")

    def incomplete(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "active"})

    with pytest.raises(TavusError, match="conversation id"):
        await client_with(incomplete).create_conversation("x")


@pytest.mark.anyio
async def test_end_conversation_posts_to_the_end_endpoint():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    await client_with(handler).end_conversation("c 123/456")
    assert seen[0].method == "POST" and seen[0].content == b""
    assert seen[0].url.raw_path == b"/v2/conversations/c%20123%2F456/end"


def test_from_settings_needs_a_key_a_face_and_an_echo_pal(tmp_path):
    def settings(**overrides) -> Settings:
        values = {
            "environment": "test",
            "data_dir": tmp_path,
            "tavus_api_key": SecretStr("k"),
            "tavus_replica_id": "r1",
            "tavus_persona_id": "p1",
            "tavus_api_base": "https://tavusapi.com/v2",
            "tavus_max_call_seconds": 300,
        }
        return Settings(_env_file=None, **{**values, **overrides})  # ignore any local LEGIBLE_* setup

    assert TavusClient.from_settings(settings(tavus_api_key=None)) is None
    assert TavusClient.from_settings(settings(tavus_replica_id=None)) is None
    assert TavusClient.from_settings(settings(tavus_persona_id=None)) is None  # no echo-mode PAL
    client = TavusClient.from_settings(settings())
    assert client is not None and client.face_id == "r1" and client.pal_id == "p1"
    assert client.max_call_seconds == 300 and client.api_base == "https://tavusapi.com/v2"
