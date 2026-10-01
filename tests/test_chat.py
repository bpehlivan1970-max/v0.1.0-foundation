"""Offline tests: the real SDK talks to a mock HTTP transport serving SSE."""

from __future__ import annotations

import json

import anthropic
import httpx2
import pytest

from claude_foundation import ChatSession, RefusalError, Settings


def _sse(events: list[dict]) -> bytes:
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
    ).encode()


def _reply(text: str, stop_reason: str = "end_turn", stop_details=None) -> list[dict]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1", "type": "message", "role": "assistant",
                "model": "claude-opus-5-5", "content": [], "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta",
         "delta": {"stop_reason": stop_reason, "stop_sequence": None,
                   "stop_details": stop_details},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]


@pytest.fixture
def harness():
    requests: list[httpx2.Request] = []
    replies: list[list[dict]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200, content=_sse(replies.pop(0)),
            headers={"content-type": "text/event-stream", "request-id": "req_test"},
        )

    client = anthropic.Anthropic(
        api_key="test",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return client, requests, replies


def test_streams_text_and_builds_history(harness):
    client, requests, replies = harness
    replies += [_reply("Hello!"), _reply("Your name is Ada.")]
    session = ChatSession(client=client, settings=Settings())

    chunks: list[str] = []
    result = session.send("Hi, I'm Ada.", on_text=chunks.append)
    assert result.text == "Hello!" and chunks == ["Hello!"]
    assert result.output_tokens == 5 and result.served_by == "claude-opus-5-5"

    session.send("What's my name?")
    body = json.loads(requests[1].content)
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert len(session.history) == 4


def test_request_shape(harness):
    client, requests, replies = harness
    replies.append(_reply("ok"))
    ChatSession(client=client, settings=Settings()).send("ping")

    body = json.loads(requests[0].content)
    assert body["model"] == "claude-opus-5-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"}
    assert body["fallbacks"] == "default"
    assert body["stream"] is True
    assert "server-side-fallback-2026-07-01" in requests[0].headers["anthropic-beta"]


def test_refusal_raises_and_rolls_back_history(harness):
    client, _, replies = harness
    replies.append(_reply("", stop_reason="refusal",
                          stop_details={"type": "refusal", "category": "cyber",
                                        "explanation": "declined"}))
    session = ChatSession(client=client, settings=Settings())

    with pytest.raises(RefusalError) as exc:
        session.send("something declined")
    assert exc.value.category == "cyber"
    assert session.history == []


def test_api_error_rolls_back_history():
    def handler(request):
        return httpx2.Response(401, json={"type": "error", "error": {
            "type": "authentication_error", "message": "bad key"}})

    client = anthropic.Anthropic(
        api_key="bad", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    session = ChatSession(client=client, settings=Settings())
    with pytest.raises(anthropic.AuthenticationError):
        session.send("hi")
    assert session.history == []
