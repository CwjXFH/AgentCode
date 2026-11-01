from __future__ import annotations

import json

import httpx
import pytest

from app.services.adapters import get_adapter
from app.services.gateway import GatewayService
from app.services.router import ModelRouter
from tests.conftest import gateway_client, make_config


def test_chat_json_schema_converts_to_responses_text_format():
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    request = get_adapter("openai_responses").build_request(
        "chat",
        {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": schema},
            },
        },
        "deepseek-v4-pro",
    )
    assert request.path == "/v1/responses"
    assert request.body["text"]["format"] == {
        "type": "json_schema",
        "name": "result",
        "strict": True,
        "schema": schema,
    }


def dual_protocol_config(database):
    return make_config(
        database,
        models={
            "deepseek-v4-pro": {
                "routes": [
                    {
                        "provider": "primary",
                        "model": "deepseek-v4-pro",
                        "protocol": "openai_responses",
                        "api": "chat",
                    }
                ]
            },
            "deepseek-v4-flash": {
                "routes": [
                    {
                        "provider": "fallback",
                        "model": "deepseek-flash",
                        "protocol": "anthropic_messages",
                        "api": "chat",
                    }
                ]
            },
        },
        providers={
            "primary": {
                "base_url": "https://primary.test",
                "api_key": "upstream-a",
                "endpoint_paths": {"openai_responses": "/responses"},
            },
            "fallback": {
                "base_url": "https://fallback.test/anthropic",
                "api_key": "upstream-b",
            },
        },
    )


@pytest.mark.asyncio
async def test_responses_and_anthropic_protocols_normalize_and_record_token_categories(
    tmp_path, auth_headers
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        if request.url.path == "/responses":
            assert request.headers["authorization"] == "Bearer upstream-a"
            assert body["model"] == "deepseek-v4-pro"
            assert body["instructions"] == "Be concise."
            assert body["input"] == [{"role": "user", "content": "Hello"}]
            return httpx.Response(
                200,
                json={
                    "id": "resp_1",
                    "output_text": "Pro response",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "input_tokens_details": {"cached_tokens": 2},
                        "output_tokens_details": {"reasoning_tokens": 1},
                    },
                },
            )
        assert request.url.path == "/anthropic/v1/messages"
        assert request.headers["x-api-key"] == "upstream-b"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert "authorization" not in request.headers
        assert body["model"] == "deepseek-flash"
        assert "Be concise." in body["system"]
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert body["max_tokens"] == 4096
        if "conforms to this schema" in body["system"]:
            return httpx.Response(
                200,
                json={
                    "id": "msg_structured",
                    "content": [{"type": "text", "text": '{"name":"Ada"}'}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 2,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "content": [{"type": "text", "text": "Flash response"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 2,
                },
            },
        )

    async with gateway_client(dual_protocol_config(tmp_path / "gateway.db"), handler) as client:
        pro = await client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
        flash = await client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "deepseek-v4-flash",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hello"},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "person",
                        "schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        )
        assert pro.status_code == flash.status_code == 200
        assert pro.json()["model"] == "deepseek-v4-pro"
        assert flash.json()["model"] == "deepseek-v4-flash"
        assert pro.json()["choices"][0]["message"]["content"] == "Pro response"
        assert json.loads(flash.json()["choices"][0]["message"]["content"]) == {"name": "Ada"}
        assert [request.url.path for request in requests] == [
            "/responses",
            "/anthropic/v1/messages",
        ]

        usage = (await client.get("/admin/usage", headers=auth_headers)).json()["data"]
        pro_usage = next(row for row in usage if row["requested_model"] == "deepseek-v4-pro")
        flash_usage = next(row for row in usage if row["requested_model"] == "deepseek-v4-flash")
        assert pro_usage["input_tokens"] == 10
        assert pro_usage["uncached_input_tokens"] == 8
        assert pro_usage["cached_tokens"] == 2
        assert pro_usage["reasoning_tokens"] == 1
        assert flash_usage["input_tokens"] == 17
        assert flash_usage["uncached_input_tokens"] == 12
        assert flash_usage["cached_tokens"] == 3
        assert flash_usage["cache_creation_tokens"] == 2
        assert flash_usage["latency_ms"] >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_path", "sse"),
    [
        (
            "deepseek-v4-pro",
            "/responses",
            'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Hi"}\n\n'
            'event: response.completed\ndata: {"type":"response.completed",'
            '"response":{"usage":{"input_tokens":4,"output_tokens":2}}}\n\n',
        ),
        (
            "deepseek-v4-flash",
            "/anthropic/v1/messages",
            'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":4}}}\n\n'
            'event: content_block_delta\ndata: {"type":"content_block_delta",'
            '"delta":{"type":"text_delta","text":"Hi"}}\n\n'
            'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":2}}\n\n'
            'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ),
    ],
)
async def test_provider_sse_is_normalized_to_chat_sse(
    tmp_path, auth_headers, model, expected_path, sse
):
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    async with gateway_client(dual_protocol_config(tmp_path / "gateway.db"), handler) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": model, "messages": [{"role": "user", "content": "Hello"}], "stream": True},
        ) as response:
            body = (await response.aread()).decode()
            request_id = response.headers["x-request-id"]
        assert response.status_code == 200
        assert seen_paths == [expected_path]
        assert '"object": "chat.completion.chunk"' in body
        assert '"content": "Hi"' in body
        assert "data: [DONE]" in body
        usage = (await client.get("/admin/usage", headers=auth_headers)).json()["data"][0]
        assert usage["request_id"] == request_id
        assert usage["input_tokens"] == 4
        assert usage["output_tokens"] == 2
        assert usage["first_token_ms"] is not None


@pytest.mark.asyncio
async def test_total_upstream_attempts_are_capped_at_three(tmp_path, auth_headers):
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": {"message": "temporarily unavailable"}})

    config = dual_protocol_config(tmp_path / "gateway.db")
    config.retry.max_attempts_per_route = 3
    config.retry.max_total_attempts = 3
    config.retry.base_delay_seconds = 0
    config.retry.max_delay_seconds = 0
    async with gateway_client(config, handler) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 502
        assert calls == 3
        event = (await client.get("/admin/usage", headers=auth_headers)).json()["data"][0]
        assert event["retries"] == 2


@pytest.mark.asyncio
async def test_backoff_delay_grows_exponentially(monkeypatch, tmp_path):
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    config = dual_protocol_config(tmp_path / "gateway.db")
    config.retry.base_delay_seconds = 0.5
    config.retry.max_delay_seconds = 2
    service = GatewayService(config, ModelRouter(config), None, None, None)
    monkeypatch.setattr("app.services.gateway.asyncio.sleep", record_sleep)
    monkeypatch.setattr("app.services.gateway.random.uniform", lambda low, high: 1)

    await service._backoff(0)
    await service._backoff(1)
    await service._backoff(2)

    assert delays == [0.5, 1, 2]
