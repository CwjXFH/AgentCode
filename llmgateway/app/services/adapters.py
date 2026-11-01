from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.errors import GatewayError
from app.services.usage import usage_from_payload


@dataclass(frozen=True)
class AdaptedRequest:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


class ProviderAdapter(Protocol):
    def build_request(self, api: str, body: dict[str, Any], model: str) -> AdaptedRequest: ...

    def normalize_response(self, api: str, payload: dict[str, Any], model: str) -> dict[str, Any]: ...


class OpenAIChatAdapter:
    def build_request(self, api: str, body: dict[str, Any], model: str) -> AdaptedRequest:
        if api != "chat":
            raise GatewayError(
                "OpenAI Chat Completions adapter only supports chat requests",
                status_code=400,
                error_type="invalid_request_error",
                code="unsupported_api_protocol",
            )
        return AdaptedRequest("/v1/chat/completions", {}, {**body, "model": model})

    def normalize_response(self, api: str, payload: dict[str, Any], model: str) -> dict[str, Any]:
        return payload


class OpenAIResponsesAdapter:
    def build_request(self, api: str, body: dict[str, Any], model: str) -> AdaptedRequest:
        if api == "responses":
            return AdaptedRequest("/v1/responses", {}, {**body, "model": model})
        if api != "chat":
            raise GatewayError(
                f"OpenAI Responses adapter does not support {api!r}",
                status_code=400,
                error_type="invalid_request_error",
                code="unsupported_api_protocol",
            )

        instructions: list[str] = []
        input_items: list[dict[str, Any]] = []
        for message in body.get("messages", []):
            role = message.get("role")
            content = message.get("content")
            if role in {"system", "developer"}:
                if isinstance(content, str):
                    instructions.append(content)
                continue
            input_items.append({"role": role, "content": content})

        converted: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": body.get("stream", False),
        }
        if instructions:
            converted["instructions"] = "\n\n".join(instructions)
        for key in ("temperature", "top_p"):
            if key in body:
                converted[key] = body[key]
        token_limit = body.get("max_completion_tokens", body.get("max_tokens"))
        if token_limit is not None:
            converted["max_output_tokens"] = token_limit
        response_format = body.get("response_format")
        if response_format:
            json_schema = response_format.get("json_schema")
            output_format = (
                {"type": "json_schema", **json_schema}
                if response_format.get("type") == "json_schema" and isinstance(json_schema, dict)
                else {"type": response_format.get("type", "json_object")}
            )
            converted["text"] = {"format": output_format}
        return AdaptedRequest("/v1/responses", {}, converted)

    def normalize_response(self, api: str, payload: dict[str, Any], model: str) -> dict[str, Any]:
        if api == "responses":
            return payload
        text = payload.get("output_text")
        if not isinstance(text, str):
            text = "".join(
                item.get("text", "")
                for output in payload.get("output", [])
                for item in output.get("content", [])
                if item.get("type") == "output_text"
            )
        input_tokens, uncached_tokens, output_tokens, cached_tokens, cache_creation, reasoning = (
            usage_from_payload(payload)
        )
        return {
            "id": payload.get("id", f"chatcmpl_{uuid.uuid4().hex}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "uncached_input_tokens": uncached_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
                "completion_tokens_details": {"reasoning_tokens": reasoning},
                "cache_creation_input_tokens": cache_creation,
            },
        }


class AnthropicMessagesAdapter:
    def build_request(self, api: str, body: dict[str, Any], model: str) -> AdaptedRequest:
        if api != "chat":
            raise GatewayError(
                "Anthropic Messages adapter only supports chat requests",
                status_code=400,
                error_type="invalid_request_error",
                code="unsupported_api_protocol",
            )
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        for message in body.get("messages", []):
            role = message.get("role")
            content = message.get("content")
            if role in {"system", "developer"}:
                if isinstance(content, str):
                    system_parts.append(content)
                continue
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise GatewayError(
                    f"Anthropic Messages adapter does not support message role/content {role!r}",
                    status_code=422,
                    error_type="invalid_request_error",
                    code="unsupported_message_format",
                )
            messages.append({"role": role, "content": content})

        response_format = body.get("response_format") or {}
        if response_format.get("type") == "json_schema":
            schema = (response_format.get("json_schema") or {}).get("schema")
            system_parts.append(
                "Return only JSON that conforms to this schema:\n"
                + json.dumps(schema, ensure_ascii=False)
            )
        converted: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": body.get("max_completion_tokens", body.get("max_tokens", 4096)),
            "stream": body.get("stream", False),
        }
        if system_parts:
            converted["system"] = "\n\n".join(system_parts)
        for key in ("temperature", "top_p"):
            if key in body:
                converted[key] = body[key]
        return AdaptedRequest(
            "/v1/messages",
            {"anthropic-version": "2023-06-01"},
            converted,
        )

    def normalize_response(self, api: str, payload: dict[str, Any], model: str) -> dict[str, Any]:
        content = "".join(
            block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
        )
        input_tokens, uncached_tokens, output_tokens, cached_tokens, cache_creation, reasoning = (
            usage_from_payload(payload)
        )
        return {
            "id": f"chatcmpl_{payload.get('id', uuid.uuid4().hex)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "length" if payload.get("stop_reason") == "max_tokens" else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "uncached_input_tokens": uncached_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
                "completion_tokens_details": {"reasoning_tokens": reasoning},
                "cache_creation_input_tokens": cache_creation,
            },
        }


ADAPTERS: dict[str, ProviderAdapter] = {
    "openai_chat": OpenAIChatAdapter(),
    "openai_responses": OpenAIResponsesAdapter(),
    "anthropic_messages": AnthropicMessagesAdapter(),
}


def get_adapter(protocol: str) -> ProviderAdapter:
    try:
        return ADAPTERS[protocol]
    except KeyError as exc:
        raise GatewayError(
            f"Unsupported provider protocol: {protocol}",
            status_code=400,
            error_type="configuration_error",
            code="unsupported_provider_protocol",
        ) from exc


@dataclass(frozen=True)
class StreamResult:
    chunks: list[bytes]
    text: str = ""
    usage: tuple[int, int, int, int, int, int] | None = None
    content_started: bool = False
    finished: bool = False


class StreamNormalizer:
    def __init__(self, protocol: str, request_id: str, model: str, output_api: str = "chat") -> None:
        self.protocol = protocol
        self.output_api = output_api
        self.request_id = request_id
        self.model = model
        self.buffer = b""
        self.created = int(time.time())
        self.chunk_id = f"chatcmpl_{uuid.uuid4().hex}"
        self.finished = False

    def feed(self, chunk: bytes) -> StreamResult:
        self.buffer = (self.buffer + chunk).replace(b"\r\n", b"\n")
        chunks: list[bytes] = []
        text_parts: list[str] = []
        token_usage = None
        content_started = False
        finished = False
        while b"\n\n" in self.buffer:
            frame, self.buffer = self.buffer.split(b"\n\n", 1)
            lines = frame.decode("utf-8", errors="replace").replace("\r", "").splitlines()
            data = "\n".join(line[5:].lstrip() for line in lines if line.startswith("data:"))
            event_name = next(
                (line[6:].strip() for line in lines if line.startswith("event:")),
                "",
            )
            if not data:
                continue
            if data == "[DONE]":
                if self.protocol == "openai_chat" or self.output_api == "responses":
                    chunks.append(b"data: [DONE]\n\n")
                finished = True
                self.finished = True
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                if self.protocol == "openai_chat":
                    chunks.append(frame + b"\n\n")
                continue

            usage_payload = payload.get("response", payload)
            if self.protocol == "anthropic_messages" and event_name == "message_start":
                usage_payload = payload.get("message", payload)
            usage_values = usage_from_payload(usage_payload)
            if any(usage_values):
                token_usage = (
                    usage_values
                    if token_usage is None
                    else tuple(
                        max(current, incoming)
                        for current, incoming in zip(token_usage, usage_values, strict=True)
                    )
                )

            if self.output_api == "responses" and self.protocol == "openai_responses":
                chunks.append(frame + b"\n\n")
                if payload.get("type") == "response.output_text.delta":
                    delta_text = payload.get("delta")
                    if isinstance(delta_text, str):
                        text_parts.append(delta_text)
                        content_started = True
                if payload.get("type") == "response.completed":
                    finished = True
                    self.finished = True
                continue

            if self.protocol == "openai_chat":
                choices = payload.get("choices") or []
                if choices:
                    choice = choices[0]
                    delta = (choice.get("delta") or {}).get("content")
                    if isinstance(delta, str) and delta:
                        text_parts.append(delta)
                        content_started = True
                    if choice.get("finish_reason") is not None:
                        finished = True
                chunks.append(frame + b"\n\n")
                if data == "[DONE]":
                    finished = True
                continue

            delta_text: str | None = None
            if self.protocol == "anthropic_messages":
                if event_name == "content_block_delta":
                    delta_text = (payload.get("delta") or {}).get("text")
                if event_name == "message_stop":
                    chunks.append(self._chat_chunk({}, "stop"))
                    chunks.append(b"data: [DONE]\n\n")
                    finished = True
            elif self.protocol == "openai_responses":
                event_type = payload.get("type", event_name)
                if event_type == "response.output_text.delta":
                    delta_text = payload.get("delta")
                if event_type == "response.completed":
                    chunks.append(self._chat_chunk({}, "stop"))
                    chunks.append(b"data: [DONE]\n\n")
                    finished = True

            if isinstance(delta_text, str) and delta_text:
                text_parts.append(delta_text)
                content_started = True
                chunks.append(self._chat_chunk({"content": delta_text}))
            self.finished = self.finished or finished

        return StreamResult(
            chunks=chunks,
            text="".join(text_parts),
            usage=token_usage,
            content_started=content_started,
            finished=finished,
        )

    def _chat_chunk(self, delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
        payload = {
            "id": self.chunk_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
