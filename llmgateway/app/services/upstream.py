from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import GatewayConfig, ProviderConfig
from app.core.errors import UpstreamError
from app.services.adapters import get_adapter


@dataclass
class OpenStream:
    response: httpx.Response
    provider_name: str
    protocol: str


class UpstreamClient:
    def __init__(self, config: GatewayConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def _headers(
        self, provider: ProviderConfig, request_id: str, protocol: str, extra_headers: dict[str, str]
    ) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
            **provider.extra_headers,
            **extra_headers,
        }
        api_key = provider.api_key.get_secret_value()
        if api_key:
            if protocol == "anthropic_messages":
                headers["x-api-key"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _timeout(self, provider: ProviderConfig) -> httpx.Timeout:
        return httpx.Timeout(
            timeout=provider.timeout_seconds,
            connect=provider.connect_timeout_seconds,
        )

    async def request_json(
        self,
        provider_name: str,
        api: str,
        protocol: str,
        body: dict[str, Any],
        request_id: str,
        model: str,
    ) -> dict[str, Any]:
        provider = self.config.providers[provider_name]
        adapter = get_adapter(protocol)
        adapted = adapter.build_request(api, body, model)
        path = provider.endpoint_paths.get(protocol, adapted.path)
        url = f"{provider.base_url}{path}"
        try:
            response = await self.client.post(
                url,
                headers=self._headers(provider, request_id, protocol, adapted.headers),
                json=adapted.body,
                timeout=self._timeout(provider),
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise UpstreamError(f"{provider_name}: {type(exc).__name__}", retryable=True) from exc
        if response.is_error:
            raise self._http_error(provider_name, response)
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise UpstreamError(f"{provider_name}: upstream returned invalid JSON") from exc
        return adapter.normalize_response(api, payload, model)

    async def open_stream(
        self,
        provider_name: str,
        api: str,
        protocol: str,
        body: dict[str, Any],
        request_id: str,
        model: str,
    ) -> OpenStream:
        provider = self.config.providers[provider_name]
        adapted = get_adapter(protocol).build_request(api, body, model)
        path = provider.endpoint_paths.get(protocol, adapted.path)
        url = f"{provider.base_url}{path}"
        request = self.client.build_request(
            "POST",
            url,
            headers={
                **self._headers(provider, request_id, protocol, adapted.headers),
                "Accept": "text/event-stream",
            },
            json=adapted.body,
            timeout=self._timeout(provider),
        )
        try:
            response = await self.client.send(request, stream=True)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise UpstreamError(f"{provider_name}: {type(exc).__name__}", retryable=True) from exc
        if response.is_error:
            await response.aread()
            error = self._http_error(provider_name, response)
            await response.aclose()
            raise error
        return OpenStream(response=response, provider_name=provider_name, protocol=protocol)

    def _http_error(self, provider_name: str, response: httpx.Response) -> UpstreamError:
        retryable = response.status_code in self.config.retry.retry_statuses
        message = f"{provider_name}: HTTP {response.status_code}"
        details: Any = None
        try:
            details = response.json()
            if isinstance(details, dict):
                upstream_error = details.get("error")
                if isinstance(upstream_error, dict):
                    upstream_message = upstream_error.get("message")
                else:
                    upstream_message = details.get("message")
                if upstream_message:
                    message = f"{message}: {upstream_message}"
        except (json.JSONDecodeError, AttributeError):
            details = response.text[:1000]
        status = response.status_code if 400 <= response.status_code < 500 else 502
        return UpstreamError(message, status_code=status, retryable=retryable, details=details)
