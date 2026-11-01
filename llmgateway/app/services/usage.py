from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from app.config import GatewayConfig


@dataclass
class UsageEvent:
    request_id: str
    api_key_hash: str
    endpoint: str
    requested_model: str
    provider: str | None = None
    upstream_model: str | None = None
    stream: bool = False
    status: str = "success"
    status_code: int = 200
    input_tokens: int = 0
    uncached_input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0
    latency_ms: float = 0
    first_token_ms: float | None = None
    retries: int = 0
    fallbacks: int = 0
    error_type: str | None = None
    error_message: str | None = None
    prompt_id: str | None = None
    prompt_version: int | None = None
    metadata: dict[str, Any] | None = None


class UsageRepository:
    def __init__(self, database_path: str, config: GatewayConfig) -> None:
        self.database_path = database_path
        self.config = config

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    api_key_hash TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    requested_model TEXT NOT NULL,
                    provider TEXT,
                    upstream_model TEXT,
                    stream INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    uncached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL,
                    cached_tokens INTEGER NOT NULL,
                    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    first_token_ms REAL,
                    retries INTEGER NOT NULL,
                    fallbacks INTEGER NOT NULL,
                    error_type TEXT,
                    error_message TEXT,
                    prompt_id TEXT,
                    prompt_version INTEGER,
                    metadata_json TEXT
                )
                """
            )
            await db.execute("CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_events(created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_events(requested_model)")
            cursor = await db.execute("PRAGMA table_info(usage_events)")
            columns = {row[1] for row in await cursor.fetchall()}
            for name in ("uncached_input_tokens", "cache_creation_tokens", "reasoning_tokens"):
                if name not in columns:
                    await db.execute(
                        f"ALTER TABLE usage_events ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
                    )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS stream_checkpoints (
                    request_id TEXT PRIMARY KEY,
                    api TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    content TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await db.commit()

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0
    ) -> float:
        price = self.config.pricing.get(model)
        if price is None:
            return 0
        fresh_input = max(0, input_tokens - cached_tokens)
        cached_rate = price.cached_input_per_million
        cost = fresh_input * price.input_per_million / 1_000_000
        cost += (
            cached_tokens * (cached_rate if cached_rate is not None else price.input_per_million) / 1_000_000
        )
        cost += output_tokens * price.output_per_million / 1_000_000
        return round(cost, 10)

    async def record(self, event: UsageEvent) -> None:
        values = asdict(event)
        created_at = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(values.pop("metadata"), ensure_ascii=False) if event.metadata else None
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO usage_events (
                    request_id, created_at, api_key_hash, endpoint, requested_model, provider,
                    upstream_model, stream, status, status_code, input_tokens, uncached_input_tokens,
                    output_tokens, cached_tokens, cache_creation_tokens, reasoning_tokens, cost_usd,
                    latency_ms, first_token_ms, retries, fallbacks, error_type, error_message,
                    prompt_id, prompt_version, metadata_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    values["request_id"],
                    created_at,
                    values["api_key_hash"],
                    values["endpoint"],
                    values["requested_model"],
                    values["provider"],
                    values["upstream_model"],
                    int(values["stream"]),
                    values["status"],
                    values["status_code"],
                    values["input_tokens"],
                    values["uncached_input_tokens"],
                    values["output_tokens"],
                    values["cached_tokens"],
                    values["cache_creation_tokens"],
                    values["reasoning_tokens"],
                    values["cost_usd"],
                    values["latency_ms"],
                    values["first_token_ms"],
                    values["retries"],
                    values["fallbacks"],
                    values["error_type"],
                    values["error_message"],
                    values["prompt_id"],
                    values["prompt_version"],
                    metadata_json,
                ),
            )
            await db.commit()

    async def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM usage_events ORDER BY created_at DESC LIMIT ?", (min(limit, 1000),)
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def save_checkpoint(
        self,
        request_id: str,
        api: str,
        model: str,
        status: str,
        content: str,
    ) -> None:
        updated_at = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute(
                """
                INSERT INTO stream_checkpoints(request_id, api, model, status, content, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    status = excluded.status,
                    content = excluded.content,
                    updated_at = excluded.updated_at
                """,
                (request_id, api, model, status, content, updated_at),
            )
            await db.commit()

    async def get_checkpoint(self, request_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT request_id, api, model, status, content, updated_at "
                "FROM stream_checkpoints WHERE request_id = ?",
                (request_id,),
            )
            row = await cursor.fetchone()
        return dict(row) if row else None


def usage_from_payload(payload: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    usage = payload.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    details = usage.get("prompt_tokens_details", usage.get("input_tokens_details", {})) or {}
    has_anthropic_cache = "cache_read_input_tokens" in usage
    cached_value = usage.get(
        "cache_read_input_tokens",
        usage.get("cached_input_tokens", details.get("cached_tokens", 0)),
    )
    cached_tokens = int(
        cached_value or 0
    )
    cache_creation_tokens = int(usage.get("cache_creation_input_tokens", 0) or 0)
    output_details = usage.get("completion_tokens_details", usage.get("output_tokens_details", {})) or {}
    reasoning_tokens = int(output_details.get("reasoning_tokens", 0) or 0)
    if has_anthropic_cache:
        uncached_input_tokens = input_tokens
        input_tokens += cached_tokens + cache_creation_tokens
    else:
        uncached_input_tokens = max(0, input_tokens - cached_tokens - cache_creation_tokens)
    return (
        input_tokens,
        uncached_input_tokens,
        output_tokens,
        cached_tokens,
        cache_creation_tokens,
        reasoning_tokens,
    )
