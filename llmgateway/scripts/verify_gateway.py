from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx
import pytest


async def verify_live_models(project_root: Path) -> int:
    sys.path.insert(0, str(project_root))
    from app.config import load_config
    from app.main import create_app

    config = load_config()
    gateway_key = next(
        (key.get_secret_value() for key in config.api_keys if key.get_secret_value()),
        None,
    )
    if gateway_key is None:
        raise RuntimeError("Live check requires a configured non-empty Gateway API key.")

    app = create_app(config)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
            for model in ("deepseek-v4-pro", "deepseek-v4-flash"):
                if model not in config.models:
                    raise RuntimeError(f"Live check requires model alias {model!r} in gateway.yaml.")
                response = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {gateway_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                        "max_tokens": 16,
                    },
                )
                if response.is_error:
                    raise RuntimeError(
                        f"Live check failed for {model!r}: HTTP {response.status_code}."
                    )
                content = response.json()["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(f"Live check for {model!r} returned no text.")
                print(f"LIVE PASS {model}: {content.strip()[:80]}")
    return 0


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run offline checks and optional live model smoke tests.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call both configured DeepSeek model aliases; this uses real API quota.",
    )
    args = parser.parse_args()
    result = int(pytest.main(["-q", str(project_root / "tests")]))
    if result or not args.live:
        return result
    return asyncio.run(verify_live_models(project_root))


if __name__ == "__main__":
    sys.exit(main())
