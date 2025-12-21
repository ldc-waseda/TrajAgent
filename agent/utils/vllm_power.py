#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to manage vLLM server power states via dev sleep mode endpoints.

Notes:
- Requires vLLM server started with env `VLLM_SERVER_DEV_MODE=1` and flag `--enable-sleep-mode`.
- Provides async helpers: `is_sleeping`, `wake_up`, `sleep`, and a compat `awake`.
- Endpoints (no `/v1` prefix):
  * POST /sleep?level=1
  * POST /wake_up?tags=weights (optional tags)
  * GET  /is_sleeping
- If endpoints are unavailable, calls are best-effort no-ops.
"""

import asyncio
import json
from typing import Optional, List

import aiohttp


def _root_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    # remove trailing /v1 if present
    if base.endswith("/v1"):
        base = base[:-3]
    return base


async def _post_json(url: str, payload: dict, api_key: str = "", timeout: float = 10.0) -> Optional[dict]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=json.dumps(payload), timeout=timeout) as resp:
                if resp.status // 100 == 2:
                    try:
                        return await resp.json()
                    except Exception:
                        return {"ok": True}
                else:
                    txt = await resp.text()
                    print(f"[vLLM] POST {url} failed: {resp.status} {txt}")
    except Exception as e:
        print(f"[vLLM] POST {url} exception: {e}")
    return None


async def _get_json(url: str, api_key: str = "", timeout: float = 10.0) -> Optional[dict]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=timeout) as resp:
                if resp.status // 100 == 2:
                    try:
                        return await resp.json()
                    except Exception:
                        txt = await resp.text()
                        return {"text": txt}
                else:
                    txt = await resp.text()
                    print(f"[vLLM] GET {url} failed: {resp.status} {txt}")
    except Exception as e:
        print(f"[vLLM] GET {url} exception: {e}")
    return None


async def is_sleeping(api_base: str, api_key: str = "", timeout: float = 5.0) -> Optional[bool]:
    """Query dev endpoint to determine if the engine is sleeping.

    Returns True/False when determinable, or None if endpoint unsupported.
    """
    root = _root_base(api_base)
    url = f"{root}/is_sleeping"
    res = await _get_json(url, api_key=api_key, timeout=timeout)
    if isinstance(res, dict):
        if "is_sleeping" in res:
            try:
                return bool(res["is_sleeping"])  # type: ignore[arg-type]
            except Exception:
                return None
        if "sleeping" in res:
            try:
                return bool(res["sleeping"])  # type: ignore[arg-type]
            except Exception:
                return None
    return None


async def wake_up(api_base: str, tags: Optional[List[str]] = None, api_key: str = "", timeout: float = 30.0) -> bool:
    """Wake up the engine via dev endpoint. Returns True if request succeeded."""
    root = _root_base(api_base)
    url = f"{root}/wake_up"
    if tags:
        # Support multiple tags by repeating the query param
        query = "&".join([f"tags={t}" for t in tags])
        url = f"{url}?{query}"
    res = await _post_json(url, payload={}, api_key=api_key, timeout=timeout)
    if res is not None:
        print(f"[vLLM] wake_up via {url}")
        return True
    print("[vLLM] wake_up not supported; continuing")
    return False


async def awake(api_base: str, model: str, api_key: str = "") -> None:
    """Compat helper: wake engine using dev endpoints if available."""
    sleeping = await is_sleeping(api_base, api_key=api_key)
    # If unknown or sleeping, try to wake up
    if sleeping is None or sleeping:
        await wake_up(api_base, api_key=api_key)
    else:
        print("[vLLM] engine already awake")


async def sleep(api_base: str, model: str, api_key: str = "", level: int = 1) -> None:
    """Put engine to sleep via dev endpoint. If unsupported, returns quickly.

    The `model` parameter is unused and kept for backward compatibility.
    """
    root = _root_base(api_base)
    url = f"{root}/sleep?level={level}"
    res = await _post_json(url, payload={}, api_key=api_key)
    if res is not None:
        print(f"[vLLM] sleep via {url}")
        return
    print("[vLLM] sleep not supported; skipping")


async def main():
    import os
    api_base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    model = os.environ.get("VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    action = os.environ.get("ACTION", "awake")
    if action == "awake":
        await awake(api_base, model, api_key)
    else:
        level = int(os.environ.get("LEVEL", "1"))
        await sleep(api_base, model, api_key, level=level)


if __name__ == "__main__":
    asyncio.run(main())
