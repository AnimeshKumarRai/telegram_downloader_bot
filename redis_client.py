from __future__ import annotations
import asyncio
import json
import time
from typing import Any, Optional
import redis.asyncio as redis
from config import get_settings

_settings = get_settings()
_redis: redis.Redis | None = None
_lock = asyncio.Lock()

async def get_redis() -> redis.Redis:
    global _redis
    async with _lock:
        if _redis is None:
            _redis = redis.from_url(_settings.redis_url, decode_responses=True)
        return _redis

async def set_json(key: str, value: Any, ex: int | None = None):
    r = await get_redis()
    await r.set(key, json.dumps(value), ex=ex)

async def get_json(key: str) -> Optional[Any]:
    r = await get_redis()
    val = await r.get(key)
    if val is None:
        return None
    return json.loads(val)

async def incr(key: str, ttl: int | None = None) -> int:
    r = await get_redis()
    pipe = r.pipeline()
    pipe.incr(key)
    if ttl:
        pipe.expire(key, ttl)
    res = await pipe.execute()
    return int(res[0])

async def acquire_lock(key: str, ttl: int) -> bool:
    r = await get_redis()
    # NX = only set if not exists
    return await r.set(key, "1", ex=ttl, nx=True) is True

async def release_lock(key: str):
    r = await get_redis()
    await r.delete(key)

async def ttl(key: str) -> int:
    r = await get_redis()
    return await r.ttl(key)

async def delete_json(key: str):
    r = await get_redis()
    await r.delete(key)
