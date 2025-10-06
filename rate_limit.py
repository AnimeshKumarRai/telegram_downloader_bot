from __future__ import annotations
from typing import Callable, Awaitable
from telegram import Update
from telegram.ext import ContextTypes
from config import get_settings
from redis_client import incr, acquire_lock, release_lock

_settings = get_settings()

class RateLimiter:
    def __init__(self, per_minute: int, concurrent_per_user: int):
        self.per_minute = per_minute
        self.concurrent_per_user = concurrent_per_user

    async def allow_message(self, user_id: int) -> tuple[bool, str | None]:
        key = f"rl:msg:{user_id}"
        count = await incr(key, ttl=60)
        if count > self.per_minute:
            return False, "Rate limit reached. Try again in a minute."
        return True, None

    async def acquire_job(self, user_id: int) -> tuple[bool, str | None]:
        lock_key = f"rl:job:{user_id}"
        ok = await acquire_lock(lock_key, ttl=60*10)  # max 10 minutes per job
        if not ok:
            remain = await self._remaining(lock_key)
            return False, f"You already have a running job. Please wait {remain} seconds."
        return True, None

    async def release_job(self, user_id: int):
        await release_lock(f"rl:job:{user_id}")

    async def _remaining(self, key: str) -> int:
        from .redis_client import ttl
        t = await ttl(key)
        return max(t, 0)

rate_limiter = RateLimiter(_settings.per_user_per_minute, _settings.concurrent_per_user)