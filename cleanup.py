from __future__ import annotations
import asyncio
import time
from pathlib import Path
from config import get_settings
from logger import setup_logging

logger = setup_logging()
_settings = get_settings()

async def cleanup_loop():
    base = Path(_settings.download_dir)
    base.mkdir(parents=True, exist_ok=True)
    ttl = _settings.ttl_download_hours * 3600
    while True:
        try:
            now = time.time()
            removed = 0
            for p in base.glob("*"):
                try:
                    if p.is_file():
                        age = now - p.stat().st_mtime
                        if age > ttl:
                            p.unlink(missing_ok=True)
                            removed += 1
                except Exception:
                    continue
            if removed:
                logger.info("Cleanup removed %d files", removed)
        except Exception as e:
            logger.error("Cleanup error: %s", e)
        await asyncio.sleep(1800)  # every 30 minutes