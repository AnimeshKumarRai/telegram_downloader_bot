from __future__ import annotations
from dataclasses import dataclass
from sqlalchemy import select
from database import SessionLocal
from models import User, UserSetting
from config import get_settings

_settings = get_settings()

@dataclass
class UserPrefs:
    default_quality: str = _settings.default_quality
    large_media_strategy: str = _settings.default_large_media_strategy
    prefer_document: bool = _settings.prefer_document_default
    # Note: This is DB-based; download.py uses Redis for prefs. Consider unifying if needed.

async def ensure_user(uid: int, username: str | None):
    async with SessionLocal() as ses:
        user = await ses.get(User, uid)
        if not user:
            ses.add(User(id=uid, username=username or None, is_admin=1 if uid in _settings.admins else 0))
            await ses.commit()

async def get_user_prefs(uid: int) -> UserPrefs:
    async with SessionLocal() as ses:
        row = await ses.get(UserSetting, uid)
        if not row:
            return UserPrefs()
        return UserPrefs(
            default_quality=row.default_quality or _settings.default_quality,
            large_media_strategy=row.large_media_strategy or _settings.default_large_media_strategy,
            prefer_document=bool(row.prefer_document or 0),
        )

async def set_user_pref(uid: int, key: str, value):
    async with SessionLocal() as ses:
        row = await ses.get(UserSetting, uid)
        if not row:
            row = UserSetting(user_id=uid)
            ses.add(row)
        if key == "default_quality":
            row.default_quality = str(value)
        elif key == "large_media_strategy":
            row.large_media_strategy = str(value)
        elif key == "prefer_document":
            row.prefer_document = 1 if bool(value) else 0
        await ses.commit()