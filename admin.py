from __future__ import annotations
from telegram import Update
from telegram.ext import ContextTypes
from sqlalchemy import select, func, delete
from config import get_settings
from database import SessionLocal
from models import CacheItem, JobLog

_settings = get_settings()

def is_admin(user_id: int) -> bool:
    return user_id in _settings.admins

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not is_admin(user.id):
        return
    async with SessionLocal() as ses:
        total_jobs = (await ses.execute(select(func.count()).select_from(JobLog))).scalar()
        success_jobs = (await ses.execute(select(func.count()).select_from(JobLog).where(JobLog.status == "success"))).scalar()
        cache_count = (await ses.execute(select(func.count()).select_from(CacheItem))).scalar()
    await update.effective_message.reply_text(
        f"Stats:\nJobs: {total_jobs} (success: {success_jobs})\nCache entries: {cache_count}"
    )

async def flushcache(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not is_admin(user.id):
        return
    async with SessionLocal() as ses:
        await ses.execute(delete(CacheItem))
        await ses.commit()
    await update.effective_message.reply_text("Cache cleared.")