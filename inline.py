from __future__ import annotations
import uuid
from urllib.parse import urlparse
from telegram import Update, InlineQueryResultArticle, InputTextMessageContent, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes
from redis_client import get_json, set_json
from config import get_settings
from text import extract_urls

_settings = get_settings()

async def inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    iq = update.inline_query
    query = iq.query or ""
    pass_prefix = f"pass:{_settings.admin_pass} " if _settings.admin_pass else ""

    # Group enabled check (inline only works in enabled groups)
    if iq.chat and iq.chat.type in ("group", "supergroup"):
        enabled = await get_json(f"group:{iq.chat.id}:enabled")
        if not enabled:
            await iq.answer(
                [],
                cache_time=1,
                switch_pm_text="Enable bot in this group with /enablebot (admins only).",
            )
            return

    # Admin pass prefix check for inline
    if _settings.admin_pass and not query.startswith(pass_prefix):
        await iq.answer(
            [],
            cache_time=1,
            switch_pm_text=f"To use inline, prefix with '{pass_prefix}' (e.g., {pass_prefix}your_link)",
        )
        return

    # Extract real query after prefix
    real_query = query[len(pass_prefix):] if _settings.admin_pass else query
    urls = extract_urls(real_query)
    results = []

    if not urls:
        await iq.answer(
            results,
            cache_time=1,
            switch_pm_text="Open bot to paste a link",
            switch_pm_parameter="start"
        )
        return

    url = urls[0]
    token = uuid.uuid4().hex
    # Store URL for 1 hour
    await set_json(f"iq:{token}", {"url": url}, ex=3600)

    host = urlparse(url).netloc
    title = f"Download from {host}"
    deep_link = f"https://t.me/{_settings.bot_name}?start=iq_{token}"

    results.append(
        InlineQueryResultArticle(
            id=token,
            title=title,
            description="Open the bot to pick format and download",
            input_message_content=InputTextMessageContent(f"{url}"),
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Open bot to download", url=deep_link)]])
        )
    )
    await iq.answer(results, cache_time=5, is_personal=True)