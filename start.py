from __future__ import annotations
from telegram import Update
from telegram.ext import ContextTypes
from config import get_settings
from redis_client import get_json
from download import handle_url, is_user_authenticated  # Import new helper
from user_settings import ensure_user

_settings = get_settings()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await ensure_user(user.id, user.username)

    # Deep-link handling: /start <param>
    args = context.args or []
    if args and args[0].startswith("iq_"):
        token = args[0][3:]
        data = await get_json(f"iq:{token}")
        if data and data.get("url"):
            # Check auth for PM deep-link
            if not await is_user_authenticated(user.id, update.effective_chat.type == "private"):
                await update.effective_message.reply_text("Authenticate with /auth <password> to download from inline.")
                return
            await update.effective_message.reply_text("Got it! Analyzing your link…")
            await handle_url(update, context, data["url"])
            return

    msg = (
        f"Hey {user.first_name or 'there'} 👋\n\n"
        f"I can download media from YouTube, Instagram, Facebook, TikTok, Twitter/X, Reddit, and more (via yt-dlp).\n"
        f"Send me a link or use /dl <url>. Try /settings to adjust defaults.\n\n"
        f"In groups, add me and share links. Use /help for commands.\n\n"
        f"🔒 Secure mode: Authenticate with /auth <password> to enable downloads in PM."
    )
    await update.effective_message.reply_text(msg)