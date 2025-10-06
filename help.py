from __future__ import annotations
from telegram import Update
from telegram.ext import ContextTypes
from config import get_settings

_settings = get_settings()

HELP_TEXT = (
    "Commands:\n"
    "/start - Welcome message\n"
    "/help - Show this help\n"
    "/dl <url> - Analyze and download media from a URL\n"
    "/best <url> - Download best available format quickly\n"
    "/audio <url> - Download best audio only\n"
    "/stats - Admin: show usage stats\n"
    "/flushcache - Admin: clear cache\n\n"
    "Notes:\n"
    "- Some platforms require login; you can mount cookies.txt via env YTDLP_COOKIES_PATH.\n"
    "- Max file size is limited by Telegram (~2GB). Larger files are filtered out.\n"
    "- In groups, the bot can react to links if privacy mode is disabled."
)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(HELP_TEXT)