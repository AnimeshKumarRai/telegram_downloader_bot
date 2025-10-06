# from __future__ import annotations
import sys
import os
import signal
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import asyncio
import backoff
import telegram
from aiohttp import web
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ChatMemberHandler,
)
from config import get_settings
from logger import setup_logging
from database import init_db, healthcheck_db
from start import start
from help import help_cmd
from download import (
    dl_command,
    best_command,
    audio_command,
    link_listener,
    callback_handler,
    auth_command,  # ✅ new
)
from admin import stats, flushcache
from group import chat_member_update
from cleanup import cleanup_loop
from redis_client import get_json, set_json, delete_json

logger = setup_logging()
_settings = get_settings()


async def health_handler(request):
    ok_db = await healthcheck_db()
    return web.json_response({"status": "ok" if ok_db else "degraded", "db": ok_db})


async def start_http(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", _settings.health_port)
    await site.start()
    logger.info(f"Health server running on port {_settings.health_port}")


@backoff.on_exception(backoff.expo, telegram.error.TimedOut, max_tries=5, max_time=60)
async def robust_initialize(application):
    """Robust initialization with exponential backoff on timeouts."""
    await application.initialize()


# ========== GROUP ADMIN APPROVAL ==========

async def enablebot(update, context):
    """Allow group admins to enable downloads in their group."""
    if update.effective_chat.type not in ("group", "supergroup"):
        return await update.message.reply_text("This command is for groups only.")

    member = await context.bot.get_chat_member(update.effective_chat.id, update.effective_user.id)
    if member.status not in ("administrator", "creator"):
        return await update.message.reply_text("❌ Only group admins can enable me.")

    await set_json(f"group:{update.effective_chat.id}:enabled", True, ex=86400*30)
    await update.message.reply_text("✅ Bot enabled for this group! I will now auto-download links in medium quality.")


async def disablebot(update, context):
    """Allow group admins to disable downloads in their group."""
    if update.effective_chat.type not in ("group", "supergroup"):
        return await update.message.reply_text("This command is for groups only.")

    member = await context.bot.get_chat_member(update.effective_chat.id, update.effective_user.id)
    if member.status not in ("administrator", "creator"):
        return await update.message.reply_text("❌ Only group admins can disable me.")

    await delete_json(f"group:{update.effective_chat.id}:enabled")
    await update.message.reply_text("🚫 Bot disabled in this group.")


# ========== MAIN ==========
async def main():
    await init_db()

    # Build application with increased timeouts
    application = (
        ApplicationBuilder()
        .token(_settings.telegram_token)
        .concurrent_updates(True)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .connect_timeout(10.0)
        .pool_timeout(10.0)
        .build()
    )

    # Add commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("dl", dl_command))
    application.add_handler(CommandHandler("best", best_command))
    application.add_handler(CommandHandler("audio", audio_command))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("flushcache", flushcache))

    # Auth + group enable/disable
    application.add_handler(CommandHandler("auth", auth_command))
    application.add_handler(CommandHandler("enablebot", enablebot))
    application.add_handler(CommandHandler("disablebot", disablebot))

    # Inline callback
    application.add_handler(CallbackQueryHandler(callback_handler))

    # URL listeners
    application.add_handler(MessageHandler(filters.TEXT & filters.Entity("url"), link_listener))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"https?://"), link_listener))

    # Chat member updates (when added to group)
    application.add_handler(ChatMemberHandler(chat_member_update, ChatMemberHandler.MY_CHAT_MEMBER))

    # Initialize + start
    try:
        logger.info("Initializing application with retries...")
        await robust_initialize(application)
        await application.start()
        logger.info("Application started")
    except Exception as e:
        logger.error(f"Failed to initialize/start application after retries: %s", e)
        sys.exit(1)

    # HTTP health check
    http_app = web.Application()
    http_app.router.add_get("/health", health_handler)
    asyncio.create_task(start_http(http_app))

    # Cleanup loop
    asyncio.create_task(cleanup_loop())

    logger.info("Starting bot polling...")
    await application.updater.start_polling(drop_pending_updates=True)

    # Block until stop
    try:
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def signal_handler(sig):
            logger.info(f"Received signal {sig}, stopping...")
            stop_event.set()

        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(s, signal_handler, s)

        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logger.info("Application shut down gracefully")


if __name__ == "__main__":
    asyncio.run(main())
