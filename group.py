from __future__ import annotations
from telegram import Update
from telegram.ext import ContextTypes
from logger import setup_logging
from sqlalchemy import select
from database import SessionLocal
from models import GroupApproval

logger = setup_logging()

async def chat_member_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Greeting when added to a group
    try:
        if update.my_chat_member:
            new = update.my_chat_member.new_chat_member
            if new and new.status in ("member", "administrator"):
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Hi! Share a video link and I'll help download it. If I don't react to plain links, disable privacy mode in @BotFather."
                )
    except Exception as e:
        logger.error("chat_member_update error: {}", e)

async def is_group_approved(chat_id: int) -> bool:
    """Check if a group is approved for inline queries."""
    async with SessionLocal() as ses:
        stmt = select(GroupApproval).where(GroupApproval.chat_id == chat_id)
        result = await ses.execute(stmt)
        return result.scalar_one_or_none() is not None

async def approve_group(chat_id: int, approved_by: int) -> bool:
    """Approve a group for inline queries (idempotent). Returns True if newly approved."""
    if await is_group_approved(chat_id):
        return False
    async with SessionLocal() as ses:
        ses.add(GroupApproval(chat_id=chat_id, approved_by=approved_by))
        await ses.commit()
    return True