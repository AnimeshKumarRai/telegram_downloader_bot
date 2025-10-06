from __future__ import annotations
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, BigInteger, Integer, JSON, Text, DateTime
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict

from database import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    username: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    is_admin: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class CacheItem(Base):
    __tablename__ = "cache_items"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    url_hash: Mapped[str] = mapped_column(String(64), index=True)
    source_url: Mapped[str] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(Text)
    format_id: Mapped[str] = mapped_column(String(64))
    content_type: Mapped[str] = mapped_column(String(32))  # video/audio/document
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    telegram_file_id: Mapped[str] = mapped_column(Text)
    extra: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_used: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class JobLog(Base):
    __tablename__ = "job_logs"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    url: Mapped[str] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))  # success, error, canceled
    duration_ms: Mapped[int] = mapped_column(Integer)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class UserSetting(Base):
    __tablename__ = "user_settings"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)  # Foreign key to User.id
    key: Mapped[str] = mapped_column(String(64))  # Setting name (e.g., "preferred_format")
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Setting value
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class GroupApproval(Base):
    __tablename__ = "group_approvals"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    approved_by: Mapped[int] = mapped_column(BigInteger)
    approved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())