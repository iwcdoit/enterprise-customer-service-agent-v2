from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy import JSON, DateTime, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from after_sales_mcp.config import get_settings


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Order(Base):
    """Read model owned by the after-sales capability service."""

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    order_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(64))
    logistics_company: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tracking_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class SupportTicket(Base):
    """Write model for after-sales and handoff requests."""

    __tablename__ = "support_tickets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    category: Mapped[str] = mapped_column(String(64))
    priority: Mapped[str] = mapped_column(String(32), default="normal")
    title: Mapped[str] = mapped_column(String(200))
    detail: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="open")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    idempotency_key: Mapped[str | None] = mapped_column(
        String(128), nullable=True, unique=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


_engine = create_async_engine(
    get_settings().database_url,
    pool_pre_ping=True,
    pool_recycle=1800,
)
_sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)


@asynccontextmanager
async def session_context() -> AsyncIterator[AsyncSession]:
    async with _sessionmaker() as session:
        yield session
