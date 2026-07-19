from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from customer_service_app.core.config import get_settings
from customer_service_app.infrastructure.db.models import Base


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """获取进程级异步数据库引擎。"""
    global _engine
    if _engine is None:
        settings = get_settings()
        database_url = settings.require("DATABASE_URL", settings.database_url)
        _engine = create_async_engine(database_url, pool_pre_ping=True, pool_recycle=1800)
    return _engine



def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """获取创建数据库 Session 的工厂对象。"""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """提供请求级数据库会话。"""
    async with get_sessionmaker()() as session:
        yield session


@asynccontextmanager
async def session_context() -> AsyncIterator[AsyncSession]:
    """提供脚本和后台任务使用的数据库会话。"""

    async with get_sessionmaker()() as session:
        yield session


async def create_db_schema() -> None:
    """根据 ORM 模型创建数据库表，主要用于本地开发和演示。"""
    async with get_engine().begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """关闭数据库连接池，通常在应用停止或测试结束时调用。"""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
