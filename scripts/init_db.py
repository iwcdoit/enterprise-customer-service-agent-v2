from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# 让脚本可以直接导入项目源码。

from customer_service_app.infrastructure.db.session import create_db_schema, dispose_engine  # noqa: E402


async def main() -> None:
    """初始化数据库表结构。"""
    try:
        await create_db_schema()
        print("Database schema initialized.")
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
