from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import select


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_service_app.infrastructure.db.models import Order  # noqa: E402
from customer_service_app.infrastructure.db.session import dispose_engine, session_context  # noqa: E402


async def seed() -> None:
    """写入或更新一条样例订单。

    这条订单用于验证 `query_order_status` 工具和退款工单流程。
    """
    try:
        async with session_context() as session:
            result = await session.execute(
                select(Order).where(
                    Order.tenant_id == "default",
                    Order.user_id == "u001",
                    Order.order_id == "202606040001",
                )
            )
            order = result.scalar_one_or_none()
            if order is None:
                order = Order(
                    tenant_id="default",
                    user_id="u001",
                    order_id="202606040001",
                    status="shipped",
                    logistics_company="顺丰速运",
                    tracking_number="SF1234567890",
                    metadata_json={"amount": 299, "product": "智能耳机", "after_sale_days": 7},
                )
                session.add(order)
                action = "created"
            else:
                order.status = "shipped"
                order.logistics_company = "顺丰速运"
                order.tracking_number = "SF1234567890"
                order.metadata_json = {"amount": 299, "product": "智能耳机", "after_sale_days": 7}
                action = "updated"
            await session.commit()
            print(f"Sample order {action}: tenant_id=default user_id=u001 order_id=202606040001")
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(seed())
