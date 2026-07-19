from __future__ import annotations

from dataclasses import dataclass

from customer_service_app.workflows.nodes import CustomerServiceGraphNodes


@dataclass(slots=True)
class CustomerServiceGraphContext:
    """Request-scoped dependencies supplied to nodes without checkpointing them."""

    nodes: CustomerServiceGraphNodes
