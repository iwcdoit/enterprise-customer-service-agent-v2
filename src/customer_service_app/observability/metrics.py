from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from prometheus_client import Counter, Gauge, Histogram


HTTP_REQUESTS = Counter(
    "customer_service_http_requests_total",
    "HTTP requests handled by the customer service API.",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "customer_service_http_request_duration_seconds",
    "HTTP request latency.",
    ["method", "path"],
)
GRAPH_RUNS = Counter(
    "customer_service_graph_runs_total",
    "LangGraph runs by outcome.",
    ["status"],
)
GRAPH_NODE_LATENCY = Histogram(
    "customer_service_graph_node_duration_seconds",
    "LangGraph node execution latency.",
    ["node"],
)
HIL_WAITING = Gauge(
    "customer_service_hil_waiting",
    "Current number of graph runs waiting for human confirmation.",
)
HIL_DECISIONS = Counter(
    "customer_service_hil_decisions_total",
    "Human-in-the-loop decisions.",
    ["decision"],
)
HIL_WAIT_SECONDS = Histogram(
    "customer_service_hil_wait_duration_seconds",
    "Time between interrupt creation and user decision.",
)
LLM_CALLS = Counter(
    "customer_service_llm_calls_total",
    "LLM calls by model and result.",
    ["model", "status"],
)
LLM_TOKENS = Counter(
    "customer_service_llm_tokens_total",
    "LLM token usage.",
    ["model", "kind"],
)
TOOL_CALLS = Counter(
    "customer_service_tool_calls_total",
    "Tool calls by tool, transport, and result.",
    ["tool", "transport", "status"],
)
TOOL_LATENCY = Histogram(
    "customer_service_tool_duration_seconds",
    "Tool execution latency.",
    ["tool", "transport"],
)
MCP_CALLS = Counter(
    "customer_service_mcp_calls_total",
    "MCP calls by server, tool, and result.",
    ["server", "tool", "status"],
)
MCP_LATENCY = Histogram(
    "customer_service_mcp_call_duration_seconds",
    "MCP call latency.",
    ["server", "tool"],
)


@contextmanager
def observe_graph_node(node: str) -> Iterator[None]:
    """Measure one meaningful LangGraph node boundary."""

    started = perf_counter()
    try:
        yield
    finally:
        GRAPH_NODE_LATENCY.labels(node=node).observe(perf_counter() - started)


@contextmanager
def observe_tool(tool: str, transport: str) -> Iterator[None]:
    """Measure tool execution without recording tool arguments."""

    started = perf_counter()
    try:
        yield
    except Exception:
        TOOL_CALLS.labels(tool=tool, transport=transport, status="error").inc()
        raise
    else:
        TOOL_CALLS.labels(tool=tool, transport=transport, status="success").inc()
    finally:
        TOOL_LATENCY.labels(tool=tool, transport=transport).observe(perf_counter() - started)


@contextmanager
def observe_mcp(server: str, tool: str) -> Iterator[None]:
    """Measure one MCP call without placing business payloads in metrics."""

    started = perf_counter()
    try:
        yield
    except Exception:
        MCP_CALLS.labels(server=server, tool=tool, status="error").inc()
        raise
    else:
        MCP_CALLS.labels(server=server, tool=tool, status="success").inc()
    finally:
        MCP_LATENCY.labels(server=server, tool=tool).observe(perf_counter() - started)
