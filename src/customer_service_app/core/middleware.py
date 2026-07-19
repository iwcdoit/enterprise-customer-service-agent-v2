from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from customer_service_app.core.logging import get_logger
from customer_service_app.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS


logger = get_logger("customer_service_app.request")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """为请求补充 request_id、访问日志和 HTTP 指标。"""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """执行请求链并记录结果。"""
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id

        started = time.perf_counter()
        response = await call_next(request)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        route = request.scope.get("route")
        # 使用路由模板而非原始 URL，避免 Prometheus 标签基数失控。
        path_template = getattr(route, "path", request.url.path)

        HTTP_REQUESTS.labels(
            method=request.method,
            path=path_template,
            status=str(response.status_code),
        ).inc()

        HTTP_LATENCY.labels(method=request.method, path=path_template).observe(
            elapsed_ms / 1000
        )

        response.headers["x-request-id"] = request_id

        logger.info(
            "request_finished",
            extra={
                "extra_fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "elapsed_ms": elapsed_ms,
                }
            },
        )
        return response
