from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from customer_service_app.core.logging import get_logger


logger = get_logger("customer_service_app.request")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """请求上下文中间件。

    `RequestContextMiddleware(BaseHTTPMiddleware)` 表示继承 Starlette 中间件基类，
    类似 Java Web 里的 Filter / Interceptor。
    它给每次请求生成 request_id，并记录一条结构化访问日志。
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """处理每个 HTTP 请求。

        `call_next(request)` 表示把请求交给后面的路由继续处理；
        路由返回响应后，这里再补响应头和访问日志。
        """
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        started = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
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
