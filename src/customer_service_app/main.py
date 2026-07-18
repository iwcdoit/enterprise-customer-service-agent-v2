from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from customer_service_app.api.routes_chat import router as chat_router
from customer_service_app.api.routes_confirmations import router as confirmations_router
from customer_service_app.api.routes_conversations import router as conversations_router
from customer_service_app.api.routes_health import router as health_router
from customer_service_app.api.routes_graph import router as graph_router
from customer_service_app.api.routes_human_support import router as human_support_router
from customer_service_app.api.routes_ops import router as ops_router
from customer_service_app.core.config import get_settings
from customer_service_app.core.exceptions import AppError
from customer_service_app.core.logging import configure_logging
from customer_service_app.core.middleware import RequestContextMiddleware
from customer_service_app.services.container import ApplicationContainer


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Graph 和 Checkpointer 是应用级资源，只在进程启动时创建一次。
        app.state.container = await ApplicationContainer.create(settings)
        try:
            yield
        finally:
            await app.state.container.close()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["https://console.example.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(ops_router)
    app.include_router(chat_router, prefix=settings.api_prefix)
    app.include_router(confirmations_router, prefix=settings.api_prefix)
    app.include_router(conversations_router, prefix=settings.api_prefix)
    app.include_router(human_support_router, prefix=settings.api_prefix)
    app.include_router(graph_router, prefix=settings.api_prefix)

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        """Convert application errors to a consistent JSON response."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
        )

    return app


app = create_app()
"""ASGI application entrypoint."""
