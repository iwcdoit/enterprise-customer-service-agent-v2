from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from customer_service_app.api import routes_health
from customer_service_app.core.config import Settings
from customer_service_app.core.middleware import RequestContextMiddleware


def _middleware_test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/items/{item_id}")
    async def read_item(item_id: str) -> dict[str, str]:
        return {"item_id": item_id}

    return app


def test_request_context_preserves_valid_request_id() -> None:
    client = TestClient(_middleware_test_app())

    response = client.get("/items/1", headers={"x-request-id": "gateway-req-001"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "gateway-req-001"


def test_request_context_replaces_untrusted_request_id() -> None:
    client = TestClient(_middleware_test_app())

    response = client.get("/items/1", headers={"x-request-id": "invalid request\nvalue"})

    assert response.status_code == 200
    generated = response.headers["x-request-id"]
    assert uuid.UUID(generated)


@pytest.mark.asyncio
async def test_readiness_accepts_milvus_vector_store(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        llm_api_key="secret",
        llm_base_url="https://llm.example.com/v1",
        llm_model="chat-model",
        database_url="mysql+aiomysql://user:secret@mysql/database",
        embedding_base_url="https://embedding.example.com/v1",
        embedding_model="embedding-model",
        vector_store_provider="milvus",
        milvus_uri="https://milvus.example.com",
    )
    monkeypatch.setattr(routes_health, "get_settings", lambda: settings)

    response = await routes_health.ready()

    assert response["status"] == "ok"
    assert response["checks"]["rag_configured"] is True


@pytest.mark.asyncio
async def test_readiness_rejects_incomplete_security_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        llm_api_key="secret",
        llm_base_url="https://llm.example.com/v1",
        llm_model="chat-model",
        database_url="mysql+aiomysql://user:secret@mysql/database",
        embedding_base_url="https://embedding.example.com/v1",
        embedding_model="embedding-model",
        qdrant_url="https://qdrant.example.com",
        security_enabled=True,
        jwt_secret_key="secret",
        jwt_issuer="",
        jwt_audience="customer-service",
    )
    monkeypatch.setattr(routes_health, "get_settings", lambda: settings)

    response = await routes_health.ready()

    assert response["status"] == "not_ready"
    assert response["checks"]["security_configured"] is False
