from __future__ import annotations

import os

from customer_service_app.core.config import Settings


def configure_langsmith(settings: Settings) -> None:
    """Configure LangSmith before LangGraph and LLM clients start producing runs."""

    if not settings.langsmith_tracing:
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint


def wrap_openai_client(client):
    """Wrap an OpenAI-compatible client so model calls appear inside LangSmith runs."""

    try:
        from langsmith.wrappers import wrap_openai
    except ImportError:
        return client
    return wrap_openai(client)
