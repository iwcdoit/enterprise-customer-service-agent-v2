from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.infrastructure.llm.base import LLMClient, LLMResponse, LLMToolCall
from customer_service_app.observability.langsmith import wrap_openai_client


class OpenAICompatibleLLMClient(LLMClient):
    """Chat-model client for OpenAI-compatible providers."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Create the HTTP client lazily and wrap it with LangSmith when enabled."""

        if self._client is None:
            api_key = self._settings.require("LLM_API_KEY", self._settings.llm_api_key)
            base_url = self._settings.require("LLM_BASE_URL", self._settings.llm_base_url)
            self._settings.require("LLM_MODEL", self._settings.llm_model)
            self._client = wrap_openai_client(
                AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self._settings.llm_timeout_seconds,
                )
            )
        return self._client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """调用 OpenAI-compatible 对话接口并转换为内部响应。"""

        kwargs: dict[str, Any] = {
            "model": model or self._settings.llm_model,
            "messages": messages,
            "temperature": temperature or self._settings.llm_temperature,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        selected_model = str(kwargs["model"])
        try:
            response = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            logging.info("llm chat create error:{}",e)
            raise ExternalServiceError(f"llm request error: {e}") from e
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[LLMToolCall] = []
        for call in message.tool_calls or []:
            tool_calls.append(
                LLMToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
            )

        usage = response.usage
        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model=getattr(response, "model", None) or selected_model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )


    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """流式调用对话接口并逐段输出文本。"""

        kwargs: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": (
                temperature
                if temperature is not None
                else self._settings.llm_temperature
            ),
            "stream": True,
        }

        try:
            stream = await self.client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if delta.content:
                    yield delta.content

        except Exception as exc:
            logging.info("llm stream chat create error: %s", exc)
            raise ExternalServiceError(f"llm stream request error: {exc}") from exc


    async def close(self) -> None:
        """Close the shared HTTP connection pool during application shutdown."""

        if self._client is not None:
            await self._client.close()
