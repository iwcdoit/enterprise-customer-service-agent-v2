from __future__ import annotations

from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.infrastructure.llm.base import LLMClient, LLMResponse, LLMToolCall


class OpenAICompatibleLLMClient(LLMClient):
    """OpenAI 兼容格式的聊天模型客户端。

    这里是项目真正调用大模型的出口。DeepSeek、通义千问、OpenAI 兼容网关等，
    只要接口协议兼容，都可以通过这里接入。
    """

    def __init__(self, settings: Settings):
        """构造方法：保存配置，真实 HTTP 客户端按需创建。"""
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """懒加载 LLM 客户端。

        `self._client is None` 表示第一次调用才创建连接对象；
        之后复用同一个客户端，避免每次请求重复初始化。
        """
        if self._client is None:
            api_key = self._settings.require("LLM_API_KEY", self._settings.llm_api_key)
            base_url = self._settings.require("LLM_BASE_URL", self._settings.llm_base_url)
            self._settings.require("LLM_MODEL", self._settings.llm_model)
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self._settings.llm_timeout_seconds,
            )
        return self._client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """调用非流式聊天接口。

        这里的关键点是：
        - `messages` 是真正传给大模型的上下文。
        - `tools` 是绑定给模型的工具定义，不是直接塞进用户 question。
        - 模型如果判断需要工具，会在响应里返回 `tool_calls`。
        """
        kwargs: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._settings.llm_temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        try:
            # create(**kwargs)等价于：
            # create(
            #     model="gpt-4o-mini",
            #     messages=[{"role": "user", "content": "你好"}],
            #     temperature=0.7,
            # )
            response = await self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ExternalServiceError(f"LLM request failed: {exc}") from exc

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
        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """调用流式聊天接口，把模型文本增量一段段 yield 给上层。

        `yield` 和 `return` 不一样：
        - `return` 是一次性返回结果。
        - `yield` 是每次产出一小段，前端就能边生成边显示。
        """
        try:
            stream = await self.client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                temperature=temperature if temperature is not None else self._settings.llm_temperature,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as exc:
            raise ExternalServiceError(f"LLM stream request failed: {exc}") from exc
