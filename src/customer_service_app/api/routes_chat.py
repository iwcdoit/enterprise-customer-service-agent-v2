from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from customer_service_app.api.dependencies import get_customer_service_agent
from customer_service_app.domain.schemas import ChatRequest, ChatResponse
from customer_service_app.services.customer_service_agent import CustomerServiceAgent


router = APIRouter(prefix="/chat", tags=["chat"])
"""聊天接口路由组。

最终完整路径会叠加 main.py 里的 `api_prefix`：
`/api/v1` + `/chat` = `/api/v1/chat`
"""


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent: CustomerServiceAgent = Depends(get_customer_service_agent),
) -> ChatResponse:
    """普通聊天接口，一次性返回完整答案和 trace。

    `@router.post` 是 FastAPI 装饰器，表示这个函数处理 POST 请求。
    `request: ChatRequest` 会自动从 JSON body 解析成 Pydantic 对象。
    """
    return await agent.answer(request)


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    agent: CustomerServiceAgent = Depends(get_customer_service_agent),
) -> StreamingResponse:
    """流式聊天接口，返回 text/event-stream。"""
    return StreamingResponse(agent.stream_answer(request), media_type="text/event-stream")
