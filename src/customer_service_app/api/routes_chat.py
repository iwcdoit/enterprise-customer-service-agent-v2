from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from customer_service_app.api.dependencies import get_customer_service_agent
from customer_service_app.domain.schemas import ChatRequest, ChatResponse
from customer_service_app.services.customer_service_agent import CustomerServiceAgent


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent: CustomerServiceAgent = Depends(get_customer_service_agent),
) -> ChatResponse:
    """返回完整回答和执行轨迹。"""
    return await agent.answer(request)


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    agent: CustomerServiceAgent = Depends(get_customer_service_agent),
) -> StreamingResponse:
    """流式聊天接口，返回 text/event-stream。"""
    return StreamingResponse(agent.stream_answer(request), media_type="text/event-stream")
