import logging

from fastapi import APIRouter, Depends, HTTPException

from src.agents.graph import agent
from src.api.deps import current_user
from src.models.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(current_user)])
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat với AI agent."""
    try:
        result = await agent.ainvoke({"query": request.message})
        return ChatResponse(
            response=result.get("response", ""),
            analysis=result.get("analysis", ""),
        )
    except Exception as exc:
        logger.exception("Agent request failed")
        raise HTTPException(status_code=503, detail="Agent service is temporarily unavailable") from exc


@router.get("/status")
async def agent_status():
    """Kiểm tra trạng thái agent."""
    return {"status": "ready", "agent": "LangGraph Agent v1.0"}
