import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.api.schemas import ChatRequest, ChatResponse
from app.agent.chat_agent import run_agent

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"/chat received {len(request.messages)} message(s)")
    reply, recommendations, eoc = await run_agent(request.messages)
    return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=eoc)