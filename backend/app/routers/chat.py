from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.graph import run_agentic_rag

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
async def chat_endpoint(body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="`message` must not be empty")

    try:
        result = await run_agentic_rag(body.message)
    except Exception as exc:  # noqa: BLE001 — surface a clean error to the client
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "answer": result.answer,
        "sources": result.sources,
        "steps": [asdict(s) for s in result.steps],
    }
