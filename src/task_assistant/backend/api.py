# src/task_assistant/backend/api.py
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from task_assistant.backend.agents import init_db
from task_assistant.backend.middleware import MLflowLoggingMiddleware, register_prompts
from task_assistant.backend.memory import get_history, save_history



# -----------------------------------------------------------------------------
# Pydantic models for request/response
# -----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to the task assistant.")
    user_id: str = Field(..., description="User identifier (for session/memory scoping).")
    session_id: str | None = Field(None, description="Optional session identifier.")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response text.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata (e.g., run_id).")


class HealthResponse(BaseModel):
    status: str
    service: str


# -----------------------------------------------------------------------------
# FastAPI app with lifespan - run once at startup and once at shutdown,
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    register_prompts()
    yield


app = FastAPI(
    title="Task Assistant API",
    description="FastAPI backend for the MLOps/LLMOps task assistant.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(MLflowLoggingMiddleware)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="task-assistant")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    from task_assistant.backend.agents import task_agent

    session_id = req.session_id or req.user_id
    history = get_history(session_id)

    result = await task_agent.run(req.message, message_history=history)

    save_history(session_id, result.new_messages())

    return ChatResponse(
        response=result.output,
        metadata={},
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str) -> dict[str, str]:
    from task_assistant.backend.memory import clear_history
    clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------

def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("task_assistant.backend.api:app", host=host, port=port)


if __name__ == "__main__":
    main()