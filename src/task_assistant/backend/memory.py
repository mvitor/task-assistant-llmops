# src/task_assistant/backend/memory.py
from __future__ import annotations

import json

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from task_assistant.backend.agents import ConversationSession, _session

MAX_HISTORY = 20  # trim to last N messages to limit context window growth


def get_history(session_id: str) -> list[ModelMessage]:
    with _session() as db:
        row = db.get(ConversationSession, session_id)
        if row is None:
            return []
        return ModelMessagesTypeAdapter.validate_json(row.history_json)


def save_history(session_id: str, new_messages: list[ModelMessage]) -> None:
    with _session() as db:
        row = db.get(ConversationSession, session_id)
        existing = ModelMessagesTypeAdapter.validate_json(row.history_json) if row else []
        combined = (existing + list(new_messages))[-MAX_HISTORY:]
        serialised = ModelMessagesTypeAdapter.dump_json(combined).decode()
        if row is None:
            db.add(ConversationSession(session_id=session_id, history_json=serialised))
        else:
            row.history_json = serialised
        db.commit()


def clear_history(session_id: str) -> None:
    with _session() as db:
        row = db.get(ConversationSession, session_id)
        if row:
            db.delete(row)
            db.commit()
