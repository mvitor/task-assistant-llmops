from __future__ import annotations
from datetime import datetime, timezone
from typing import Any

import mlflow
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    Integer,
    DateTime,
    Boolean,
    select,
)
from sqlalchemy.orm import Session, DeclarativeBase, Mapped, mapped_column


# MLflow tracking server and experiment
from task_assistant.utils.constants import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Autolog for agent tracing and LLM calls
mlflow.pydantic_ai.autolog()

SYSTEM_PROMPT = """
Act as my expert Chief of Staff and personal lifestyle assistant. Your goal is to help me balance, organize, and execute both my work projects and personal life with maximum efficiency.

When I give you a brain dump, messy notes, or a list of tasks, you must:
1. Segregate inputs clearly into two distinct buckets: [WORK] and [PERSONAL].
2. Deconstruct complex items into clear, atomic, and actionable subtasks.
3. Prioritize both buckets using the Eisenhower Matrix, ensuring personal well-being is not overshadowed by work.
4. Identify hidden gaps, scheduling conflicts, missing dependencies, or boundary issues based *only* on the provided context. Do not list missing information, missing files, or things you do not know about unless they directly block an immediate task.
5. Flag what needs my direct focus today versus what can be scheduled, automated, or delegated.

Strict Operational Rule: Focus strictly on organizing and optimizing the data provided. Do not generate generic lists of assumptions, disclaimers, or metadata about what data or context you lack.

Keep your responses scannable, direct, and action-oriented. Do not write fluff. End every response by asking me one precise clarifying question to move the highest-priority task forward.
"""

def _load_system_prompt() -> str:
    try:
        prompt = mlflow.genai.load_prompt(
            "prompts:/task-assistant-system@latest",
            allow_missing=True,
            link_to_model=False,
        )
        if prompt is not None:
            return prompt.template
    except Exception:
        pass
    return SYSTEM_PROMPT


task_agent = Agent(
    model=OpenAIChatModel(
        model_name="llama3.1:latest",
        provider=OpenAIProvider(
            api_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
        ),
    ),
    system_prompt=_load_system_prompt(),
)

# Sql Alchemy ORM models for tasks and database session management
class Base(DeclarativeBase):
    pass


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="todo"
    )  # todo, doing, done
    priority: Mapped[str] = mapped_column(
        String(50), nullable=False, default="medium"
    )  # low, medium, high
    due_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


def _engine():
    from task_assistant.utils.constants import DATABASE_URL

    return create_engine(DATABASE_URL, echo=False)


def init_db():
    engine = _engine()
    Base.metadata.create_all(engine)


def _session() -> Session:
    engine = _engine()
    return Session(engine, expire_on_commit=False)


# -----------------------------------------------------------------------------
# Memory / task tools for the agent
# -----------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _list_tasks(status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """
    List tasks, optionally filtered by status (todo, doing, done).
    Returns a list of dicts with basic fields.
    """
    with _session() as session:
        stmt = select(Task)
        if status:
            stmt = stmt.where(Task.status == status.lower())
        stmt = stmt.order_by(Task.created_at.desc()).limit(limit)
        rows = session.execute(stmt).scalars().all()

        return [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "status": t.status,
                "priority": t.priority,
                "due_date": t.due_date.isoformat() if t.due_date else None,
                "completed": t.completed,
                "created_at": t.created_at.isoformat(),
                "updated_at": t.updated_at.isoformat(),
            }
            for t in rows
        ]


def _create_task(
    title: str,
    description: str | None = None,
    priority: str = "medium",
    due_date: str | None = None,
) -> dict[str, Any]:
    """
    Create a new task.
    - title: short title (required)
    - description: optional details
    - priority: low, medium, high (default: medium)
    - due_date: ISO 8601 datetime string (optional), e.g. '2026-08-25T18:00:00Z'
    Returns the created task as a dict.
    """
    with _session() as session:
        due_dt = None
        if due_date and due_date.lower() not in {"null", "none"}:
            due_dt = datetime.fromisoformat(due_date.replace("Z", "+00:00"))

        task = Task(
            title=title,
            description=description,
            priority=priority.lower(),
            due_date=due_dt,
            status="todo",
            completed=False,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        return {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "status": task.status,
            "priority": task.priority,
            "due_date": task.due_date.isoformat() if task.due_date else None,
            "completed": task.completed,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
        }


def _update_task_status(task_id: int, status: str) -> dict[str, Any] | None:
    """
    Update task status (todo, doing, done).
    Returns the updated task as a dict, or None if not found.
    """
    status = status.lower()
    if status not in {"todo", "doing", "done"}:
        raise ValueError("status must be one of: todo, doing, done")

    with _session() as session:
        task = session.get(Task, task_id)
        if not task:
            return None

        task.status = status
        task.completed = status == "done"
        session.commit()
        session.refresh(task)

        return {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "status": task.status,
            "priority": task.priority,
            "due_date": task.due_date.isoformat() if task.due_date else None,
            "completed": task.completed,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
        }


def _get_task(task_id: int) -> dict[str, Any] | None:
    """
    Get a single task by ID.
    Returns the task as a dict, or None if not found.
    """
    with _session() as session:
        task = session.get(Task, task_id)
        if not task:
            return None

        return {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "status": task.status,
            "priority": task.priority,
            "due_date": task.due_date.isoformat() if task.due_date else None,
            "completed": task.completed,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
        }


def _search_tasks(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Search tasks by title/description (simple LIKE search).
    Returns a list of matching tasks as dicts.
    """
    q = f"%{query}%"
    with _session() as session:
        stmt = (
            select(Task)
            .where((Task.title.ilike(q)) | (Task.description.ilike(q)))
            .order_by(Task.created_at.desc())
            .limit(limit)
        )
        rows = session.execute(stmt).scalars().all()

        return [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "status": t.status,
                "priority": t.priority,
                "due_date": t.due_date.isoformat() if t.due_date else None,
                "completed": t.completed,
                "created_at": t.created_at.isoformat(),
                "updated_at": t.updated_at.isoformat(),
            }
            for t in rows
        ]


def _summarize_my_day() -> str:
    """
    Generate a short text summary of today's tasks.
    Useful as a tool the agent can call to give you a quick overview of the day.
    """
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    today_end = datetime.now(timezone.utc).replace(
        hour=23, minute=59, second=59, microsecond=999999
    )

    with _session() as session:
        stmt = (
            select(Task)
            .where(
                Task.created_at >= today_start,
                Task.created_at <= today_end,
            )
            .order_by(Task.created_at)
        )
        rows = session.execute(stmt).scalars().all()

    if not rows:
        return "No tasks created today."

    lines = []
    for t in rows:
        status_emoji = {"todo": "⏳", "doing": "🔄", "done": "✅"}.get(t.status, "•")
        lines.append(
            f"{status_emoji} [{t.status.upper()}] {t.title} (priority: {t.priority})"
        )

    return "Today's summary:\n" + "\n".join(lines)


@task_agent.tool_plain
def list_tasks(status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """
    List tasks, optionally filtered by status (todo, doing, done).
    Returns a list of dicts with basic fields.
    """
    return _list_tasks(status=status, limit=limit)


@task_agent.tool_plain
def create_task(
    title: str,
    description: str | None = None,
    priority: str = "medium",
    due_date: str | None = None,
) -> dict[str, Any]:
    """
    Create a new task.
    - title: short title (required)
    - description: optional details
    - priority: low, medium, high (default: medium)
    - due_date: ISO 8601 datetime string (optional), e.g. '2026-08-25T18:00:00Z'
    Returns the created task as a dict.
    """
    return _create_task(
        title=title,
        description=description,
        priority=priority,
        due_date=due_date,
    )


@task_agent.tool_plain
def update_task_status(task_id: int, status: str) -> dict[str, Any] | None:
    """
    Update task status (todo, doing, done).
    Returns the updated task as a dict, or None if not found.
    """
    return _update_task_status(task_id=task_id, status=status)


@task_agent.tool_plain
def get_task(task_id: int) -> dict[str, Any] | None:
    """
    Get a single task by ID.
    Returns the task as a dict, or None if not found.
    """
    return _get_task(task_id=task_id)


@task_agent.tool_plain
def search_tasks(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Search tasks by title/description (simple LIKE search).
    Returns a list of matching tasks as dicts.
    """
    return _search_tasks(query=query, limit=limit)


@task_agent.tool_plain
def summarize_my_day() -> str:
    """
    Generate a short text summary of today's tasks.
    Useful for the user to get a quick overview of the day.
    """
    return _summarize_my_day()
