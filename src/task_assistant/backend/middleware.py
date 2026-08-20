# src/task_assistant/backend/middleware.py
from __future__ import annotations

import asyncio
import time

import mlflow
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from task_assistant.utils.constants import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

# Paths that generate too much noise to log individually
_SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}


class MLflowLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        # Run blocking MLflow I/O off the event loop
        await asyncio.to_thread(
            _log_to_mlflow,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency=elapsed,
            client_ip=request.client.host if request.client else "unknown",
        )
        return response


def _log_to_mlflow(method: str, path: str, status_code: int, latency: float, client_ip: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{method} {path}"):
        mlflow.log_metrics({"status_code": status_code, "latency_seconds": latency, "is_error": int(status_code >= 400)})
        mlflow.log_params({"endpoint": path, "method": method})
        mlflow.set_tags({"environment": "dev", "client_ip": client_ip, "endpoint": path, "method": method})


def register_prompts(template: str | None = None, commit_message: str = "Update system prompt") -> None:
    """Register or version the system prompt in the MLflow Prompt Registry."""
    from task_assistant.backend.agents import SYSTEM_PROMPT

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Skip registration if the latest version already has the same template.
    existing = mlflow.genai.load_prompt(
        "prompts:/task-assistant-system@latest",
        allow_missing=True,
        link_to_model=False,
    )
    target = template or SYSTEM_PROMPT
    if existing is not None and existing.template == target:
        return

    mlflow.genai.register_prompt(
        name="task-assistant-system",
        template=target,
        commit_message=commit_message,
        tags={"role": "system"},
    )
