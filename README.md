# Task Assistant — LLMOps PoC

A local AI task assistant built with **pydantic-ai**, **FastAPI**, **Streamlit**, and **MLflow**, running on **Ollama** (llama3.1).

## Architecture

```
Streamlit frontend  →  FastAPI backend  →  pydantic-ai agent (Ollama)
                                        ↓
                                   SQLite (tasks)
                                        ↓
                                 MLflow tracking server
                                 (traces · runs · prompt registry)
```

## Stack

| Layer | Tech |
|---|---|
| LLM | Ollama `llama3.1:latest` (OpenAI-compatible API) |
| Agent framework | pydantic-ai |
| Backend API | FastAPI + uvicorn |
| Frontend | Streamlit |
| Task storage | SQLite via SQLAlchemy |
| Observability | MLflow 3 (autolog traces, HTTP run metrics, prompt registry) |
| Evaluation | `mlflow.genai.evaluate` + LLM judges (Correctness, RelevanceToQuery, Safety) |

## Quickstart

```bash
# 1. Start Ollama
ollama serve

# 2. Start MLflow tracking server
uv run mlflow server --host 0.0.0.0 --port 5000

# 3. Start the API (port 3030)
uv run uvicorn task_assistant.backend.api:app --host 0.0.0.0 --port 3030

# 4. Start the frontend
API_BASE_URL=http://localhost:3030 uv run streamlit run src/task_assistant/frontend/app.py
```

## Features

- **Chat interface** — create, list, update, search, and summarize tasks via natural language
- **Tool-calling agent** — 6 structured tools backed by a persistent SQLite database
- **MLflow tracing** — every agent run is autologged as a trace
- **HTTP metrics middleware** — latency and status code logged per request
- **Prompt Registry** — system prompt versioned in MLflow; loaded at startup with fallback
- **Offline evaluation** — `monitoring/evaluation.ipynb` runs LLM-judge scoring on a curated dataset

## Project structure

```
src/task_assistant/
├── backend/
│   ├── agents.py       # pydantic-ai agent, tools, SQLAlchemy models
│   ├── api.py          # FastAPI app
│   └── middleware.py   # MLflow HTTP logging + prompt registry
├── frontend/
│   └── app.py          # Streamlit chat UI
├── monitoring/
│   └── evaluation.ipynb
└── utils/
    └── constants.py    # MLFLOW_TRACKING_URI, DATABASE_URL, etc.
```
