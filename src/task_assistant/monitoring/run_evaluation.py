import asyncio
import json
import os

import mlflow
from mlflow.genai import evaluate
from mlflow.genai.scorers import Correctness, Guidelines

from task_assistant.backend.agents import task_agent, init_db

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "task-assistant"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

init_db()

BASELINE_DIR = "src/task_assistant/monitoring"

with open(f"{BASELINE_DIR}/evaluation_dataset.json") as f:
    eval_data = json.load(f)


def bot_answer(question: str) -> str:
    result = asyncio.run(task_agent.run(question))
    return result.output


# LiteLLM uses OPENAI_API_BASE (not OPENAI_BASE_URL which is the OpenAI SDK var)
JUDGE_MODEL = "openai/llama3.1:latest"
os.environ.setdefault("OPENAI_API_KEY", "ollama")
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:11434/v1")

scorers = [
    Correctness(name="factual_accuracy", model=JUDGE_MODEL),
    Guidelines(
        name="task_assistant_quality",
        model=JUDGE_MODEL,
        guidelines=(
            "The response must be helpful, direct, and focused on task management. "
            "It must confirm actions taken (create, update, list, summarize) clearly. "
            "It must refuse or redirect requests unrelated to work tasks."
        ),
    ),
]

mlflow.set_experiment("task-assistant-evaluation-local")

# eval_data is passed directly — create_dataset requires Databricks
results = evaluate(
    data=eval_data,
    predict_fn=bot_answer,
    scorers=scorers,
)

os.makedirs("src/task_assistant/monitoring/evaluation_results", exist_ok=True)

print(f"Available metrics: {list(results.metrics.keys())}")

factual_accuracy = results.metrics.get("factual_accuracy/mean", -1.0)
task_quality = results.metrics.get("task_assistant_quality/mean", -1.0)

if factual_accuracy < 0 or task_quality < 0:
    print("⚠️  One or more LLM judges failed — scores set to -1. Check judge model connectivity.")

with open("src/task_assistant/monitoring/current_scores.json", "w") as f:
    json.dump(
        {
            "factual_accuracy_mean": factual_accuracy,
            "task_assistant_quality_mean": task_quality,
        },
        f,
        indent=2,
    )

with open("src/task_assistant/monitoring/evaluation_results/results.json", "w") as f:
    json.dump(results.metrics, f, indent=2)

print("✅ Evaluation completed")
print(f"Factual accuracy: {factual_accuracy:.3f}")
print(f"Task assistant quality: {task_quality:.3f}")

if factual_accuracy < 0 or task_quality < 0:
    raise SystemExit("Scorer failures detected — marking run as failed.")
