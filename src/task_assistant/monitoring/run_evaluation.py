import json
import os
import mlflow
from mlflow.genai.datasets import create_dataset
from mlflow.genai import evaluate
from mlflow.genai.scorers import Correctness, Guidelines

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "task-assistant"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

BASELINE_DIR = "src/task_assistant/monitoring"

with open(f"{BASELINE_DIR}/evaluation_dataset.json") as f:
    eval_data = json.load(f)

dataset = create_dataset(
    name="task-assistant-eval-local",
    experiment_id=experiment.experiment_id,
    tags={"stage": "ci", "domain": "task-assistant"},
)
dataset.merge_records(eval_data)

import asyncio
from task_assistant.backend.agents import task_agent

def bot_answer(question: str) -> str:
    result = asyncio.run(task_agent.run(question))
    return result.output

# Usa Ollama local como judge (ou OpenAI se tiver key)
JUDGE_MODEL = "ollama:/llama3.1:latest"

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

results = evaluate(
    data=dataset,
    predict_fn=bot_answer,
    scorers=scorers,
)

# Salva scores
os.makedirs("src/task_assistant/monitoring/evaluation_results", exist_ok=True)

with open("src/task_assistant/monitoring/current_scores.json", "w") as f:
    json.dump(
        {
            "factual_accuracy_mean": results.metrics["factual_accuracy/mean"],
            "task_assistant_quality_mean": results.metrics["task_assistant_quality/mean"],
        },
        f,
        indent=2,
    )

# Salva resultados detalhados
with open("src/task_assistant/monitoring/evaluation_results/results.json", "w") as f:
    json.dump(results.metrics, f, indent=2)

print("✅ Evaluation completed")
print(f"Factual accuracy: {results.metrics['factual_accuracy/mean']:.3f}")
print(f"Task assistant quality: {results.metrics['task_assistant_quality/mean']:.3f}")
