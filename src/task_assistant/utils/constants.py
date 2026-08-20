import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "task-assistant")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./task_assistant.db")