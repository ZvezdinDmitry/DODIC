import os

import mlflow
import yaml
from ultralytics import YOLO, settings

CARPARTS_MODEL_CONFIG_PATH = "train_configs/carparts_model.yml"
DAMAGES_MODEL_CONFIG_PATH = "train_configs/damages_model.yml"


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_mlflow(mlflow_cfg: dict):
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    mlflow.autolog()

    settings.update({"mlflow": True})
    print(f"✅ MLflow setup succeded: '{mlflow_cfg['experiment_name']}'")


def train_model(config_path):
    config = load_config(config_path)
    setup_mlflow(config["mlflow"])

    model_cfg = config["model"]
    model = YOLO(model_cfg["checkpoint"])

    train_cfg = config["training"]
    results = model.train(
        data=model_cfg["data_yaml"],
        **train_cfg,
    )
    return results


def train_both_models():
    print("CAR PARTS MODEL TRAINING STARTED")
    parts_results = train_model(CARPARTS_MODEL_CONFIG_PATH)
    print("DAMAGES MODEL TRAINING STARTED")
    damages_results = train_model(DAMAGES_MODEL_CONFIG_PATH)


if __name__ == "__main__":
    train_both_models()
