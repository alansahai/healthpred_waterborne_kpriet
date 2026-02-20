"""Runtime configuration loader for operational monitoring mode."""

import json
import os
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "data_path": "data/coimbatore_weekly_water_disease_2024.csv",
    "model_path": "model/outbreak_model.pkl",
    "threshold_override": None,
    "retraining_frequency_days": 30,
    "forward_prediction_days": 7,
    "enable_14_day_projection": True,
    "artifact_version": "2.1.0-operational",
    "display_low_cutoff": 0.15,
    "display_high_cutoff": 0.30,
    "safe_state_message": "No outbreak risk above configured threshold for the next 7 days.",
    "api_readiness_note": "System ready for integration with real-time data APIs.",
}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def _config_path() -> str:
    return os.path.join(_project_root(), "config", "system_config.json")


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_runtime_config() -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    config_file = _config_path()

    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as handle:
            file_cfg = json.load(handle)
        config.update(file_cfg or {})

    if os.getenv("HP_DATA_PATH"):
        config["data_path"] = os.getenv("HP_DATA_PATH")
    if os.getenv("HP_MODEL_PATH"):
        config["model_path"] = os.getenv("HP_MODEL_PATH")
    if os.getenv("HP_THRESHOLD_OVERRIDE"):
        config["threshold_override"] = float(os.getenv("HP_THRESHOLD_OVERRIDE"))
    if os.getenv("HP_RETRAIN_FREQUENCY_DAYS"):
        config["retraining_frequency_days"] = int(os.getenv("HP_RETRAIN_FREQUENCY_DAYS"))
    if os.getenv("HP_ENABLE_14_DAY_PROJECTION"):
        config["enable_14_day_projection"] = _to_bool(os.getenv("HP_ENABLE_14_DAY_PROJECTION"))
    if os.getenv("HP_DISPLAY_LOW_CUTOFF"):
        config["display_low_cutoff"] = float(os.getenv("HP_DISPLAY_LOW_CUTOFF"))
    if os.getenv("HP_DISPLAY_HIGH_CUTOFF"):
        config["display_high_cutoff"] = float(os.getenv("HP_DISPLAY_HIGH_CUTOFF"))

    return config
