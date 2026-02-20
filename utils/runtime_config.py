"""Runtime configuration loader for operational monitoring mode."""

import json
import os
from typing import Any, Dict

from .constants import GLOBAL_THRESHOLD

DEFAULT_CONFIG: Dict[str, Any] = {
    "include_spatial_features": False,
    "retraining_frequency_days": 30,
    "forward_prediction_days": 7,
    "enable_14_day_projection": True,
    "artifact_version": "3.0-final",
    "safe_state_message": "No outbreak risk above configured threshold for the next 7 days.",
    "api_readiness_note": "System ready for integration with real-time data APIs.",
}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def _config_path() -> str:
    return os.path.join(_project_root(), "config", "system_config.json")


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _required_config_keys() -> tuple[str, ...]:
    return ("model_artifact_path", "data_path", "global_threshold")


def load_runtime_config() -> Dict[str, Any]:
    config_file = _config_path()

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Runtime config not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as handle:
        file_cfg = json.load(handle)

    if not isinstance(file_cfg, dict):
        raise ValueError("Runtime config must be a JSON object.")

    config = DEFAULT_CONFIG.copy()
    config.update(file_cfg)

    missing = [key for key in _required_config_keys() if key not in config]
    if missing:
        raise KeyError(f"Runtime config missing required keys: {', '.join(missing)}")

    for path_key in ("model_artifact_path", "data_path"):
        if not str(config.get(path_key, "")).strip():
            raise ValueError(f"Runtime config key '{path_key}' must be a non-empty string.")

    try:
        configured_threshold = float(config["global_threshold"])
    except (TypeError, ValueError) as error:
        raise ValueError("Runtime config key 'global_threshold' must be numeric.") from error

    if abs(configured_threshold - float(GLOBAL_THRESHOLD)) > 1e-9:
        raise RuntimeError(
            f"Threshold policy drift detected in runtime config: "
            f"configured={configured_threshold:.6f}, expected={float(GLOBAL_THRESHOLD):.6f}."
        )

    config["global_threshold"] = configured_threshold

    if os.getenv("HP_RETRAIN_FREQUENCY_DAYS"):
        config["retraining_frequency_days"] = int(os.getenv("HP_RETRAIN_FREQUENCY_DAYS"))
    if os.getenv("HP_ENABLE_14_DAY_PROJECTION"):
        config["enable_14_day_projection"] = _to_bool(os.getenv("HP_ENABLE_14_DAY_PROJECTION"))

    return config
