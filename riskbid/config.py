from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping")
    required = ("dataset", "data", "risk", "constraints", "optimizer", "evaluation")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    config = deepcopy(config)
    config["_config_path"] = str(source)
    if "root" not in config["data"]:
        config["data"]["root"] = str(source.parent.parent.parent)
    if "seeds" not in config["evaluation"]:
        raise ValueError("evaluation.seeds must be provided")
    if not config["evaluation"]["seeds"]:
        raise ValueError("evaluation.seeds must be nonempty")
    risk = config["risk"]
    if not (0 < float(risk["alpha"]) < 0.5):
        raise ValueError("risk.alpha must be in (0, 0.5)")
    if float(risk["kappa"]) <= 0:
        raise ValueError("risk.kappa must be positive")
    optimizer = config["optimizer"]
    if optimizer["method"] != "riskbid":
        raise ValueError("optimizer.method must be riskbid")
    if optimizer["mode"] not in ("direct", "smoothed"):
        raise ValueError("optimizer.mode must be direct or smoothed")
    if not (0 < float(optimizer["safety_factor"]) < 1):
        raise ValueError("optimizer.safety_factor must be in (0, 1)")
    if float(optimizer["query_radius"]) <= 0:
        raise ValueError("optimizer.query_radius must be positive")
    if float(optimizer["smoothing_radius"]) < 0:
        raise ValueError("optimizer.smoothing_radius cannot be negative")
    if optimizer["mode"] == "smoothed" and int(optimizer["smoothing_samples"]) <= 0:
        raise ValueError("Smoothed mode requires positive smoothing_samples")
    if optimizer["mode"] == "direct" and int(optimizer["smoothing_samples"]) != 0:
        raise ValueError("Direct mode requires zero smoothing_samples")
    if int(optimizer["max_evaluations"]) <= 0 or int(optimizer["max_iterations"]) <= 0:
        raise ValueError("Optimization budgets must be positive")
    return config


def resolve_path(config: dict[str, Any], path: str | Path) -> Path:
    item = Path(path).expanduser()
    return item if item.is_absolute() else Path(config["data"]["root"]) / item
