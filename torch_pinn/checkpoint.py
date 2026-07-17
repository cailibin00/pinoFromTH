from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .geometry import compute_physical_params
from .networks import CVALModel


def save_checkpoint(
    path: Path,
    model: CVALModel,
    config: ExperimentConfig,
    stage_index: int,
    stage_name: str,
    epoch: int,
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
    al_state: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "architecture": "cv_al_amin",
        "model_state": model.state_dict(),
        "config": config.to_dict(),
        "stage_index": stage_index,
        "stage_name": stage_name,
        "epoch": epoch,
        "metrics": metrics,
        "gamma_enabled": model.gamma_enabled,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if al_state is not None:
        payload["al_state"] = al_state
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[CVALModel, ExperimentConfig, dict[str, Any]]:
    target = torch.device(device)
    payload = torch.load(Path(path), map_location=target)
    config = ExperimentConfig.from_dict(payload["config"])
    dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }.get(config.dtype)
    if dtype is None:
        raise ValueError(f"Unsupported checkpoint dtype: {config.dtype}")
    params = compute_physical_params(config)
    model = CVALModel(config, params).to(device=target, dtype=dtype)
    model.load_state_dict(payload["model_state"])
    model.set_gamma_enabled(bool(payload.get("gamma_enabled", True)))
    model.eval()
    return model, config, payload
