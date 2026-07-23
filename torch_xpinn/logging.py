from __future__ import annotations

from collections.abc import Mapping

import torch


LOSS_ORDER = [
    "reynolds",
    "jfo",
    "boundary",
]


def scalar(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def weighted_total(
    losses: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> torch.Tensor:
    total: torch.Tensor | None = None
    for key, value in losses.items():
        term = value * float(weights.get(key, 1.0))
        total = term if total is None else total + term
    if total is None:
        raise ValueError("Cannot compute a total from an empty loss dictionary")
    return total


def format_loss_block(
    *,
    stage: str,
    epoch: int,
    losses: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    learning_rate: float | None = None,
) -> str:
    total = scalar(weighted_total(losses, weights))
    header = f"[train] stage={stage} epoch={epoch:06d}"
    if learning_rate is not None:
        header += f" lr={learning_rate:.3e}"
    lines = [header, f"  total{'':<24s}= {total:.8e}"]
    ordered_keys = [key for key in LOSS_ORDER if key in losses]
    ordered_keys.extend(key for key in losses if key not in ordered_keys)
    for key in ordered_keys:
        raw = scalar(losses[key])
        weight = float(weights.get(key, 1.0))
        lines.append(
            f"  {key:<29s}= {raw:.8e}  w={weight:.3e}  contrib={raw * weight:.8e}"
        )
    return "\n".join(lines)
