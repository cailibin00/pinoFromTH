from __future__ import annotations

from dataclasses import dataclass

import torch


def fischer_burmeister(
    pressure: torch.Tensor,
    gamma: torch.Tensor,
    pressure_ref: float,
    epsilon: float,
) -> torch.Tensor:
    pressure_bar = pressure / pressure_ref
    return pressure_bar + gamma - torch.sqrt(
        pressure_bar.square() + gamma.square() + epsilon**2
    )


@dataclass
class ALUpdate:
    violation_max: float
    violation_mean: float
    mu: float
    mu_increased: bool


class AugmentedLagrangian:
    def __init__(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
        mu_initial: float,
        mu_growth: float,
        mu_max: float,
        progress_ratio: float,
    ):
        self.multipliers = torch.zeros(size, 1, device=device, dtype=dtype)
        self.mu = float(mu_initial)
        self.mu_growth = float(mu_growth)
        self.mu_max = float(mu_max)
        self.progress_ratio = float(progress_ratio)
        self.previous_violation: float | None = None

    def reset(self, mu_initial: float | None = None) -> None:
        self.multipliers.zero_()
        if mu_initial is not None:
            self.mu = float(mu_initial)
        self.previous_violation = None

    def loss(self, indices: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
        multipliers = self.multipliers[indices]
        return torch.mean(multipliers * constraint) + 0.5 * self.mu * torch.mean(
            constraint.square()
        )

    def update(self, constraint_all: torch.Tensor) -> ALUpdate:
        detached = constraint_all.detach()
        violation_max = float(detached.abs().max().cpu())
        violation_mean = float(detached.abs().mean().cpu())
        self.multipliers.add_(self.mu * detached)
        increased = False
        if (
            self.previous_violation is not None
            and violation_max > self.progress_ratio * self.previous_violation
            and self.mu < self.mu_max
        ):
            self.mu = min(self.mu * self.mu_growth, self.mu_max)
            increased = True
        self.previous_violation = violation_max
        return ALUpdate(violation_max, violation_mean, self.mu, increased)

    def state_dict(self) -> dict[str, object]:
        return {
            "multipliers": self.multipliers.detach().cpu(),
            "mu": self.mu,
            "previous_violation": self.previous_violation,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        values = state.get("multipliers")
        if isinstance(values, torch.Tensor) and values.shape == self.multipliers.shape:
            self.multipliers.copy_(values.to(self.multipliers))
        self.mu = float(state.get("mu", self.mu))
        previous = state.get("previous_violation")
        self.previous_violation = None if previous is None else float(previous)
