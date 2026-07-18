from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ExperimentConfig
from .geometry import PhysicalParams


class FixedPeriodicFeatures(nn.Module):
    def __init__(self, params: PhysicalParams, modes: int):
        super().__init__()
        self.r_min = params.r_min
        self.r_max = params.r_max
        self.periods = params.periods
        self.modes = modes
        frequencies = torch.arange(1, modes + 1, dtype=torch.get_default_dtype())
        self.register_buffer("frequencies", frequencies)

    @property
    def output_dim(self) -> int:
        return 1 + 2 * self.modes

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        radius = coords[:, 0:1]
        theta = coords[:, 1:2]
        rho = 2.0 * (radius - self.r_min) / (self.r_max - self.r_min) - 1.0
        phase = theta * self.periods * self.frequencies.view(1, -1)
        return torch.cat([rho, torch.sin(phase), torch.cos(phase)], dim=1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, width: int, layers: int, output_dim: int):
        super().__init__()
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(layers):
            linear = nn.Linear(current, width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            modules.extend([linear, nn.SiLU()])
            current = width
        output = nn.Linear(current, output_dim)
        nn.init.xavier_normal_(output.weight, gain=0.1)
        nn.init.zeros_(output.bias)
        modules.append(output)
        self.layers = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _softplus_inverse(value: float, beta: float) -> float:
    return math.log(math.expm1(beta * value)) / beta


@dataclass
class OutputComponents:
    pressure: torch.Tensor
    gamma: torch.Tensor
    pressure_bar: torch.Tensor
    active_gate: torch.Tensor
    gamma_candidate: torch.Tensor


class CVALModel(nn.Module):
    """A-min model: independent pressure and cavitation networks."""

    def __init__(self, cfg: ExperimentConfig, params: PhysicalParams):
        super().__init__()
        self.features = FixedPeriodicFeatures(params, cfg.fourier_modes)
        self.pressure_net = MLP(
            self.features.output_dim,
            cfg.hidden_width,
            cfg.hidden_layers,
            1,
        )
        self.gamma_net = MLP(
            self.features.output_dim,
            cfg.hidden_width,
            cfg.hidden_layers,
            1,
        )
        self.r_min = params.r_min
        self.r_max = params.r_max
        self.beta = cfg.softplus_beta
        self.pressure_ref = float(cfg.pressure_ref)
        self.register_buffer(
            "pressure_latent_inner",
            torch.tensor(_softplus_inverse(params.pressure_inner, self.beta)),
        )
        self.register_buffer(
            "pressure_latent_outer",
            torch.tensor(_softplus_inverse(params.pressure_outer, self.beta)),
        )
        self.gamma_enabled = True
        self.active_gate_enabled = True
        self.active_pressure_ratio = 0.25
        self.active_gate_tau = 0.05

    def set_gamma_enabled(self, enabled: bool) -> None:
        self.gamma_enabled = enabled

    def configure_active_gate(
        self,
        enabled: bool,
        pressure_ratio: float,
        tau: float,
    ) -> None:
        self.active_gate_enabled = bool(enabled)
        self.active_pressure_ratio = float(pressure_ratio)
        self.active_gate_tau = max(float(tau), 1e-8)

    def active_gate_state(self) -> dict[str, bool | float]:
        return {
            "enabled": self.active_gate_enabled,
            "pressure_ratio": self.active_pressure_ratio,
            "tau": self.active_gate_tau,
        }

    def load_active_gate_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        self.configure_active_gate(
            bool(state.get("enabled", self.active_gate_enabled)),
            float(state.get("pressure_ratio", self.active_pressure_ratio)),
            float(state.get("tau", self.active_gate_tau)),
        )

    def output_components(self, coords: torch.Tensor) -> OutputComponents:
        features = self.features(coords)
        t = (coords[:, 0:1] - self.r_min) / (self.r_max - self.r_min)
        distance = 4.0 * t * (1.0 - t)
        pressure_latent_bc = (
            (1.0 - t) * self.pressure_latent_inner
            + t * self.pressure_latent_outer
        )
        pressure_latent = pressure_latent_bc + distance * self.pressure_net(features)
        pressure = F.softplus(pressure_latent, beta=self.beta)
        pressure_bar = pressure / self.pressure_ref
        if self.gamma_enabled:
            gamma_candidate = distance * torch.sigmoid(self.gamma_net(features))
            if self.active_gate_enabled:
                active_gate = torch.sigmoid(
                    (self.active_pressure_ratio - pressure_bar)
                    / self.active_gate_tau
                )
            else:
                active_gate = torch.ones_like(pressure)
            gamma = gamma_candidate * active_gate
        else:
            gamma_candidate = torch.zeros_like(pressure)
            active_gate = torch.zeros_like(pressure)
            gamma = torch.zeros_like(pressure)
        return OutputComponents(
            pressure=pressure,
            gamma=gamma,
            pressure_bar=pressure_bar,
            active_gate=active_gate,
            gamma_candidate=gamma_candidate,
        )

    def output_fields(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        components = self.output_components(coords)
        return components.pressure, components.gamma

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        pressure, gamma = self.output_fields(coords)
        return torch.cat([pressure, gamma], dim=1)

    def parameter_summary(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "pressure_net": count(self.pressure_net),
            "gamma_net": count(self.gamma_net),
            "total": count(self),
        }
