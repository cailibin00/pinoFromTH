from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch

from .config import XPINNConfig


class Region(str, Enum):
    THIN = "thin"
    GROOVE = "groove"


@dataclass(frozen=True)
class PhysicalParams:
    lambda_value: float
    pressure_inner: float
    pressure_outer: float
    r_min: float
    r_max: float
    theta_min: float
    theta_max: float
    groove_r_min: float
    groove_r_max: float
    spiral_origin_r: float
    spiral_angle_rad: float
    groove_width_ratio: float
    periods: int


def compute_physical_params(cfg: XPINNConfig) -> PhysicalParams:
    pressure_outer_pa = cfg.pressure_outer_ratio * cfg.pressure_inner_pa
    pressure_base = 10.0 * pressure_outer_pa
    omega = cfg.speed_rpm * 2.0 * math.pi / 60.0
    lambda_value = (
        6.0
        * cfg.viscosity_pa_s
        * omega
        * cfg.r_o**2
        / (cfg.h_i**2 * pressure_base)
    )
    r_min = cfg.r_i / cfg.r_o
    r_max = 1.0
    groove_r_min = cfg.groove_start_ratio * cfg.r_i / cfg.r_o
    groove_r_max = min(cfg.groove_end_ratio * cfg.r_i / cfg.r_o, r_max)
    return PhysicalParams(
        lambda_value=lambda_value,
        pressure_inner=cfg.pressure_inner_pa / pressure_base,
        pressure_outer=pressure_outer_pa / pressure_base,
        r_min=r_min,
        r_max=r_max,
        theta_min=0.0,
        theta_max=2.0 * math.pi / cfg.periods,
        groove_r_min=groove_r_min,
        groove_r_max=groove_r_max,
        spiral_origin_r=groove_r_min,
        spiral_angle_rad=math.radians(cfg.spiral_angle_deg),
        groove_width_ratio=cfg.groove_width_ratio,
        periods=cfg.periods,
    )


class HardGrooveGeometry:
    """Hard H=1/H=4 spiral-groove partition, with no sigmoid transition."""

    def __init__(self, cfg: XPINNConfig):
        self.cfg = cfg
        self.params = compute_physical_params(cfg)
        self.period = self.params.theta_max - self.params.theta_min
        self.theta_offset = math.pi / 6.0
        self._tan_alpha = math.tan(self.params.spiral_angle_rad)

    def spiral_theta(self, radius: torch.Tensor) -> torch.Tensor:
        return (
            torch.log(radius / self.params.spiral_origin_r) / self._tan_alpha
            + self.theta_offset
        )

    def local_theta(self, coords: torch.Tensor) -> torch.Tensor:
        radius = coords[:, 0:1]
        theta = coords[:, 1:2]
        phase = torch.remainder(theta - self.spiral_theta(radius), self.period)
        return phase

    def groove_mask(self, coords: torch.Tensor) -> torch.Tensor:
        radius = coords[:, 0:1]
        phase = self.local_theta(coords)
        radial = (radius >= self.params.groove_r_min) & (
            radius <= self.params.groove_r_max
        )
        angular = phase <= self.params.groove_width_ratio * self.period
        return radial & angular

    def region_film(self, region: Region) -> float:
        return self.cfg.groove_film if region is Region.GROOVE else self.cfg.thin_film

    def film_thickness(self, coords: torch.Tensor) -> torch.Tensor:
        mask = self.groove_mask(coords)
        return torch.where(
            mask,
            torch.full_like(coords[:, 0:1], self.cfg.groove_film),
            torch.full_like(coords[:, 0:1], self.cfg.thin_film),
        )

    def sample_uniform(
        self,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        radius = self.params.r_min + (
            self.params.r_max - self.params.r_min
        ) * torch.rand((count, 1), device=device, dtype=dtype, generator=generator)
        theta = self.params.theta_min + self.period * torch.rand(
            (count, 1), device=device, dtype=dtype, generator=generator
        )
        return torch.cat([radius, theta], dim=1)

    def sample_region(
        self,
        region: Region,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if count <= 0:
            return torch.empty((0, 2), device=device, dtype=dtype)

        chunks: list[torch.Tensor] = []
        accepted = 0
        attempts = 0
        draw_count = max(256, count * 3)
        want_groove = region is Region.GROOVE
        while accepted < count and attempts < 128:
            candidates = self.sample_uniform(draw_count, device, dtype, generator)
            mask = self.groove_mask(candidates).squeeze(1)
            selected = candidates[mask if want_groove else ~mask]
            if selected.numel() > 0:
                chunks.append(selected)
                accepted += selected.shape[0]
            attempts += 1
            remaining = count - accepted
            draw_count = max(256, remaining * 3)

        if accepted < count:
            raise RuntimeError(
                f"Could not sample {count} points for region={region.value}; "
                f"only accepted {accepted}."
            )
        return torch.cat(chunks, dim=0)[:count]

    def radial_boundary_points(
        self,
        radius_value: float,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        radius = torch.full((count, 1), radius_value, device=device, dtype=dtype)
        theta = self.params.theta_min + self.period * torch.rand(
            (count, 1), device=device, dtype=dtype, generator=generator
        )
        return torch.cat([radius, theta], dim=1)

    def periodic_pairs(
        self,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        radius = self.params.r_min + (
            self.params.r_max - self.params.r_min
        ) * torch.rand((count, 1), device=device, dtype=dtype, generator=generator)
        left = torch.cat(
            [
                radius,
                torch.full((count, 1), self.params.theta_min, device=device, dtype=dtype),
            ],
            dim=1,
        )
        right = torch.cat(
            [
                radius,
                torch.full((count, 1), self.params.theta_max, device=device, dtype=dtype),
            ],
            dim=1,
        )
        return left, right

    def interface_points(
        self,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        half = max(1, count // 2)
        radius = torch.linspace(
            self.params.groove_r_min,
            self.params.groove_r_max,
            half,
            device=device,
            dtype=dtype,
        ).reshape(-1, 1)
        theta_entry = torch.remainder(self.spiral_theta(radius), self.period)
        theta_exit = torch.remainder(
            self.spiral_theta(radius)
            + self.params.groove_width_ratio * self.period,
            self.period,
        )
        entry = torch.cat([radius, theta_entry], dim=1)
        exit_ = torch.cat([radius, theta_exit], dim=1)
        points = torch.cat([entry, exit_], dim=0)
        normals = torch.cat(
            [
                self.interface_normal(radius, sign=1.0),
                self.interface_normal(radius, sign=-1.0),
            ],
            dim=0,
        )
        return points, normals

    def interface_normal(self, radius: torch.Tensor, sign: float) -> torch.Tensor:
        dtheta_dr = 1.0 / (radius * self._tan_alpha)
        grad_r = -dtheta_dr
        grad_theta = torch.ones_like(radius)
        normal = torch.cat([grad_r, grad_theta], dim=1)
        normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
        return normal * sign
