from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from .config import ExperimentConfig


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
    groove_depth_ratio: float
    groove_width_ratio: float
    periods: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_physical_params(cfg: ExperimentConfig) -> PhysicalParams:
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
    groove_r_max = cfg.groove_end_ratio * cfg.r_i / cfg.r_o
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
        groove_depth_ratio=cfg.groove_depth_ratio,
        groove_width_ratio=cfg.groove_width_ratio,
        periods=cfg.periods,
    )


class FilmGeometry:
    """Known spiral-groove film thickness; it is never a trainable field."""

    def __init__(self, params: PhysicalParams):
        self.params = params
        self.base_xi_r = (params.r_max - params.r_min) / 100.0
        self.base_xi_theta = (params.theta_max - params.theta_min) / 50.0
        self.theta_offset = math.pi / 6.0
        self._tan_alpha = math.tan(params.spiral_angle_rad)

    def spiral_theta(self, radius: torch.Tensor) -> torch.Tensor:
        return (
            torch.log(radius / self.params.spiral_origin_r) / self._tan_alpha
            + self.theta_offset
        )

    def film_thickness(
        self, coords: torch.Tensor, xi_ratio: float = 1.0
    ) -> torch.Tensor:
        radius = coords[:, 0:1]
        theta = coords[:, 1:2]
        xi_r = self.base_xi_r * xi_ratio
        xi_theta = self.base_xi_theta * xi_ratio
        spiral = self.spiral_theta(radius)
        period = 2.0 * math.pi / self.params.periods
        offsets = [0.0, -period, -2.0 * period, period, 2.0 * period]
        windows = []
        for offset in offsets:
            enter = torch.sigmoid((theta - spiral + offset) / xi_theta)
            leave = torch.sigmoid(
                (spiral - theta + period * self.params.groove_width_ratio - offset)
                / xi_theta
            )
            windows.append(enter * leave)
        theta_window = torch.stack(windows, dim=0).sum(dim=0)
        radial_window = torch.sigmoid(
            (radius - self.params.groove_r_min) / xi_r
        ) * torch.sigmoid((self.params.groove_r_max - radius) / xi_r)
        texture = radial_window * theta_window
        return 1.0 + self.params.groove_depth_ratio * texture
