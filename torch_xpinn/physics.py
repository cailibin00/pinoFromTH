from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import HardGrooveGeometry, Region
from .networks import XPINNModel


@dataclass
class RegionValues:
    pressure: torch.Tensor
    gamma: torch.Tensor
    residual: torch.Tensor


def _gradient(
    values: torch.Tensor,
    coords: torch.Tensor,
    *,
    create_graph: bool,
    retain_graph: bool = True,
) -> torch.Tensor:
    return torch.autograd.grad(
        values,
        coords,
        grad_outputs=torch.ones_like(values),
        create_graph=create_graph,
        retain_graph=retain_graph,
    )[0]


def evaluate_region_residual(
    model: XPINNModel,
    geometry: HardGrooveGeometry,
    coords: torch.Tensor,
    region: Region,
) -> RegionValues:
    points = coords.detach().clone().requires_grad_(True)
    pressure, gamma = model.forward_region(points, region)
    pressure_grad = _gradient(pressure, points, create_graph=True)
    pressure_radius = pressure_grad[:, 0:1]
    pressure_theta = pressure_grad[:, 1:2]
    pressure_rr = _gradient(pressure_radius, points, create_graph=True)[:, 0:1]
    pressure_tt = _gradient(pressure_theta, points, create_graph=True)[:, 1:2]
    gamma_theta = _gradient(gamma, points, create_graph=True)[:, 1:2]

    radius = points[:, 0:1]
    film = geometry.region_film(region)
    residual = film**3 * (
        pressure_rr + pressure_radius / radius + pressure_tt / radius.square()
    ) + geometry.params.lambda_value * film * gamma_theta

    return RegionValues(pressure, gamma, residual)


def region_reynolds_loss(
    model: XPINNModel,
    geometry: HardGrooveGeometry,
    coords: torch.Tensor,
    region: Region,
) -> torch.Tensor:
    return torch.mean(
        evaluate_region_residual(model, geometry, coords, region).residual.square()
    )


def region_jfo_loss(
    model: XPINNModel,
    coords: torch.Tensor,
    region: Region,
) -> torch.Tensor:
    pressure, gamma = model.forward_region(coords, region)
    return torch.mean((pressure * gamma).square())
