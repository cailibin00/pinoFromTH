from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import HardGrooveGeometry, Region
from .networks import XPINNModel


@dataclass
class FluxValues:
    pressure: torch.Tensor
    gamma: torch.Tensor
    q_radius: torch.Tensor
    q_theta: torch.Tensor


def evaluate_region_flux(
    model: XPINNModel,
    geometry: HardGrooveGeometry,
    coords: torch.Tensor,
    region: Region,
    create_graph: bool,
) -> FluxValues:
    points = coords.detach().clone().requires_grad_(True)
    pressure, gamma = model.forward_region(points, region)
    gradient = torch.autograd.grad(
        pressure,
        points,
        grad_outputs=torch.ones_like(pressure),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    pressure_radius = gradient[:, 0:1]
    pressure_theta = gradient[:, 1:2]
    radius = points[:, 0:1]
    film = geometry.region_film(region)
    q_radius = radius * film**3 * pressure_radius
    q_theta = (
        film**3 / radius * pressure_theta
        - geometry.params.lambda_value * radius * (1.0 - gamma) * film
    )
    return FluxValues(pressure, gamma, q_radius, q_theta)


def normal_flux(flux: FluxValues, normals: torch.Tensor) -> torch.Tensor:
    return flux.q_radius * normals[:, 0:1] + flux.q_theta * normals[:, 1:2]


def interface_losses(
    model: XPINNModel,
    geometry: HardGrooveGeometry,
    points: torch.Tensor,
    normals: torch.Tensor,
    create_graph: bool = True,
) -> dict[str, torch.Tensor]:
    thin = evaluate_region_flux(model, geometry, points, Region.THIN, create_graph)
    groove = evaluate_region_flux(model, geometry, points, Region.GROOVE, create_graph)
    return {
        "interface_pressure": torch.mean((thin.pressure - groove.pressure).square()),
        "interface_flux": torch.mean(
            (normal_flux(thin, normals) - normal_flux(groove, normals)).square()
        ),
    }
