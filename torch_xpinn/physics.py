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


def evaluate_region_flux(
    model: XPINNModel,
    geometry: HardGrooveGeometry,
    coords: torch.Tensor,
    region: Region,
    create_graph: bool,
) -> FluxValues:
    points = coords.detach().clone().requires_grad_(True)
    pressure, gamma = model.forward_region(points, region)
    gradient = _gradient(pressure, points, create_graph=create_graph)
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

    radial_scale = (geometry.params.r_max - geometry.params.r_min) ** -2
    theta_scale = (geometry.params.theta_max - geometry.params.theta_min) ** -2
    scale = (
        film**3 * (radial_scale + theta_scale) * geometry.cfg.pressure_ref
        + abs(geometry.params.lambda_value) * film
        + 1.0
    )
    return RegionValues(pressure, gamma, residual / scale)


def region_pde_loss(
    model: XPINNModel,
    geometry: HardGrooveGeometry,
    coords: torch.Tensor,
    region: Region,
) -> torch.Tensor:
    return torch.mean(
        evaluate_region_residual(model, geometry, coords, region).residual.square()
    )


def fischer_burmeister_loss(
    pressure: torch.Tensor,
    gamma: torch.Tensor,
    *,
    pressure_ref: float,
    epsilon: float,
) -> torch.Tensor:
    scaled_pressure = pressure / pressure_ref
    fb = (
        torch.sqrt(scaled_pressure.square() + gamma.square() + epsilon)
        - scaled_pressure
        - gamma
    )
    return torch.mean(fb.square())


def region_fb_loss(
    model: XPINNModel,
    geometry: HardGrooveGeometry,
    coords: torch.Tensor,
    region: Region,
) -> torch.Tensor:
    pressure, gamma = model.forward_region(coords, region)
    return fischer_burmeister_loss(
        pressure,
        gamma,
        pressure_ref=geometry.cfg.pressure_ref,
        epsilon=geometry.cfg.fb_epsilon,
    )


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
