from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .config import ExperimentConfig
from .control_volume import FaceBatch
from .geometry import FilmGeometry, PhysicalParams
from .networks import CVALModel


@dataclass(frozen=True)
class FluxScales:
    q_radius: float
    q_theta: float
    q_theta_pressure: float
    q_theta_motion: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_flux_scales(
    cfg: ExperimentConfig, params: PhysicalParams
) -> FluxScales:
    delta_r = params.r_max - params.r_min
    delta_theta = params.theta_max - params.theta_min
    radius_ref = 0.5 * (params.r_min + params.r_max)
    pressure_part = (
        cfg.film_ref**3
        * cfg.pressure_ref
        / (radius_ref * delta_theta)
    )
    motion_part = params.lambda_value * radius_ref * cfg.film_ref
    return FluxScales(
        q_radius=(
            radius_ref * cfg.film_ref**3 * cfg.pressure_ref / delta_r
        ),
        q_theta=pressure_part + motion_part,
        q_theta_pressure=pressure_part,
        q_theta_motion=motion_part,
    )


@dataclass
class FluxValues:
    pressure: torch.Tensor
    gamma: torch.Tensor
    film: torch.Tensor
    pressure_radius: torch.Tensor
    pressure_theta: torch.Tensor
    q_radius: torch.Tensor
    q_theta: torch.Tensor


def evaluate_flux(
    model: CVALModel,
    geometry: FilmGeometry,
    coords: torch.Tensor,
    lambda_value: float,
    xi_ratio: float,
    create_graph: bool,
) -> FluxValues:
    points = coords.detach().clone().requires_grad_(True)
    pressure, gamma = model.output_fields(points)
    pressure_grad = torch.autograd.grad(
        pressure,
        points,
        grad_outputs=torch.ones_like(pressure),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    pressure_radius = pressure_grad[:, 0:1]
    pressure_theta = pressure_grad[:, 1:2]
    # H enters algebraically. Detaching it guarantees that no H derivative is
    # accidentally reintroduced into the control-volume objective.
    with torch.no_grad():
        film = geometry.film_thickness(points.detach(), xi_ratio=xi_ratio)
    radius = points[:, 0:1]
    q_radius = radius * film**3 * pressure_radius
    q_theta = (
        film**3 / radius * pressure_theta
        - lambda_value * radius * (1.0 - gamma) * film
    )
    return FluxValues(
        pressure=pressure,
        gamma=gamma,
        film=film,
        pressure_radius=pressure_radius,
        pressure_theta=pressure_theta,
        q_radius=q_radius,
        q_theta=q_theta,
    )


@dataclass
class CellResidual:
    normalized: torch.Tensor
    raw_flux: torch.Tensor


def control_volume_residual(
    model: CVALModel,
    geometry: FilmGeometry,
    batch: FaceBatch,
    scales: FluxScales,
    lambda_value: float,
    xi_ratio: float,
    create_graph: bool,
) -> CellResidual:
    cell_count, gauss_order, _ = batch.right.shape
    points = torch.cat(
        [
            batch.right.reshape(-1, 2),
            batch.left.reshape(-1, 2),
            batch.top.reshape(-1, 2),
            batch.bottom.reshape(-1, 2),
        ],
        dim=0,
    )
    flux = evaluate_flux(
        model,
        geometry,
        points,
        lambda_value=lambda_value,
        xi_ratio=xi_ratio,
        create_graph=create_graph,
    )
    block = cell_count * gauss_order
    q_r_right = flux.q_radius[0:block].reshape(cell_count, gauss_order)
    q_r_left = flux.q_radius[block : 2 * block].reshape(cell_count, gauss_order)
    q_t_top = flux.q_theta[2 * block : 3 * block].reshape(cell_count, gauss_order)
    q_t_bottom = flux.q_theta[3 * block : 4 * block].reshape(
        cell_count, gauss_order
    )
    radial_net = torch.sum(
        (q_r_right - q_r_left) * batch.weights_theta, dim=1, keepdim=True
    )
    theta_net = torch.sum(
        (q_t_top - q_t_bottom) * batch.weights_radius, dim=1, keepdim=True
    )
    raw_flux = radial_net + theta_net
    cell_scale = (
        scales.q_radius * batch.delta_theta
        + scales.q_theta * batch.delta_radius
    )
    return CellResidual(normalized=raw_flux / cell_scale, raw_flux=raw_flux)
