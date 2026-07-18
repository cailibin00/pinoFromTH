from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from torch_pinn.config import ExperimentConfig
from torch_pinn.control_volume import ControlVolumeGrid
from torch_pinn.geometry import compute_physical_params
from torch_pinn.networks import CVALModel
from torch_pinn.physics import FluxScales, control_volume_residual, evaluate_flux


class ConstantFilm:
    def film_thickness(self, coords: torch.Tensor, xi_ratio: float = 1.0) -> torch.Tensor:
        return torch.ones_like(coords[:, 0:1])


class LogPressure(nn.Module):
    def output_fields(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pressure = torch.log(coords[:, 0:1])
        return pressure, torch.zeros_like(pressure)


class AnalyticPressure(nn.Module):
    def output_fields(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        radius = coords[:, 0:1]
        theta = coords[:, 1:2]
        pressure = 0.5 * radius.square() + torch.cos(6.0 * theta)
        return pressure, torch.full_like(pressure, 0.2)


class TorchCVALTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self.cfg = ExperimentConfig.smoke()
        self.params = compute_physical_params(self.cfg)

    def test_model_enforces_radial_bc_and_periodicity(self) -> None:
        model = CVALModel(self.cfg, self.params)
        theta = torch.linspace(self.params.theta_min, self.params.theta_max, 31).reshape(-1, 1)
        inner = torch.cat([torch.full_like(theta, self.params.r_min), theta], dim=1)
        outer = torch.cat([torch.full_like(theta, self.params.r_max), theta], dim=1)
        self.assertLess(float((model(inner)[:, 0] - self.params.pressure_inner).abs().max()), 1e-13)
        self.assertLess(float((model(outer)[:, 0] - self.params.pressure_outer).abs().max()), 1e-13)
        self.assertEqual(float(model(inner)[:, 1].abs().max()), 0.0)
        radius = torch.linspace(self.params.r_min, self.params.r_max, 31).reshape(-1, 1)
        start = torch.cat([radius, torch.zeros_like(radius)], dim=1)
        end = torch.cat([radius, torch.full_like(radius, self.params.theta_max)], dim=1)
        self.assertLess(float((model(start) - model(end)).abs().max()), 1e-12)

    def test_active_gate_suppresses_gamma_away_from_low_pressure(self) -> None:
        model = CVALModel(self.cfg, self.params)
        model.configure_active_gate(True, pressure_ratio=0.25, tau=0.05)
        coords = torch.tensor(
            [[0.5 * (self.params.r_min + self.params.r_max), 0.3]],
            dtype=torch.float64,
        )
        components = model.output_components(coords)
        self.assertGreater(float(components.gamma_candidate.max()), 0.1)
        self.assertLess(float(components.active_gate.max()), 1e-3)
        self.assertLess(float(components.gamma.max()), 1e-3)

    def test_active_gate_can_open_gamma_channel(self) -> None:
        model = CVALModel(self.cfg, self.params)
        model.configure_active_gate(True, pressure_ratio=2.0, tau=0.05)
        coords = torch.tensor(
            [[0.5 * (self.params.r_min + self.params.r_max), 0.3]],
            dtype=torch.float64,
        )
        components = model.output_components(coords)
        self.assertGreater(float(components.active_gate.min()), 0.99)
        self.assertGreater(float(components.gamma.min()), 0.1)

    def test_direct_flux_signs_and_factors(self) -> None:
        coords = torch.tensor([[0.95, 0.13], [0.98, 0.31]])
        values = evaluate_flux(
            AnalyticPressure(), ConstantFilm(), coords, lambda_value=2.5,
            xi_ratio=1.0, create_graph=False,
        )
        radius, theta = coords[:, 0:1], coords[:, 1:2]
        expected_qr = radius.square()
        expected_qt = -6.0 * torch.sin(6.0 * theta) / radius - 2.5 * radius * 0.8
        self.assertTrue(torch.allclose(values.q_radius, expected_qr, atol=1e-12))
        self.assertTrue(torch.allclose(values.q_theta, expected_qt, atol=1e-12))

    def test_divergence_free_manufactured_flux_has_zero_cv_residual(self) -> None:
        grid = ControlVolumeGrid(self.params, cells_r=4, cells_theta=5, gauss_order=3)
        batch = grid.face_batch(grid.all_indices(), torch.device("cpu"), torch.float64)
        residual = control_volume_residual(
            LogPressure(), ConstantFilm(), batch,
            FluxScales(1.0, 1.0, 1.0, 0.0),
            lambda_value=0.0, xi_ratio=1.0, create_graph=False,
        )
        self.assertLess(float(residual.normalized.abs().max()), 2e-14)


if __name__ == "__main__":
    unittest.main()
