from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import XPINNConfig
from .geometry import HardGrooveGeometry, Region, compute_physical_params
from .logging import format_loss_block, scalar, weighted_total
from .networks import XPINNModel
from .physics import (
    evaluate_region_flux,
    interface_losses,
    region_fb_loss,
    region_pde_loss,
)


def resolve_dtype(value: str) -> torch.dtype:
    choices = {"float32": torch.float32, "float64": torch.float64}
    if value not in choices:
        raise ValueError(f"Unsupported dtype: {value}")
    return choices[value]


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XPINNTrainer:
    def __init__(self, cfg: XPINNConfig, project_root: Path):
        self.cfg = cfg
        self.project_root = project_root
        self.dtype = resolve_dtype(cfg.dtype)
        self.device = resolve_device(cfg.device)
        torch.set_default_dtype(self.dtype)
        torch.manual_seed(cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)

        self.geometry = HardGrooveGeometry(cfg)
        self.params = compute_physical_params(cfg)
        self.model = XPINNModel(cfg, self.params).to(device=self.device, dtype=self.dtype)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.scheduler_gamma,
            patience=cfg.scheduler_patience,
            min_lr=cfg.min_learning_rate,
        )
        self.generator = self._make_generator()
        self.output_dir = cfg.resolve_path(project_root, cfg.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_path = self.output_dir / "train.log"

    def _make_generator(self) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.cfg.seed)
        return generator

    def weights(self) -> dict[str, float]:
        return {
            "thin_pde": self.cfg.thin_pde_weight,
            "groove_pde": self.cfg.groove_pde_weight,
            "interface_pressure": self.cfg.interface_pressure_weight,
            "interface_flux": self.cfg.interface_flux_weight,
            "thin_fb": self.cfg.thin_fb_weight,
            "groove_fb": self.cfg.groove_fb_weight,
            "inner_boundary": self.cfg.boundary_weight,
            "outer_boundary": self.cfg.boundary_weight,
            "periodic_value": self.cfg.periodic_weight,
            "periodic_flux": self.cfg.periodic_weight,
        }

    def _region_points(self, region: Region, count: int) -> torch.Tensor:
        return self.geometry.sample_region(
            region, count, self.device, self.dtype, self.generator
        )

    def _boundary_loss(self, radius: float, target: float) -> torch.Tensor:
        points = self.geometry.radial_boundary_points(
            radius,
            self.cfg.boundary_points,
            self.device,
            self.dtype,
            self.generator,
        )
        losses = []
        for region in (Region.THIN, Region.GROOVE):
            pressure, _ = self.model.forward_region(points, region)
            losses.append(torch.mean((pressure - target).square()))
        return torch.stack(losses).mean()

    def _periodic_losses(self) -> dict[str, torch.Tensor]:
        left, right = self.geometry.periodic_pairs(
            self.cfg.periodic_points,
            self.device,
            self.dtype,
            self.generator,
        )
        value_terms = []
        flux_terms = []
        for region in (Region.THIN, Region.GROOVE):
            pressure_left, gamma_left = self.model.forward_region(left, region)
            pressure_right, gamma_right = self.model.forward_region(right, region)
            value_terms.append(
                torch.mean((pressure_left - pressure_right).square())
                + torch.mean((gamma_left - gamma_right).square())
            )

            flux_left = evaluate_region_flux(
                self.model, self.geometry, left, region, create_graph=True
            )
            flux_right = evaluate_region_flux(
                self.model, self.geometry, right, region, create_graph=True
            )
            flux_terms.append(
                torch.mean((flux_left.q_radius - flux_right.q_radius).square())
                + torch.mean((flux_left.q_theta - flux_right.q_theta).square())
            )
        return {
            "periodic_value": torch.stack(value_terms).mean(),
            "periodic_flux": torch.stack(flux_terms).mean(),
        }

    def losses(self) -> dict[str, torch.Tensor]:
        thin_points = self._region_points(Region.THIN, self.cfg.thin_points)
        groove_points = self._region_points(Region.GROOVE, self.cfg.groove_points)
        interface_points_, normals = self.geometry.interface_points(
            self.cfg.interface_points, self.device, self.dtype
        )
        losses: dict[str, torch.Tensor] = {
            "thin_pde": region_pde_loss(
                self.model, self.geometry, thin_points, Region.THIN
            ),
            "groove_pde": region_pde_loss(
                self.model, self.geometry, groove_points, Region.GROOVE
            ),
            "thin_fb": region_fb_loss(
                self.model, self.geometry, thin_points, Region.THIN
            ),
            "groove_fb": region_fb_loss(
                self.model, self.geometry, groove_points, Region.GROOVE
            ),
            "inner_boundary": self._boundary_loss(
                self.params.r_min, self.params.pressure_inner
            ),
            "outer_boundary": self._boundary_loss(
                self.params.r_max, self.params.pressure_outer
            ),
        }
        losses.update(
            interface_losses(
                self.model,
                self.geometry,
                interface_points_,
                normals,
                create_graph=True,
            )
        )
        losses.update(self._periodic_losses())
        return losses

    def save_checkpoint(self, name: str, epoch: int, metric: float) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / name
        torch.save(
            {
                "epoch": epoch,
                "metric": metric,
                "config": self.cfg.to_dict(),
                "params": self.params,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )
        return path

    def _write_log(self, text: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")

    def prepare_output(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "config.json").write_text(
            json.dumps(self.cfg.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.log_path.write_text("", encoding="utf-8")

    def train(self) -> dict[str, Any]:
        self.prepare_output()
        weights = self.weights()
        best = float("inf")
        best_epoch = 0
        history: list[dict[str, float]] = []

        print("Hard-partition XPINN training")
        print(f"device={self.device}, dtype={self.dtype}")
        print(f"lambda={self.params.lambda_value:.6f}")
        print(
            "domain "
            f"R=[{self.params.r_min:.10f}, {self.params.r_max:.10f}] "
            f"theta=[0, {self.params.theta_max:.10f}]"
        )
        print(
            "groove "
            f"R=[{self.params.groove_r_min:.10f}, {self.params.groove_r_max:.10f}]"
        )

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            losses = self.losses()
            total = weighted_total(losses, weights)
            total.backward()
            if self.cfg.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.gradient_clip_norm
                )
            self.optimizer.step()

            total_value = scalar(total)
            self.scheduler.step(total_value)
            if total_value < best:
                best = total_value
                best_epoch = epoch
                self.save_checkpoint("best.pt", epoch, best)

            should_log = epoch == 1 or epoch % self.cfg.log_interval == 0
            if should_log:
                lr = self.optimizer.param_groups[0]["lr"]
                block = format_loss_block(
                    stage="adam",
                    epoch=epoch,
                    losses=losses,
                    weights=weights,
                    learning_rate=lr,
                )
                print(block)
                self._write_log(block)
                history.append(
                    {
                        "epoch": float(epoch),
                        "total": total_value,
                        **{key: scalar(value) for key, value in losses.items()},
                    }
                )

            if (
                self.cfg.checkpoint_interval > 0
                and epoch % self.cfg.checkpoint_interval == 0
            ):
                self.save_checkpoint(f"epoch_{epoch:06d}.pt", epoch, total_value)

        final_path = self.save_checkpoint("final.pt", self.cfg.epochs, total_value)
        (self.output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        return {
            "best": best,
            "best_epoch": best_epoch,
            "final": total_value,
            "best_checkpoint": str(self.checkpoint_dir / "best.pt"),
            "final_checkpoint": str(final_path),
            "output_dir": str(self.output_dir),
        }
