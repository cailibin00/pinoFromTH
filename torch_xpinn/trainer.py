from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import XPINNConfig
from .geometry import HardGrooveGeometry, Region, compute_physical_params
from .logging import format_loss_block, scalar, weighted_total
from .networks import XPINNModel
from .physics import interface_reynolds_loss, region_jfo_loss, region_reynolds_loss

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional display dependency
    tqdm = None


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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, cfg.epochs),
            eta_min=cfg.min_learning_rate,
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
            "reynolds": self.cfg.reynolds_weight,
            "jfo": self.cfg.jfo_weight,
            "boundary": self.cfg.boundary_weight,
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

    def losses(self) -> dict[str, torch.Tensor]:
        thin_points = self._region_points(Region.THIN, self.cfg.thin_points)
        groove_points = self._region_points(Region.GROOVE, self.cfg.groove_points)
        interface_points, normals = self.geometry.interface_points(
            self.cfg.interface_points, self.device, self.dtype
        )

        reynolds = torch.stack(
            [
                region_reynolds_loss(
                    self.model, self.geometry, thin_points, Region.THIN
                ),
                region_reynolds_loss(
                    self.model, self.geometry, groove_points, Region.GROOVE
                ),
                interface_reynolds_loss(
                    self.model, self.geometry, interface_points, normals
                ),
            ]
        ).mean()
        jfo = torch.stack(
            [
                region_jfo_loss(self.model, thin_points, Region.THIN),
                region_jfo_loss(self.model, groove_points, Region.GROOVE),
            ]
        ).mean()
        boundary = torch.stack(
            [
                self._boundary_loss(
                    self.params.r_min, self.params.pressure_inner
                ),
                self._boundary_loss(
                    self.params.r_max, self.params.pressure_outer
                ),
            ]
        ).mean()
        return {"reynolds": reynolds, "jfo": jfo, "boundary": boundary}

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
        use_tqdm = self.cfg.use_tqdm and tqdm is not None

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

        epoch_range = range(1, self.cfg.epochs + 1)
        progress = (
            tqdm(epoch_range, desc="XPINN adam", dynamic_ncols=True)
            if use_tqdm
            else epoch_range
        )

        for epoch in progress:
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
            self.scheduler.step()
            if total_value < best:
                best = total_value
                best_epoch = epoch
                self.save_checkpoint("best.pt", epoch, best)

            if use_tqdm:
                progress.set_postfix(
                    total=f"{total_value:.3e}",
                    reynolds=f"{scalar(losses['reynolds']):.2e}",
                    jfo=f"{scalar(losses['jfo']):.2e}",
                    bc=f"{scalar(losses['boundary']):.2e}",
                )

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
                if use_tqdm:
                    progress.write(block)
                else:
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
