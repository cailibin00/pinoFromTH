from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig, StageConfig
from .constraints import AugmentedLagrangian, fischer_burmeister
from .control_volume import ControlVolumeGrid
from .diagnostics import (
    HistoryRecorder,
    RunArtifacts,
    TrainingDiagnostics,
    gradient_stats,
    plot_solution_fields,
    write_json,
)
from .evaluation import (
    prepare_evaluation,
    run_final_evaluation,
    run_isoline_evaluation,
)
from .geometry import FilmGeometry, compute_physical_params
from .networks import CVALModel
from .physics import (
    FluxScales,
    compute_flux_scales,
    control_volume_residual,
    evaluate_flux,
)


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(value: str) -> torch.dtype:
    choices = {"float32": torch.float32, "float64": torch.float64}
    if value not in choices:
        raise ValueError(f"Unsupported dtype: {value}")
    return choices[value]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg: ExperimentConfig, project_root: Path):
        self.cfg = cfg
        self.project_root = project_root.resolve()
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype)
        torch.set_default_dtype(self.dtype)
        _seed_everything(cfg.seed)

        self.params = compute_physical_params(cfg)
        self.geometry = FilmGeometry(self.params)
        self.scales = compute_flux_scales(cfg, self.params)
        self.model = CVALModel(cfg, self.params).to(
            device=self.device, dtype=self.dtype
        )
        self.train_grid = ControlVolumeGrid(
            self.params,
            cfg.train_cells_r,
            cfg.train_cells_theta,
            cfg.gauss_order,
        )
        self.validation_grid = ControlVolumeGrid(
            self.params,
            cfg.validation_cells_r,
            cfg.validation_cells_theta,
            cfg.gauss_order,
        )
        output_dir = cfg.resolve_path(self.project_root, cfg.output_dir)
        self.artifacts = RunArtifacts(output_dir)
        self.history = HistoryRecorder(self.artifacts)
        self.diagnostics = TrainingDiagnostics(self.artifacts)
        self.rng = np.random.default_rng(cfg.seed)
        self.al = AugmentedLagrangian(
            len(self.train_grid),
            self.device,
            self.dtype,
            cfg.al_mu_initial,
            cfg.al_mu_growth,
            cfg.al_mu_max,
            cfg.al_progress_ratio,
        )
        self.global_epoch = 0
        self.best_score: tuple[float, ...] | None = None
        self.best_metrics: dict[str, float] = {}
        self.best_path = self.artifacts.checkpoints / "best.pt"
        self.static_sampling_weights = self._build_static_sampling_weights()
        self.active_sampling_weights: np.ndarray | None = None
        self.active_sampling_epoch = -1

    def _stage_lambda(self, stage: StageConfig) -> float:
        return self.params.lambda_value * stage.lambda_ratio

    def _lr(self, epoch: int, total_epochs: int) -> float:
        warmup = min(self.cfg.warmup_epochs, max(1, total_epochs // 3))
        if epoch <= warmup:
            return self.cfg.peak_lr * epoch / warmup
        progress = (epoch - warmup) / max(1, total_epochs - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.cfg.min_lr + (self.cfg.peak_lr - self.cfg.min_lr) * cosine

    def _configure_stage_model(self, stage: StageConfig) -> None:
        self.model.set_gamma_enabled(stage.gamma_enabled)
        self.model.configure_active_gate(
            stage.gamma_enabled and stage.active_gate_enabled,
            stage.active_pressure_ratio,
            stage.active_gate_tau,
        )

    def _build_static_sampling_weights(self) -> np.ndarray:
        if not self.cfg.stratified_sampling:
            return np.ones(len(self.train_grid), dtype=np.float64)
        centers = torch.as_tensor(self.train_grid.centers, dtype=self.dtype)
        with torch.no_grad():
            film = self.geometry.film_thickness(centers, xi_ratio=1.0).cpu().numpy()
        h_min = float(film.min())
        h_max = float(film.max())
        h_norm = (film.reshape(-1) - h_min) / max(h_max - h_min, 1e-12)
        transition = 4.0 * h_norm * (1.0 - h_norm)

        radius = self.train_grid.centers[:, 0]
        theta = self.train_grid.centers[:, 1]
        radial_t = (radius - self.params.r_min) / (self.params.r_max - self.params.r_min)
        theta_t = (theta - self.params.theta_min) / (
            self.params.theta_max - self.params.theta_min
        )
        radial_edge = np.exp(-np.minimum(radial_t, 1.0 - radial_t) / 0.08)
        seam_edge = np.exp(-np.minimum(theta_t, 1.0 - theta_t) / 0.08)

        weights = (
            1.0
            + self.cfg.transition_sampling_weight * transition
            + self.cfg.radial_boundary_sampling_weight * radial_edge
            + self.cfg.periodic_seam_sampling_weight * seam_edge
        )
        return np.maximum(weights.astype(np.float64), 1.0e-12)

    @torch.no_grad()
    def _refresh_active_sampling_weights(self, stage: StageConfig) -> np.ndarray:
        if (
            not self.cfg.stratified_sampling
            or not stage.gamma_enabled
            or self.cfg.active_sampling_weight <= 0.0
        ):
            return self.static_sampling_weights
        interval = max(1, self.cfg.active_sampling_interval)
        if (
            self.active_sampling_weights is not None
            and self.global_epoch - self.active_sampling_epoch < interval
        ):
            return self.active_sampling_weights
        parts: list[np.ndarray] = []
        all_indices = self.train_grid.all_indices()
        for start in range(0, len(all_indices), self.cfg.cell_batch_size):
            index = all_indices[start : start + self.cfg.cell_batch_size]
            centers = self.train_grid.center_tensor(
                index, self.device, self.dtype
            )
            components = self.model.output_components(centers)
            active = torch.maximum(
                components.active_gate,
                (components.gamma > self.cfg.gamma_active_threshold).to(self.dtype),
            )
            parts.append(active.reshape(-1).detach().cpu().numpy())
        active_values = np.concatenate(parts)
        self.active_sampling_weights = self.static_sampling_weights * (
            1.0 + self.cfg.active_sampling_weight * active_values
        )
        self.active_sampling_epoch = self.global_epoch
        return self.active_sampling_weights

    def _sample_train_indices(self, stage: StageConfig) -> np.ndarray:
        weights = self._refresh_active_sampling_weights(stage)
        return self.train_grid.sample_indices(
            self.cfg.cell_batch_size,
            self.rng,
            weights=weights,
        )

    def _objective(
        self,
        stage: StageConfig,
        indices_np: np.ndarray,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = self.train_grid.face_batch(
            indices_np, self.device, self.dtype
        )
        residual = control_volume_residual(
            self.model,
            self.geometry,
            batch,
            self.scales,
            self._stage_lambda(stage),
            stage.xi_ratio,
            create_graph=create_graph,
        )
        loss_cv = torch.mean(residual.normalized.square())
        centers = batch.centers
        center_components = self.model.output_components(centers)
        pressure = center_components.pressure
        gamma = center_components.gamma
        fb = fischer_burmeister(
            pressure, gamma, self.cfg.pressure_ref, stage.epsilon_fb
        )
        if stage.gamma_enabled:
            indices = torch.as_tensor(
                indices_np, device=self.device, dtype=torch.long
            )
            loss_constraint = self.al.loss(indices, fb)
            loss_fb = torch.mean(fb.square())
        else:
            loss_constraint = torch.zeros((), device=self.device, dtype=self.dtype)
            loss_fb = torch.zeros((), device=self.device, dtype=self.dtype)
        total = loss_cv + loss_constraint
        return total, {
            "loss_cv": loss_cv,
            "loss_constraint": loss_constraint,
            "loss_fb": loss_fb,
            "pressure": pressure,
            "gamma": gamma,
            "active_gate": center_components.active_gate,
            "gamma_candidate": center_components.gamma_candidate,
            "fb": fb,
            "cv_residual": residual.normalized,
        }

    @torch.no_grad()
    def _update_al(self, stage: StageConfig) -> dict[str, float]:
        if not stage.gamma_enabled:
            return {
                "al_violation_max": 0.0,
                "al_violation_mean": 0.0,
                "al_mu": self.al.mu,
            }
        all_indices = self.train_grid.all_indices()
        constraints = []
        for start in range(0, len(all_indices), self.cfg.cell_batch_size):
            index = all_indices[start : start + self.cfg.cell_batch_size]
            centers = self.train_grid.center_tensor(
                index, self.device, self.dtype
            )
            pressure, gamma = self.model.output_fields(centers)
            constraints.append(
                fischer_burmeister(
                    pressure,
                    gamma,
                    self.cfg.pressure_ref,
                    stage.epsilon_fb,
                )
            )
        update = self.al.update(torch.cat(constraints, dim=0))
        return {
            "al_violation_max": update.violation_max,
            "al_violation_mean": update.violation_mean,
            "al_mu": update.mu,
            "al_mu_increased": float(update.mu_increased),
        }

    def _validation(self, stage: StageConfig) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        self.model.eval()
        all_indices = self.validation_grid.all_indices()
        normalized_parts: list[torch.Tensor] = []
        raw_parts: list[torch.Tensor] = []
        pressure_parts: list[torch.Tensor] = []
        gamma_parts: list[torch.Tensor] = []
        active_gate_parts: list[torch.Tensor] = []
        gamma_candidate_parts: list[torch.Tensor] = []
        pressure_bar_parts: list[torch.Tensor] = []
        film_parts: list[torch.Tensor] = []
        qr_parts: list[torch.Tensor] = []
        qt_parts: list[torch.Tensor] = []
        fb_parts: list[torch.Tensor] = []
        scale_sum = 0.0
        for start in range(0, len(all_indices), self.cfg.cell_batch_size):
            index = all_indices[start : start + self.cfg.cell_batch_size]
            batch = self.validation_grid.face_batch(
                index, self.device, self.dtype
            )
            residual = control_volume_residual(
                self.model,
                self.geometry,
                batch,
                self.scales,
                self._stage_lambda(stage),
                stage.xi_ratio,
                create_graph=False,
            )
            flux = evaluate_flux(
                self.model,
                self.geometry,
                batch.centers,
                self._stage_lambda(stage),
                stage.xi_ratio,
                create_graph=False,
            )
            fb = fischer_burmeister(
                flux.pressure,
                flux.gamma,
                self.cfg.pressure_ref,
                stage.epsilon_fb,
            )
            center_components = self.model.output_components(batch.centers)
            normalized_parts.append(residual.normalized.detach())
            raw_parts.append(residual.raw_flux.detach())
            pressure_parts.append(flux.pressure.detach())
            gamma_parts.append(flux.gamma.detach())
            active_gate_parts.append(center_components.active_gate.detach())
            gamma_candidate_parts.append(center_components.gamma_candidate.detach())
            pressure_bar_parts.append(center_components.pressure_bar.detach())
            film_parts.append(flux.film.detach())
            qr_parts.append(flux.q_radius.detach())
            qt_parts.append(flux.q_theta.detach())
            fb_parts.append(fb.detach())
            cell_scale = (
                self.scales.q_radius * batch.delta_theta
                + self.scales.q_theta * batch.delta_radius
            )
            scale_sum += float(cell_scale.sum().detach().cpu())

        cv = torch.cat(normalized_parts)
        raw = torch.cat(raw_parts)
        pressure = torch.cat(pressure_parts)
        gamma = torch.cat(gamma_parts)
        active_gate = torch.cat(active_gate_parts)
        gamma_candidate = torch.cat(gamma_candidate_parts)
        pressure_bar = torch.cat(pressure_bar_parts)
        film = torch.cat(film_parts)
        q_radius = torch.cat(qr_parts)
        q_theta = torch.cat(qt_parts)
        fb = torch.cat(fb_parts)
        boundary_error, periodic_error = self._hard_constraint_errors()
        metrics = {
            "cv_mean_abs": float(cv.abs().mean().cpu()),
            "cv_p99": float(torch.quantile(cv.abs(), 0.99).cpu()),
            "cv_rmse": float(torch.sqrt(torch.mean(cv.square())).cpu()),
            "global_flux_balance": float(raw.sum().abs().cpu()) / max(scale_sum, 1e-30),
            "fb_mean_abs": float(fb.abs().mean().cpu()),
            "fb_p99": float(torch.quantile(fb.abs(), 0.99).cpu()),
            "max_p_times_gamma": float((pressure * gamma).abs().max().cpu()),
            "mean_p_times_gamma": float((pressure * gamma).abs().mean().cpu()),
            "pressure_min": float(pressure.min().cpu()),
            "pressure_max": float(pressure.max().cpu()),
            "gamma_min": float(gamma.min().cpu()),
            "gamma_max": float(gamma.max().cpu()),
            "gamma_active_fraction": float(
                (gamma > self.cfg.gamma_active_threshold).to(self.dtype).mean().cpu()
            ),
            "gamma_candidate_max": float(gamma_candidate.max().cpu()),
            "active_gate_mean": float(active_gate.mean().cpu()),
            "active_gate_p99": float(torch.quantile(active_gate, 0.99).cpu()),
            "active_gate_max": float(active_gate.max().cpu()),
            "low_pressure_fraction": float(
                (pressure_bar < stage.active_pressure_ratio).to(self.dtype).mean().cpu()
            ),
            "boundary_max_error": boundary_error,
            "periodic_max_error": periodic_error,
        }
        fields = {
            "pressure": pressure,
            "gamma": gamma,
            "active_gate": active_gate,
            "gamma_candidate": gamma_candidate,
            "pressure_bar": pressure_bar,
            "film": film,
            "q_radius": q_radius,
            "q_theta": q_theta,
            "cv_residual": cv,
            "fb": fb,
        }
        return metrics, fields

    @torch.no_grad()
    def _hard_constraint_errors(self) -> tuple[float, float]:
        theta = torch.linspace(
            self.params.theta_min,
            self.params.theta_max,
            129,
            device=self.device,
            dtype=self.dtype,
        ).reshape(-1, 1)
        inner = torch.cat([torch.full_like(theta, self.params.r_min), theta], dim=1)
        outer = torch.cat([torch.full_like(theta, self.params.r_max), theta], dim=1)
        inner_values = self.model(inner)
        outer_values = self.model(outer)
        errors = torch.cat(
            [
                (inner_values[:, 0] - self.params.pressure_inner).abs(),
                (outer_values[:, 0] - self.params.pressure_outer).abs(),
                inner_values[:, 1].abs(),
                outer_values[:, 1].abs(),
            ]
        )
        radius = torch.linspace(
            self.params.r_min,
            self.params.r_max,
            129,
            device=self.device,
            dtype=self.dtype,
        ).reshape(-1, 1)
        start = torch.cat(
            [radius, torch.full_like(radius, self.params.theta_min)], dim=1
        )
        end = torch.cat(
            [radius, torch.full_like(radius, self.params.theta_max)], dim=1
        )
        periodic = (self.model(start) - self.model(end)).abs().max()
        return float(errors.max().cpu()), float(periodic.cpu())

    def _save_best_if_needed(
        self,
        stage_index: int,
        stage: StageConfig,
        epoch: int,
        metrics: dict[str, float],
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        is_true_physics = (
            stage.gamma_enabled
            and abs(stage.lambda_ratio - 1.0) < 1e-12
            and abs(stage.xi_ratio - 1.0) < 1e-12
        )
        if not is_true_physics:
            return False
        feasibility = max(
            metrics["global_flux_balance"] / self.cfg.best_global_flux_tolerance,
            metrics["boundary_max_error"] / self.cfg.best_boundary_tolerance,
            metrics["periodic_max_error"] / self.cfg.best_periodic_tolerance,
        )
        feasibility_rank = 0.0 if feasibility <= 1.0 else feasibility
        active_shortfall = max(
            0.0,
            self.cfg.best_min_gamma_active_fraction
            - metrics["gamma_active_fraction"],
        ) / max(self.cfg.best_min_gamma_active_fraction, 1e-12)
        score = (
            feasibility_rank,
            active_shortfall,
            metrics["cv_p99"],
            metrics["fb_p99"],
            metrics["max_p_times_gamma"],
            metrics["cv_mean_abs"],
        )
        if self.best_score is not None and score >= self.best_score:
            return False
        self.best_score = score
        self.best_metrics = dict(metrics)
        save_checkpoint(
            self.best_path,
            self.model,
            self.cfg,
            stage_index,
            stage.name,
            epoch,
            metrics,
            optimizer,
            self.al.state_dict(),
        )
        return True

    def _record_validation(
        self,
        stage_index: int,
        stage: StageConfig,
        stage_epoch: int,
        train_values: dict[str, float],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        metrics, fields = self._validation(stage)
        record: dict[str, Any] = {
            "global_epoch": self.global_epoch,
            "stage_index": stage_index,
            "stage": stage.name,
            "stage_epoch": stage_epoch,
            **train_values,
            **metrics,
            "al_mu": self.al.mu,
        }
        self.history.append(record)
        gradients = gradient_stats(self.model)
        self.diagnostics.snapshot(
            stage_index,
            stage.name,
            self.global_epoch,
            metrics,
            fields,
            gradients,
        )
        is_best = self._save_best_if_needed(
            stage_index, stage, stage_epoch, metrics, optimizer
        )
        print(
            f"[validation] stage={stage.name} epoch={stage_epoch} "
            f"cv_p99={metrics['cv_p99']:.4e} fb_p99={metrics['fb_p99']:.4e} "
            f"gamma_frac={metrics['gamma_active_fraction']:.3e} "
            f"global={metrics['global_flux_balance']:.4e} best={is_best}"
        )
        self.model.train()
        return metrics

    def _run_adam_stage(self, stage_index: int, stage: StageConfig) -> None:
        self._configure_stage_model(stage)
        self.active_sampling_weights = None
        self.active_sampling_epoch = -1
        if self.cfg.reset_al_each_stage:
            self.al.reset(self.cfg.al_mu_initial)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.peak_lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
        )
        last_train: dict[str, float] = {}
        print(
            f"\n[stage {stage_index + 1}/{len(self.cfg.stages)}] {stage.name}: "
            f"lambda={stage.lambda_ratio:.3f}, xi={stage.xi_ratio:.3f}, "
            f"gamma={stage.gamma_enabled}, epochs={stage.adam_epochs}"
        )
        for epoch in range(1, stage.adam_epochs + 1):
            self.global_epoch += 1
            learning_rate = self._lr(epoch, stage.adam_epochs)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            indices = self._sample_train_indices(stage)
            optimizer.zero_grad(set_to_none=True)
            total, components = self._objective(stage, indices, create_graph=True)
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite objective in {stage.name} epoch {epoch}: {total.item()}"
                )
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.gradient_clip_norm
            )
            optimizer.step()
            last_train = {
                "loss_total": float(total.detach().cpu()),
                "loss_cv": float(components["loss_cv"].detach().cpu()),
                "loss_constraint": float(
                    components["loss_constraint"].detach().cpu()
                ),
                "loss_fb": float(components["loss_fb"].detach().cpu()),
                "train_gamma_max": float(components["gamma"].max().detach().cpu()),
                "train_active_gate_mean": float(
                    components["active_gate"].mean().detach().cpu()
                ),
                "train_gamma_candidate_max": float(
                    components["gamma_candidate"].max().detach().cpu()
                ),
                "gradient_norm": float(torch.as_tensor(grad_norm).cpu()),
                "learning_rate": learning_rate,
            }
            if epoch % self.cfg.al_update_interval == 0 or epoch == stage.adam_epochs:
                last_train.update(self._update_al(stage))
            should_validate = (
                epoch == 1
                or epoch == stage.adam_epochs
                or epoch % self.cfg.validation_interval == 0
            )
            if should_validate:
                self._record_validation(
                    stage_index, stage, epoch, last_train, optimizer
                )
            elif epoch % self.cfg.log_interval == 0:
                print(
                    f"[train] stage={stage.name} epoch={epoch}/{stage.adam_epochs} "
                    f"loss={last_train['loss_total']:.4e} "
                    f"cv={last_train['loss_cv']:.4e} lr={learning_rate:.3e}"
                )

        save_checkpoint(
            self.artifacts.checkpoints / f"stage_{stage_index + 1:02d}_{stage.name}.pt",
            self.model,
            self.cfg,
            stage_index,
            stage.name,
            stage.adam_epochs,
            self.diagnostics.snapshots[-1]["metrics"],
            optimizer,
            self.al.state_dict(),
        )
        if stage.lbfgs_steps > 0:
            self._run_lbfgs(stage_index, stage)

    def _run_lbfgs(self, stage_index: int, stage: StageConfig) -> None:
        indices = self.train_grid.all_indices()
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=stage.lbfgs_steps,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0
        latest: dict[str, float] = {}

        def closure() -> torch.Tensor:
            nonlocal closure_calls, latest
            optimizer.zero_grad(set_to_none=True)
            total, components = self._objective(stage, indices, create_graph=True)
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite objective during L-BFGS")
            total.backward()
            closure_calls += 1
            latest = {
                "loss_total": float(total.detach().cpu()),
                "loss_cv": float(components["loss_cv"].detach().cpu()),
                "loss_constraint": float(
                    components["loss_constraint"].detach().cpu()
                ),
                "loss_fb": float(components["loss_fb"].detach().cpu()),
                "gradient_norm": float("nan"),
                "learning_rate": 1.0,
            }
            return total

        print(f"[L-BFGS] stage={stage.name}, max_iter={stage.lbfgs_steps}")
        optimizer.step(closure)
        self.global_epoch += 1
        latest["lbfgs_closure_calls"] = float(closure_calls)
        self._update_al(stage)
        metrics = self._record_validation(
            stage_index,
            stage,
            stage.adam_epochs + stage.lbfgs_steps,
            latest,
            optimizer,
        )
        save_checkpoint(
            self.artifacts.checkpoints / f"stage_{stage_index + 1:02d}_{stage.name}_lbfgs.pt",
            self.model,
            self.cfg,
            stage_index,
            stage.name,
            stage.adam_epochs + stage.lbfgs_steps,
            metrics,
            optimizer,
            self.al.state_dict(),
        )

    def _restore_best(self) -> None:
        if not self.best_path.exists():
            raise RuntimeError(
                "No best checkpoint was produced. The schedule must include a "
                "gamma-enabled stage with lambda_ratio=xi_ratio=1."
            )
        model, _, payload = load_checkpoint(self.best_path, self.device)
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.model.set_gamma_enabled(True)
        self.best_metrics = dict(payload.get("metrics", {}))
        print(
            f"\n[restore] best checkpoint: stage={payload.get('stage_name')} "
            f"epoch={payload.get('epoch')} cv_p99={self.best_metrics.get('cv_p99', float('nan')):.4e}"
        )

    def _snapshot_restored_best(self) -> None:
        true_stage = next(
            (
                stage
                for stage in reversed(self.cfg.stages)
                if stage.gamma_enabled
                and abs(stage.lambda_ratio - 1.0) < 1e-12
                and abs(stage.xi_ratio - 1.0) < 1e-12
            ),
            None,
        )
        if true_stage is None:
            raise RuntimeError("Cannot diagnose best model without a true-physics stage")
        metrics, fields = self._validation(true_stage)
        self.best_metrics = metrics
        self.diagnostics.snapshot(
            len(self.cfg.stages),
            "best_restored",
            self.global_epoch,
            metrics,
            fields,
            gradient_stats(self.model),
        )
        post_dir = self.artifacts.diagnostics / "post_training"
        post_dir.mkdir(parents=True, exist_ok=True)
        write_json(post_dir / "physics_metrics_full.json", metrics)
        write_json(
            post_dir / "jfo_full.json",
            {
                key: metrics[key]
                for key in (
                    "fb_mean_abs",
                    "fb_p99",
                    "max_p_times_gamma",
                    "mean_p_times_gamma",
                    "pressure_min",
                    "gamma_min",
                    "gamma_max",
                )
            },
        )

    def _automatic_evaluation(self) -> None:
        print("\n[evaluation] preparing FEM data and best-model predictions")
        bundle = prepare_evaluation(
            self.model, self.cfg, self.project_root, self.device
        )
        errors: list[str] = []
        try:
            run_final_evaluation(
                self.model,
                self.cfg,
                self.project_root,
                self.artifacts.comparison,
                self.device,
                bundle=bundle,
            )
            print(f"[evaluation] final comparison: {self.artifacts.comparison}")
        except Exception as exc:  # Run the second report even if the first fails.
            errors.append(f"final comparison: {type(exc).__name__}: {exc}")
        try:
            run_isoline_evaluation(
                self.model,
                self.cfg,
                self.project_root,
                self.artifacts.isolines,
                self.device,
                bundle=bundle,
            )
            print(f"[evaluation] isoline analysis: {self.artifacts.isolines}")
        except Exception as exc:
            errors.append(f"isoline analysis: {type(exc).__name__}: {exc}")
        if errors:
            message = "\n".join(errors) + "\n"
            (self.artifacts.root / "evaluation_errors.txt").write_text(
                message, encoding="utf-8"
            )
            raise RuntimeError("Automatic evaluation failed:\n" + message)

    def run(self) -> Path:
        write_json(self.artifacts.root / "config.json", self.cfg.to_dict())
        write_json(self.artifacts.root / "physical_params.json", self.params.to_dict())
        write_json(self.artifacts.root / "flux_scales.json", self.scales.to_dict())
        write_json(
            self.artifacts.root / "model_summary.json",
            self.model.parameter_summary(),
        )
        print("Torch CV-AL PINN")
        print(f"device={self.device}, dtype={self.dtype}, output={self.artifacts.root}")
        print(f"parameters={self.model.parameter_summary()}")
        print(f"flux_scales={self.scales.to_dict()}")
        for stage_index, stage in enumerate(self.cfg.stages):
            self._run_adam_stage(stage_index, stage)

        self._restore_best()
        self._snapshot_restored_best()
        save_checkpoint(
            self.artifacts.checkpoints / "final_best_restored.pt",
            self.model,
            self.cfg,
            len(self.cfg.stages) - 1,
            "best_restored",
            self.global_epoch,
            self.best_metrics,
        )
        plot_solution_fields(
            self.model,
            self.geometry,
            self.params,
            1.0,
            self.artifacts.figures / "best_solution_fields.png",
            self.device,
            self.dtype,
            self.cfg.dpi,
        )
        self.history.save()
        self.history.plot(self.cfg.dpi)
        self.diagnostics.finalize()
        if self.cfg.auto_run_evaluations:
            self._automatic_evaluation()
        write_json(
            self.artifacts.root / "run_complete.json",
            {
                "status": "complete",
                "best_checkpoint": str(self.best_path),
                "best_metrics": self.best_metrics,
                "automatic_evaluations": self.cfg.auto_run_evaluations,
            },
        )
        print(f"\n[complete] {self.artifacts.root}")
        return self.artifacts.root
