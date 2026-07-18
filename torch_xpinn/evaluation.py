from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import XPINNConfig
from .geometry import HardGrooveGeometry, Region, compute_physical_params
from .networks import XPINNModel
from .trainer import resolve_device, resolve_dtype


@dataclass
class EvaluationBundle:
    radius_points: np.ndarray
    theta_points: np.ndarray
    pressure_fem: np.ndarray
    gamma_fem: np.ndarray
    pressure_xpinn: np.ndarray
    gamma_xpinn: np.ndarray
    radius_unique: np.ndarray
    theta_unique: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.radius_unique), len(self.theta_unique)

    def grids(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shape = self.shape
        return (
            self.pressure_fem.reshape(shape),
            self.gamma_fem.reshape(shape),
            self.pressure_xpinn.reshape(shape),
            self.gamma_xpinn.reshape(shape),
        )


def load_fem(pressure_path: Path, gamma_path: Path) -> tuple[np.ndarray, ...]:
    pressure_raw = np.loadtxt(pressure_path)
    gamma_raw = np.loadtxt(gamma_path)
    if pressure_raw.shape[1] < 3 or gamma_raw.shape[1] < 3:
        raise ValueError("FEM files must contain R, theta, and value columns")
    if pressure_raw.shape[0] != gamma_raw.shape[0]:
        raise ValueError("Pressure and gamma FEM files have different row counts")
    if not np.allclose(pressure_raw[:, :2], gamma_raw[:, :2]):
        raise ValueError("Pressure and gamma FEM coordinates do not match")
    radius = pressure_raw[:, 0]
    theta = pressure_raw[:, 1]
    return (
        radius,
        theta,
        pressure_raw[:, 2],
        gamma_raw[:, 2],
        np.unique(radius),
        np.unique(theta),
    )


def predict_in_batches(
    model: XPINNModel,
    geometry: HardGrooveGeometry,
    coords: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = next(model.parameters()).dtype
    pressure_chunks: list[np.ndarray] = []
    gamma_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            batch = torch.as_tensor(
                coords[start : start + batch_size], device=device, dtype=dtype
            )
            mask = geometry.groove_mask(batch).squeeze(1)
            pressure = torch.empty((len(batch), 1), device=device, dtype=dtype)
            gamma = torch.empty((len(batch), 1), device=device, dtype=dtype)
            if mask.any():
                pressure_groove, gamma_groove = model.forward_region(
                    batch[mask], Region.GROOVE
                )
                pressure[mask] = pressure_groove
                gamma[mask] = gamma_groove
            if (~mask).any():
                pressure_thin, gamma_thin = model.forward_region(
                    batch[~mask], Region.THIN
                )
                pressure[~mask] = pressure_thin
                gamma[~mask] = gamma_thin
            pressure_chunks.append(pressure.squeeze(1).cpu().numpy())
            gamma_chunks.append(gamma.squeeze(1).cpu().numpy())
    return np.concatenate(pressure_chunks), np.concatenate(gamma_chunks)


def prepare_evaluation(
    model: XPINNModel,
    cfg: XPINNConfig,
    project_root: Path,
    device: torch.device,
) -> EvaluationBundle:
    geometry = HardGrooveGeometry(cfg)
    p_path = cfg.resolve_path(project_root, cfg.fem_pressure_path)
    g_path = cfg.resolve_path(project_root, cfg.fem_gamma_path)
    radius, theta, p_fem, g_fem, r_unique, t_unique = load_fem(p_path, g_path)
    coords = np.column_stack([radius, theta])
    p_xpinn, g_xpinn = predict_in_batches(
        model, geometry, coords, cfg.prediction_batch_size, device
    )
    return EvaluationBundle(
        radius,
        theta,
        p_fem,
        g_fem,
        p_xpinn,
        g_xpinn,
        r_unique,
        t_unique,
    )


def _rel_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.linalg.norm(reference - prediction)
        / (np.linalg.norm(reference) + np.finfo(float).eps)
    )


def _rel_linf(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.max(np.abs(reference - prediction))
        / (np.max(np.abs(reference)) + np.finfo(float).eps)
    )


def _threshold_key(prefix: str, threshold: float) -> str:
    return f"{prefix}_{threshold:.0e}".replace("+", "")


def compute_metrics(
    bundle: EvaluationBundle, thresholds: Iterable[float]
) -> dict[str, float]:
    p_ref, g_ref = bundle.pressure_fem, bundle.gamma_fem
    p_pred, g_pred = bundle.pressure_xpinn, bundle.gamma_xpinn
    metrics = {
        "P_rel_L2": _rel_l2(p_ref, p_pred),
        "P_rel_Linf": _rel_linf(p_ref, p_pred),
        "G_rel_L2": _rel_l2(g_ref, g_pred),
        "G_rel_Linf": _rel_linf(g_ref, g_pred),
        "P_MAE": float(np.mean(np.abs(p_ref - p_pred))),
        "P_RMSE": float(np.sqrt(np.mean((p_ref - p_pred) ** 2))),
        "G_MAE": float(np.mean(np.abs(g_ref - g_pred))),
        "G_RMSE": float(np.sqrt(np.mean((g_ref - g_pred) ** 2))),
        "complementarity_violation_max": float(np.max(p_pred * g_pred)),
        "complementarity_violation_mean": float(np.mean(p_pred * g_pred)),
    }
    cavitation = g_ref > 1.0e-6
    full_film = ~cavitation
    if cavitation.any():
        metrics["P_rel_L2_cavRegion"] = _rel_l2(
            p_ref[cavitation], p_pred[cavitation]
        )
        metrics["G_rel_L2_cavRegion"] = _rel_l2(
            g_ref[cavitation], g_pred[cavitation]
        )
    if full_film.any():
        metrics["P_rel_L2_fullRegion"] = _rel_l2(
            p_ref[full_film], p_pred[full_film]
        )
        metrics["G_abs_RMSE_fullRegion"] = float(
            np.sqrt(np.mean((g_ref[full_film] - g_pred[full_film]) ** 2))
        )
    eps = np.finfo(float).eps
    for threshold in thresholds:
        ref_mask = g_ref > threshold
        pred_mask = g_pred > threshold
        intersection = np.logical_and(ref_mask, pred_mask).sum()
        union = np.logical_or(ref_mask, pred_mask).sum()
        iou = intersection / (union + eps)
        dice = 2.0 * intersection / (ref_mask.sum() + pred_mask.sum() + eps)
        metrics[_threshold_key("cavitation_IoU", threshold)] = float(iou)
        metrics[_threshold_key("cavitation_Dice", threshold)] = float(dice)
    metrics["cavitation_IoU"] = metrics[_threshold_key("cavitation_IoU", 1.0e-6)]
    metrics["cavitation_Dice"] = metrics[_threshold_key("cavitation_Dice", 1.0e-6)]
    return metrics


def save_metrics(metrics: dict[str, float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as handle:
        handle.write("FEM vs Torch XPINN metrics\n" + "=" * 64 + "\n")
        for key, value in metrics.items():
            handle.write(f"{key:<48s} = {value:.8e}\n")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_field_data(bundle: EvaluationBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = np.column_stack(
        [
            bundle.radius_points,
            bundle.theta_points,
            bundle.pressure_fem,
            bundle.gamma_fem,
            bundle.pressure_xpinn,
            bundle.gamma_xpinn,
        ]
    )
    np.savetxt(
        output_dir / "fem_xpinn_fields.txt",
        columns,
        fmt="%.10e",
        header="R theta P_FEM gamma_FEM P_XPINN gamma_XPINN",
    )
    np.savez_compressed(
        output_dir / "fem_xpinn_fields.npz",
        radius=bundle.radius_points,
        theta=bundle.theta_points,
        pressure_fem=bundle.pressure_fem,
        gamma_fem=bundle.gamma_fem,
        pressure_xpinn=bundle.pressure_xpinn,
        gamma_xpinn=bundle.gamma_xpinn,
    )


def _mesh(bundle: EvaluationBundle) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(bundle.radius_unique, bundle.theta_unique, indexing="ij")


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_field_comparison(bundle: EvaluationBundle, output_dir: Path, dpi: int) -> None:
    p_fem, g_fem, p_xpinn, g_xpinn = bundle.grids()
    rr, tt = _mesh(bundle)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for values, title, ax in (
        (p_fem, "FEM pressure", axes[0, 0]),
        (p_xpinn, "XPINN pressure", axes[0, 1]),
        (g_fem, "FEM gamma", axes[1, 0]),
        (g_xpinn, "XPINN gamma", axes[1, 1]),
    ):
        mesh = ax.pcolormesh(rr, tt, values, shading="auto", cmap="RdYlBu_r")
        fig.colorbar(mesh, ax=ax, pad=0.02)
        ax.set_title(title)
        ax.set_xlabel("R")
        ax.set_ylabel("theta")
    fig.tight_layout()
    _save(fig, output_dir / "fig1_field_comparison.png", dpi)


def plot_error_maps(bundle: EvaluationBundle, output_dir: Path, dpi: int) -> None:
    p_fem, g_fem, p_xpinn, g_xpinn = bundle.grids()
    rr, tt = _mesh(bundle)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for values, title, ax in (
        (np.abs(p_fem - p_xpinn), "Absolute pressure error", axes[0]),
        (np.abs(g_fem - g_xpinn), "Absolute gamma error", axes[1]),
    ):
        mesh = ax.pcolormesh(rr, tt, values, shading="auto", cmap="hot_r")
        fig.colorbar(mesh, ax=ax, pad=0.02)
        ax.set_title(title)
        ax.set_xlabel("R")
        ax.set_ylabel("theta")
    fig.tight_layout()
    _save(fig, output_dir / "fig2_error_maps.png", dpi)


def plot_cavitation_boundary(
    bundle: EvaluationBundle, output_dir: Path, dpi: int, threshold: float = 1.0e-6
) -> None:
    _, g_fem, _, g_xpinn = bundle.grids()
    rr, tt = _mesh(bundle)
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(rr, tt, g_xpinn, shading="auto", cmap="Blues", alpha=0.55)
    fig.colorbar(mesh, ax=ax, label="XPINN gamma", pad=0.02)
    ax.contour(rr, tt, g_fem, levels=[threshold], colors="red", linewidths=1.8)
    ax.contour(
        rr,
        tt,
        g_xpinn,
        levels=[threshold],
        colors="blue",
        linewidths=1.8,
        linestyles="--",
    )
    ax.set_title(f"Cavitation boundary at gamma={threshold:.0e}")
    ax.set_xlabel("R")
    ax.set_ylabel("theta")
    fig.tight_layout()
    _save(fig, output_dir / "fig3_cavitation_boundary.png", dpi)


def run_final_evaluation(
    model: XPINNModel,
    cfg: XPINNConfig,
    project_root: Path,
    output_dir: Path,
    device: torch.device,
    bundle: EvaluationBundle | None = None,
) -> tuple[dict[str, float], EvaluationBundle]:
    print("[evaluation] preparing FEM data and XPINN predictions")
    data = bundle or prepare_evaluation(model, cfg, project_root, device)
    metrics = compute_metrics(data, cfg.iou_thresholds)
    save_metrics(metrics, output_dir)
    save_field_data(data, output_dir)
    plot_field_comparison(data, output_dir, cfg.dpi)
    plot_error_maps(data, output_dir, cfg.dpi)
    plot_cavitation_boundary(data, output_dir, cfg.dpi)
    for key in ("P_rel_L2", "G_rel_L2", "cavitation_IoU"):
        print(f"  {key:<28s} {metrics[key]:.8e}")
    print(f"[evaluation] saved metrics and figures to {output_dir}")
    return metrics, data


def load_checkpoint(
    checkpoint_path: Path, device: torch.device | None = None
) -> tuple[XPINNModel, XPINNConfig, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = XPINNConfig.from_dict(checkpoint["config"])
    if device is not None:
        cfg.device = str(device)
    dtype = resolve_dtype(cfg.dtype)
    resolved_device = device or resolve_device(cfg.device)
    torch.set_default_dtype(dtype)
    params = compute_physical_params(cfg)
    model = XPINNModel(cfg, params).to(device=resolved_device, dtype=dtype)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, cfg, checkpoint
