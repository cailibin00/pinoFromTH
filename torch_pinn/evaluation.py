from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from .config import ExperimentConfig
from .networks import CVALModel


@dataclass
class EvaluationBundle:
    radius_points: np.ndarray
    theta_points: np.ndarray
    pressure_fem: np.ndarray
    gamma_fem: np.ndarray
    pressure_pinn: np.ndarray
    gamma_pinn: np.ndarray
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
            self.pressure_pinn.reshape(shape),
            self.gamma_pinn.reshape(shape),
        )


def load_fem(pressure_path: Path, gamma_path: Path) -> tuple[np.ndarray, ...]:
    pressure_raw = np.loadtxt(pressure_path)
    gamma_raw = np.loadtxt(gamma_path)
    if pressure_raw.shape[1] < 3 or gamma_raw.shape[1] < 3:
        raise ValueError("FEM files must contain R, theta and field-value columns")
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
    model: CVALModel,
    coords: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = next(model.parameters()).dtype
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            batch = torch.as_tensor(
                coords[start : start + batch_size], device=device, dtype=dtype
            )
            outputs.append(model(batch).cpu().numpy())
    values = np.concatenate(outputs, axis=0)
    return values[:, 0], values[:, 1]


def prepare_evaluation(
    model: CVALModel,
    cfg: ExperimentConfig,
    project_root: Path,
    device: torch.device,
) -> EvaluationBundle:
    p_path = cfg.resolve_path(project_root, cfg.fem_pressure_path)
    g_path = cfg.resolve_path(project_root, cfg.fem_gamma_path)
    radius, theta, p_fem, g_fem, r_unique, t_unique = load_fem(p_path, g_path)
    coords = np.column_stack([radius, theta])
    p_pinn, g_pinn = predict_in_batches(
        model, coords, cfg.prediction_batch_size, device
    )
    return EvaluationBundle(
        radius,
        theta,
        p_fem,
        g_fem,
        p_pinn,
        g_pinn,
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
    p_pred, g_pred = bundle.pressure_pinn, bundle.gamma_pinn
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
    cavitation = g_ref > 1e-6
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
    metrics["cavitation_IoU"] = metrics[_threshold_key("cavitation_IoU", 1e-6)]
    metrics["cavitation_Dice"] = metrics[
        _threshold_key("cavitation_Dice", 1e-6)
    ]
    return metrics


def save_metrics(metrics: dict[str, float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as handle:
        handle.write("FEM vs Torch CV-AL PINN metrics\n" + "=" * 64 + "\n")
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
            bundle.pressure_pinn,
            bundle.gamma_pinn,
        ]
    )
    np.savetxt(
        output_dir / "fem_pinn_fields.txt",
        columns,
        fmt="%.10e",
        header="R theta P_FEM gamma_FEM P_PINN gamma_PINN",
    )
    np.savez_compressed(
        output_dir / "fem_pinn_fields.npz",
        radius=bundle.radius_points,
        theta=bundle.theta_points,
        pressure_fem=bundle.pressure_fem,
        gamma_fem=bundle.gamma_fem,
        pressure_pinn=bundle.pressure_pinn,
        gamma_pinn=bundle.gamma_pinn,
    )


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _mesh(bundle: EvaluationBundle) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(bundle.radius_unique, bundle.theta_unique, indexing="ij")


def plot_field_comparison(bundle: EvaluationBundle, output_dir: Path, dpi: int) -> None:
    p_fem, g_fem, p_pinn, g_pinn = bundle.grids()
    rr, tt = _mesh(bundle)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for values, title, ax in (
        (p_fem, "FEM pressure", axes[0, 0]),
        (p_pinn, "PINN pressure", axes[0, 1]),
        (g_fem, "FEM gamma", axes[1, 0]),
        (g_pinn, "PINN gamma", axes[1, 1]),
    ):
        mesh = ax.pcolormesh(rr, tt, values, shading="auto", cmap="RdYlBu_r")
        fig.colorbar(mesh, ax=ax, pad=0.02)
        ax.set_title(title)
        ax.set_xlabel("R")
        ax.set_ylabel("theta")
    fig.tight_layout()
    _save(fig, output_dir / "fig1_field_comparison.png", dpi)


def plot_error_maps(bundle: EvaluationBundle, output_dir: Path, dpi: int) -> None:
    p_fem, g_fem, p_pinn, g_pinn = bundle.grids()
    rr, tt = _mesh(bundle)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for values, title, ax in (
        (np.abs(p_fem - p_pinn), "Absolute pressure error", axes[0]),
        (np.abs(g_fem - g_pinn), "Absolute gamma error", axes[1]),
    ):
        mesh = ax.pcolormesh(rr, tt, values, shading="auto", cmap="hot_r")
        fig.colorbar(mesh, ax=ax, pad=0.02)
        ax.set_title(title)
        ax.set_xlabel("R")
        ax.set_ylabel("theta")
    fig.tight_layout()
    _save(fig, output_dir / "fig2_error_maps.png", dpi)


def plot_cavitation_boundary(
    bundle: EvaluationBundle, output_dir: Path, dpi: int, threshold: float = 1e-6
) -> None:
    _, g_fem, _, g_pinn = bundle.grids()
    rr, tt = _mesh(bundle)
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(rr, tt, g_pinn, shading="auto", cmap="Blues", alpha=0.55)
    fig.colorbar(mesh, ax=ax, label="PINN gamma", pad=0.02)
    ax.contour(rr, tt, g_fem, levels=[threshold], colors="red", linewidths=1.8)
    ax.contour(
        rr, tt, g_pinn, levels=[threshold], colors="blue", linewidths=1.8,
        linestyles="--"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="red", label="FEM"),
            Line2D([0], [0], color="blue", linestyle="--", label="PINN"),
        ]
    )
    ax.set_title(f"Cavitation boundary at gamma={threshold:.0e}")
    ax.set_xlabel("R")
    ax.set_ylabel("theta")
    fig.tight_layout()
    _save(fig, output_dir / "fig3_cavitation_boundary.png", dpi)


def plot_pressure_overlay(bundle: EvaluationBundle, output_dir: Path, dpi: int) -> None:
    p_fem, _, p_pinn, _ = bundle.grids()
    rr, tt = _mesh(bundle)
    levels = np.linspace(min(p_fem.min(), p_pinn.min()), max(p_fem.max(), p_pinn.max()), 15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contour(rr, tt, p_fem, levels=levels, colors="red", linewidths=1.1)
    ax.contour(
        rr, tt, p_pinn, levels=levels, colors="blue", linewidths=1.1,
        linestyles="--"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="red", label="FEM"),
            Line2D([0], [0], color="blue", linestyle="--", label="PINN"),
        ]
    )
    ax.set_title("Pressure contour overlay")
    ax.set_xlabel("R")
    ax.set_ylabel("theta")
    fig.tight_layout()
    _save(fig, output_dir / "fig4_pressure_contour_overlay.png", dpi)


def plot_profiles(bundle: EvaluationBundle, output_dir: Path, dpi: int) -> None:
    p_fem, g_fem, p_pinn, g_pinn = bundle.grids()
    i = len(bundle.radius_unique) // 2
    j = len(bundle.theta_unique) // 2
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    items = [
        (axes[0, 0], bundle.theta_unique, p_fem[i], p_pinn[i], "p(theta)"),
        (axes[0, 1], bundle.theta_unique, g_fem[i], g_pinn[i], "gamma(theta)"),
        (axes[1, 0], bundle.radius_unique, p_fem[:, j], p_pinn[:, j], "p(R)"),
        (axes[1, 1], bundle.radius_unique, g_fem[:, j], g_pinn[:, j], "gamma(R)"),
    ]
    for ax, x, reference, prediction, title in items:
        ax.plot(x, reference, "r-", label="FEM")
        ax.plot(x, prediction, "b--", label="PINN")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, output_dir / "fig5_profiles.png", dpi)


def plot_periodic_check(bundle: EvaluationBundle, output_dir: Path, dpi: int) -> None:
    p_fem, g_fem, p_pinn, g_pinn = bundle.grids()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, reference, prediction, title in (
        (axes[0], p_fem, p_pinn, "Pressure periodic seam"),
        (axes[1], g_fem, g_pinn, "Gamma periodic seam"),
    ):
        ax.plot(bundle.radius_unique, reference[:, 0], "r-", label="FEM theta=0")
        ax.plot(bundle.radius_unique, reference[:, -1], "r--", label="FEM theta=max")
        ax.plot(bundle.radius_unique, prediction[:, 0], "b-", label="PINN theta=0")
        ax.plot(bundle.radius_unique, prediction[:, -1], "b--", label="PINN theta=max")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, output_dir / "fig6_periodic_check.png", dpi)


def plot_scatter(bundle: EvaluationBundle, output_dir: Path, dpi: int) -> None:
    rng = np.random.default_rng(42)
    sample = rng.choice(len(bundle.pressure_fem), min(5000, len(bundle.pressure_fem)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, reference, prediction, title in (
        (axes[0], bundle.pressure_fem[sample], bundle.pressure_pinn[sample], "Pressure"),
        (axes[1], bundle.gamma_fem[sample], bundle.gamma_pinn[sample], "Gamma"),
    ):
        ax.scatter(reference, prediction, s=4, alpha=0.35)
        low = min(reference.min(), prediction.min())
        high = max(reference.max(), prediction.max())
        ax.plot([low, high], [low, high], "r--")
        ax.set_title(title)
        ax.set_xlabel("FEM")
        ax.set_ylabel("PINN")
    fig.tight_layout()
    _save(fig, output_dir / "fig7_scatter_correlation.png", dpi)


def plot_complementarity(bundle: EvaluationBundle, output_dir: Path, dpi: int) -> None:
    rng = np.random.default_rng(7)
    sample = rng.choice(len(bundle.pressure_fem), min(8000, len(bundle.pressure_fem)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pressure, gamma, title in (
        (axes[0], bundle.pressure_fem[sample], bundle.gamma_fem[sample], "FEM"),
        (axes[1], bundle.pressure_pinn[sample], bundle.gamma_pinn[sample], "PINN"),
    ):
        product = pressure * gamma
        scatter = ax.scatter(pressure, gamma, c=product, s=4, alpha=0.4, cmap="hot_r")
        fig.colorbar(scatter, ax=ax, label="p*gamma")
        ax.set_title(f"{title}: max={product.max():.2e}")
        ax.set_xlabel("p")
        ax.set_ylabel("gamma")
    fig.tight_layout()
    _save(fig, output_dir / "fig8_jfo_complement.png", dpi)


def run_final_evaluation(
    model: CVALModel,
    cfg: ExperimentConfig,
    project_root: Path,
    output_dir: Path,
    device: torch.device,
    bundle: EvaluationBundle | None = None,
) -> tuple[dict[str, float], EvaluationBundle]:
    print("[Evaluation 1/2] FEM field and metric comparison")
    data = bundle or prepare_evaluation(model, cfg, project_root, device)
    metrics = compute_metrics(data, cfg.iou_thresholds)
    save_metrics(metrics, output_dir)
    save_field_data(data, output_dir)
    plot_field_comparison(data, output_dir, cfg.dpi)
    plot_error_maps(data, output_dir, cfg.dpi)
    plot_cavitation_boundary(data, output_dir, cfg.dpi)
    plot_pressure_overlay(data, output_dir, cfg.dpi)
    plot_profiles(data, output_dir, cfg.dpi)
    plot_periodic_check(data, output_dir, cfg.dpi)
    plot_scatter(data, output_dir, cfg.dpi)
    plot_complementarity(data, output_dir, cfg.dpi)
    for key in ("P_rel_L2", "G_rel_L2", "cavitation_IoU"):
        print(f"  {key:<28s} {metrics[key]:.8e}")
    return metrics, data


def _find_crossings(
    field: np.ndarray, radius: np.ndarray, theta: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_radius, entry, exit_ = [], [], []
    for i, row in enumerate(field):
        above = row >= level
        crossing = np.where(np.diff(above.astype(np.int8)) != 0)[0]
        if len(crossing) < 2:
            continue

        def interpolate(index: int) -> float:
            t0, t1 = theta[index], theta[index + 1]
            g0, g1 = row[index], row[index + 1]
            if abs(g1 - g0) < 1e-14:
                return 0.5 * (t0 + t1)
            return float(t0 + (level - g0) / (g1 - g0) * (t1 - t0))

        valid_radius.append(radius[i])
        entry.append(interpolate(int(crossing[0])))
        exit_.append(interpolate(int(crossing[-1])))
    return np.asarray(valid_radius), np.asarray(entry), np.asarray(exit_)


def run_isoline_evaluation(
    model: CVALModel,
    cfg: ExperimentConfig,
    project_root: Path,
    output_dir: Path,
    device: torch.device,
    bundle: EvaluationBundle | None = None,
) -> dict[str, object]:
    print("[Evaluation 2/2] Multi-threshold IoU and cavitation isolines")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = bundle or prepare_evaluation(model, cfg, project_root, device)
    _, g_fem, _, g_pinn = data.grids()
    rr, tt = _mesh(data)
    threshold_rows = []
    eps = np.finfo(float).eps
    for threshold in cfg.iou_thresholds:
        reference = data.gamma_fem > threshold
        prediction = data.gamma_pinn > threshold
        intersection = np.logical_and(reference, prediction).sum()
        union = np.logical_or(reference, prediction).sum()
        threshold_rows.append(
            {
                "threshold": threshold,
                "iou": float(intersection / (union + eps)),
                "dice": float(
                    2.0 * intersection
                    / (reference.sum() + prediction.sum() + eps)
                ),
                "fem_fraction": float(reference.mean()),
                "pinn_fraction": float(prediction.mean()),
            }
        )
    with (output_dir / "multithreshold_iou.csv").open("w", encoding="utf-8") as handle:
        handle.write("threshold,iou,dice,fem_fraction,pinn_fraction\n")
        for row in threshold_rows:
            handle.write(
                f"{row['threshold']:.8e},{row['iou']:.8e},{row['dice']:.8e},"
                f"{row['fem_fraction']:.8e},{row['pinn_fraction']:.8e}\n"
            )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    thresholds = np.asarray([row["threshold"] for row in threshold_rows])
    axes[0].semilogx(thresholds, [row["iou"] for row in threshold_rows], "o-", label="IoU")
    axes[0].semilogx(thresholds, [row["dice"] for row in threshold_rows], "s--", label="Dice")
    axes[0].set_xlabel("gamma threshold")
    axes[0].set_ylabel("score")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].semilogx(
        thresholds, [row["fem_fraction"] for row in threshold_rows], "r-", label="FEM"
    )
    axes[1].semilogx(
        thresholds, [row["pinn_fraction"] for row in threshold_rows], "b--", label="PINN"
    )
    axes[1].set_xlabel("gamma threshold")
    axes[1].set_ylabel("cavitation area fraction")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    _save(fig, output_dir / "multithreshold_iou.png", cfg.dpi)

    positive = data.gamma_pinn[data.gamma_pinn > 1e-6]
    if positive.size:
        level = float(0.5 * (positive.min() + positive.max()))
    else:
        level = 1e-6
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(rr, tt, g_pinn, shading="auto", cmap="RdYlBu_r")
    fig.colorbar(mesh, ax=ax, label="PINN gamma")
    ax.contour(rr, tt, g_fem, levels=[level], colors="red", linewidths=2.0)
    ax.contour(
        rr, tt, g_pinn, levels=[level], colors="blue", linewidths=2.0,
        linestyles="--"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="red", label="FEM"),
            Line2D([0], [0], color="blue", linestyle="--", label="PINN"),
        ]
    )
    ax.set_title(f"Cavitation isoline gamma={level:.4g}")
    ax.set_xlabel("R")
    ax.set_ylabel("theta")
    fig.tight_layout()
    _save(fig, output_dir / "cavitation_isoline_overlay.png", cfg.dpi)

    r_fem, fem_entry, fem_exit = _find_crossings(
        g_fem, data.radius_unique, data.theta_unique, level
    )
    r_pinn, pinn_entry, pinn_exit = _find_crossings(
        g_pinn, data.radius_unique, data.theta_unique, level
    )
    report: dict[str, object] = {
        "isoline_level": level,
        "thresholds": threshold_rows,
        "matched_radius_count": 0,
    }
    common = np.intersect1d(np.round(r_fem, 10), np.round(r_pinn, 10))
    if common.size:
        fem_index = [int(np.argmin(np.abs(r_fem - value))) for value in common]
        pinn_index = [int(np.argmin(np.abs(r_pinn - value))) for value in common]
        entry_error = pinn_entry[pinn_index] - fem_entry[fem_index]
        exit_error = pinn_exit[pinn_index] - fem_exit[fem_index]
        report.update(
            {
                "matched_radius_count": int(common.size),
                "entry_error_mean": float(entry_error.mean()),
                "entry_error_max_abs": float(np.abs(entry_error).max()),
                "exit_error_mean": float(exit_error.mean()),
                "exit_error_max_abs": float(np.abs(exit_error).max()),
            }
        )
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(entry_error * 1e3, common, "b-", label="Entry")
        ax.plot(exit_error * 1e3, common, "r--", label="Exit")
        ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
        ax.set_xlabel("theta error x 1e-3 rad (PINN-FEM)")
        ax.set_ylabel("R")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, output_dir / "cavitation_isoline_position_error.png", cfg.dpi)
    (output_dir / "isoline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def run_all_evaluations(
    model: CVALModel,
    cfg: ExperimentConfig,
    project_root: Path,
    comparison_dir: Path,
    isoline_dir: Path,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, object]]:
    bundle = prepare_evaluation(model, cfg, project_root, device)
    metrics, _ = run_final_evaluation(
        model, cfg, project_root, comparison_dir, device, bundle=bundle
    )
    isolines = run_isoline_evaluation(
        model, cfg, project_root, isoline_dir, device, bundle=bundle
    )
    return metrics, isolines
