from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .geometry import FilmGeometry, PhysicalParams
from .networks import CVALModel


class Tee:
    def __init__(self, path: Path, stream: TextIO | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = stream or sys.stdout
        self.file = path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        self.stream.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class RunArtifacts:
    def __init__(self, output_dir: Path):
        self.root = output_dir
        self.logs = output_dir / "log"
        self.checkpoints = output_dir / "checkpoints"
        self.diagnostics = output_dir / "diagnostics"
        self.snapshots = self.diagnostics / "snapshots"
        self.figures = output_dir / "figures"
        self.comparison = output_dir / "comparison_results"
        self.isolines = output_dir / "isoline_results"
        for path in (
            self.root,
            self.logs,
            self.checkpoints,
            self.diagnostics,
            self.snapshots,
            self.figures,
            self.comparison,
            self.isolines,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def array_stats(values: np.ndarray | torch.Tensor) -> dict[str, float]:
    array = (
        values.detach().cpu().numpy()
        if isinstance(values, torch.Tensor)
        else np.asarray(values)
    )
    flat = array.reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "median": float(np.median(flat)),
        "p90_abs": float(np.percentile(np.abs(flat), 90)),
        "p99_abs": float(np.percentile(np.abs(flat), 99)),
        "l1": float(np.linalg.norm(flat, ord=1)),
        "l2": float(np.linalg.norm(flat, ord=2)),
    }


def gradient_stats(model: CVALModel) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for group_name, module in (
        ("pressure_net", model.pressure_net),
        ("gamma_net", model.gamma_net),
    ):
        norms = []
        maximum = 0.0
        nonfinite = 0
        count = 0
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            count += grad.numel()
            nonfinite += int((~torch.isfinite(grad)).sum().cpu())
            norms.append(float(torch.linalg.vector_norm(grad).cpu()))
            if grad.numel():
                maximum = max(maximum, float(grad.abs().max().cpu()))
        result[group_name] = {
            "global_l2": float(np.linalg.norm(norms)) if norms else 0.0,
            "max_abs": maximum,
            "nonfinite": float(nonfinite),
            "element_count": float(count),
        }
    return result


class HistoryRecorder:
    def __init__(self, artifacts: RunArtifacts):
        self.artifacts = artifacts
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def save(self) -> None:
        write_json(self.artifacts.logs / "history.json", self.records)
        if not self.records:
            return
        columns: list[str] = []
        for record in self.records:
            for key in record:
                if key not in columns and not isinstance(record[key], dict):
                    columns.append(key)
        with (self.artifacts.logs / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.records)

    def plot(self, dpi: int) -> None:
        if not self.records:
            return
        epochs = np.asarray([row["global_epoch"] for row in self.records])
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        series = [
            ("loss_total", "Training objective"),
            ("loss_cv", "Control-volume loss"),
            ("fb_p99", "FB P99"),
            ("learning_rate", "Learning rate"),
        ]
        for ax, (key, title) in zip(axes.flat, series):
            values = np.asarray([float(row.get(key, np.nan)) for row in self.records])
            valid = np.isfinite(values)
            if valid.any():
                ax.plot(epochs[valid], values[valid], linewidth=1.2)
                if key != "learning_rate" and np.all(values[valid] > 0):
                    ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("Global epoch")
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            self.artifacts.figures / "training_history.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


class TrainingDiagnostics:
    def __init__(self, artifacts: RunArtifacts):
        self.artifacts = artifacts
        self.snapshots: list[dict[str, Any]] = []

    def snapshot(
        self,
        stage_index: int,
        stage_name: str,
        global_epoch: int,
        metrics: dict[str, float],
        fields: dict[str, torch.Tensor],
        gradients: dict[str, dict[str, float]],
    ) -> None:
        payload = {
            "stage_index": stage_index,
            "stage_name": stage_name,
            "global_epoch": global_epoch,
            "metrics": metrics,
            "fields": {key: array_stats(value) for key, value in fields.items()},
            "gradients": gradients,
        }
        self.snapshots.append(payload)
        name = f"snapshot_{global_epoch:07d}_{stage_name}.json"
        write_json(self.artifacts.snapshots / name, payload)
        arrays = {
            key: value.detach().cpu().numpy()
            for key, value in fields.items()
        }
        np.savez_compressed(
            self.artifacts.snapshots / f"snapshot_{global_epoch:07d}_{stage_name}.npz",
            **arrays,
        )

    def finalize(self) -> None:
        write_json(self.artifacts.diagnostics / "training_summary.json", self.snapshots)
        lines = [
            "=" * 72,
            "  Torch CV-AL PINN Training Diagnostic Report",
            "=" * 72,
        ]
        if not self.snapshots:
            lines.append("No snapshots were recorded.")
        else:
            first = self.snapshots[0]
            last = self.snapshots[-1]
            lines.extend(
                [
                    f"Snapshots: {len(self.snapshots)}",
                    f"First: epoch={first['global_epoch']} stage={first['stage_name']}",
                    f"Last:  epoch={last['global_epoch']} stage={last['stage_name']}",
                    "",
                    "Last fixed-validation metrics:",
                ]
            )
            for key, value in last["metrics"].items():
                lines.append(f"  {key:<32s} {value:.8e}")
            lines.extend(["", "Last field P99 absolute magnitudes:"])
            for key, values in last["fields"].items():
                lines.append(f"  {key:<20s} {values['p99_abs']:.8e}")
            lines.extend(["", "Last gradient norms:"])
            for key, values in last["gradients"].items():
                lines.append(
                    f"  {key:<20s} L2={values['global_l2']:.8e} "
                    f"max={values['max_abs']:.8e} nonfinite={int(values['nonfinite'])}"
                )
        (self.artifacts.diagnostics / "diagnostic_report.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def plot_solution_fields(
    model: CVALModel,
    geometry: FilmGeometry,
    params: PhysicalParams,
    xi_ratio: float,
    output_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    dpi: int,
    resolution: int = 201,
) -> None:
    radius = np.linspace(params.r_min, params.r_max, resolution)
    theta = np.linspace(params.theta_min, params.theta_max, resolution)
    rr, tt = np.meshgrid(radius, theta, indexing="ij")
    coords_np = np.column_stack([rr.reshape(-1), tt.reshape(-1)])
    coords = torch.as_tensor(coords_np, device=device, dtype=dtype)
    with torch.no_grad():
        output = model(coords).cpu().numpy()
        film = geometry.film_thickness(coords, xi_ratio).cpu().numpy()
    pressure = output[:, 0].reshape(rr.shape)
    gamma = output[:, 1].reshape(rr.shape)
    film_grid = film.reshape(rr.shape)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, values, title in (
        (axes[0], pressure, "Pressure p"),
        (axes[1], gamma, "Cavitation fraction gamma"),
        (axes[2], film_grid, "Film thickness H"),
    ):
        mesh = ax.pcolormesh(rr, tt, values, shading="auto", cmap="RdYlBu_r")
        fig.colorbar(mesh, ax=ax, pad=0.02)
        ax.set_title(title)
        ax.set_xlabel("R")
        ax.set_ylabel("theta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
