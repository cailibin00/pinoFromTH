from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageConfig:
    name: str
    lambda_ratio: float
    xi_ratio: float
    epsilon_fb: float
    gamma_enabled: bool
    adam_epochs: int
    lbfgs_steps: int = 0
    active_gate_enabled: bool = True
    active_pressure_ratio: float = 0.25
    active_gate_tau: float = 0.05


def default_stages() -> list[StageConfig]:
    return [
        StageConfig("full_film_010", 0.10, 4.0, 1e-2, False, 2_000),
        StageConfig("cavitation_025", 0.25, 4.0, 1e-2, True, 3_000, active_gate_tau=0.08),
        StageConfig("cavitation_050", 0.50, 4.0, 1e-3, True, 3_000, active_gate_tau=0.05),
        StageConfig("cavitation_075", 0.75, 2.0, 1e-4, True, 3_000, active_gate_tau=0.03),
        StageConfig("true_physics", 1.00, 1.0, 1e-5, True, 5_000, 200, active_gate_tau=0.02),
    ]


@dataclass
class ExperimentConfig:
    # Physical geometry and operating point
    r_i: float = 47.0e-3
    r_o: float = 52.0e-3
    h_i: float = 3.0e-6
    periods: int = 6
    groove_start_ratio: float = 1.043
    groove_end_ratio: float = 1.106 * 2.0
    spiral_angle_deg: float = 3.0
    groove_depth_ratio: float = 3.0
    groove_width_ratio: float = 0.5
    pressure_inner_pa: float = 0.1e6
    pressure_outer_ratio: float = 1.5
    viscosity_pa_s: float = 8.0e-4
    speed_rpm: float = 6_000.0

    # Model
    hidden_width: int = 128
    hidden_layers: int = 4
    fourier_modes: int = 8
    softplus_beta: float = 20.0
    dtype: str = "float64"
    device: str = "cuda:5"

    # Control-volume grids
    train_cells_r: int = 48
    train_cells_theta: int = 72
    validation_cells_r: int = 32
    validation_cells_theta: int = 48
    gauss_order: int = 3
    cell_batch_size: int = 256

    # Fixed physical preconditioning scales
    pressure_ref: float = 0.1
    film_ref: float = 2.0
    gamma_active_threshold: float = 1.0e-6

    # Optimisation
    seed: int = 20260718
    peak_lr: float = 1e-3
    min_lr: float = 1e-6
    warmup_epochs: int = 500
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    gradient_clip_norm: float = 100.0
    validation_interval: int = 250
    log_interval: int = 50
    al_update_interval: int = 250
    al_mu_initial: float = 10.0
    al_mu_growth: float = 2.0
    al_mu_max: float = 1.0e6
    al_progress_ratio: float = 0.9
    reset_al_each_stage: bool = True
    best_global_flux_tolerance: float = 1.0e-5
    best_boundary_tolerance: float = 1.0e-10
    best_periodic_tolerance: float = 1.0e-10
    best_min_gamma_active_fraction: float = 1.0e-3
    stages: list[StageConfig] = field(default_factory=default_stages)

    # Sampling emphasis for the A-min active-set run. The base grid stays fixed;
    # only minibatch probabilities change.
    stratified_sampling: bool = True
    transition_sampling_weight: float = 4.0
    radial_boundary_sampling_weight: float = 1.5
    periodic_seam_sampling_weight: float = 1.0
    active_sampling_weight: float = 8.0
    active_sampling_interval: int = 50

    # Outputs and automatic post-training evaluation
    output_dir: str = "output_torch_cv_al"
    fem_pressure_path: str = "p_FBNS.txt"
    fem_gamma_path: str = "g_FBNS.txt"
    auto_run_evaluations: bool = True
    prediction_batch_size: int = 16_384
    dpi: int = 300
    iou_thresholds: list[float] = field(
        default_factory=lambda: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        data = dict(values)
        data["stages"] = [
            item if isinstance(item, StageConfig) else StageConfig(**item)
            for item in data.get("stages", default_stages())
        ]
        return cls(**data)

    def resolve_path(self, project_root: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    @classmethod
    def smoke(cls, output_dir: str = "output_torch_smoke") -> "ExperimentConfig":
        cfg = cls(
            hidden_width=24,
            hidden_layers=2,
            fourier_modes=3,
            train_cells_r=4,
            train_cells_theta=6,
            validation_cells_r=4,
            validation_cells_theta=6,
            gauss_order=2,
            cell_batch_size=8,
            warmup_epochs=1,
            validation_interval=1,
            log_interval=1,
            al_update_interval=1,
            output_dir=output_dir,
            auto_run_evaluations=False,
            dpi=80,
        )
        cfg.stages = [
            StageConfig("smoke_full_film", 0.1, 4.0, 1e-2, False, 2),
            StageConfig("smoke_true", 1.0, 1.0, 1e-3, True, 2, active_gate_tau=0.08),
        ]
        return cfg
