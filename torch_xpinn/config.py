from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class XPINNConfig:
    # Geometry and operating point
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
    hidden_width: int = 192
    hidden_layers: int = 5
    fourier_modes: int = 8
    spiral_feature_modes: int = 2
    pressure_scale: float = 0.1
    softplus_beta: float = 20.0
    hard_pressure_boundary: bool = True
    dtype: str = "float64"
    device: str = "auto"

    # Hard partition
    thin_film: float = 1.0
    groove_film: float = 4.0
    interface_points: int = 1024
    thin_points: int = 4096
    groove_points: int = 4096
    boundary_points: int = 1024

    # Three public loss terms. Reynolds includes the regional residual and
    # conservative interface coupling required by discontinuous film thickness.
    reynolds_weight: float = 1.0
    jfo_weight: float = 10.0
    boundary_weight: float = 1.0e3
    reynolds_interface_pressure_weight: float = 1.0
    reynolds_interface_flux_weight: float = 1.0e-3

    # Optimisation
    seed: int = 20260718
    epochs: int = 20_000
    learning_rate: float = 1.0e-3
    min_learning_rate: float = 1.0e-5
    log_interval: int = 100
    checkpoint_interval: int = 1_000
    gradient_clip_norm: float = 1.0
    use_tqdm: bool = True

    # Evaluation
    prediction_batch_size: int = 65_536
    iou_thresholds: tuple[float, ...] = field(
        default_factory=lambda: (1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4)
    )
    dpi: int = 180
    evaluate_after_training: bool = True

    # Outputs
    output_dir: str = "output_xpinn"
    fem_pressure_path: str = "p_FBNS.txt"
    fem_gamma_path: str = "g_FBNS.txt"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "XPINNConfig":
        valid_keys = set(cls.__dataclass_fields__)
        clean = {key: value for key, value in values.items() if key in valid_keys}
        if "iou_thresholds" in clean:
            clean["iou_thresholds"] = tuple(clean["iou_thresholds"])
        return cls(**clean)

    def resolve_path(self, project_root: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path
