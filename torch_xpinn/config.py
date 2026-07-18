from __future__ import annotations

from dataclasses import asdict, dataclass
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
    hidden_width: int = 128
    hidden_layers: int = 4
    fourier_modes: int = 8
    softplus_beta: float = 20.0
    dtype: str = "float64"
    device: str = "auto"

    # Hard partition
    thin_film: float = 1.0
    groove_film: float = 4.0
    interface_points: int = 512
    interior_points: int = 4096

    # Loss weights for the fixed film-thickness interface.
    interface_pressure_weight: float = 1.0
    interface_flux_weight: float = 1.0
    boundary_weight: float = 10.0
    periodic_weight: float = 1.0

    # Outputs
    output_dir: str = "output_xpinn"
    fem_pressure_path: str = "p_FBNS.txt"
    fem_gamma_path: str = "g_FBNS.txt"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "XPINNConfig":
        return cls(**dict(values))

    def resolve_path(self, project_root: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path
