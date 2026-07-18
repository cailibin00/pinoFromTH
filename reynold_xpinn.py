from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from torch_xpinn.config import XPINNConfig
from torch_xpinn.geometry import HardGrooveGeometry, compute_physical_params
from torch_xpinn.logging import format_loss_block
from torch_xpinn.networks import XPINNModel
from torch_xpinn.physics import interface_losses


def resolve_dtype(value: str) -> torch.dtype:
    choices = {"float32": torch.float32, "float64": torch.float64}
    if value not in choices:
        raise ValueError(f"Unsupported dtype: {value}")
    return choices[value]


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hard-partition XPINN scaffold for spiral-groove Reynolds cavitation."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--inspect-geometry", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> XPINNConfig:
    if args.config:
        cfg = XPINNConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    else:
        cfg = XPINNConfig()
    if args.device:
        cfg.device = args.device
    return cfg


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    dtype = resolve_dtype(cfg.dtype)
    device = resolve_device(cfg.device)
    torch.set_default_dtype(dtype)

    geometry = HardGrooveGeometry(cfg)
    params = compute_physical_params(cfg)
    model = XPINNModel(cfg, params).to(device=device, dtype=dtype)
    points, normals = geometry.interface_points(
        cfg.interface_points, device=device, dtype=dtype
    )
    losses = interface_losses(model, geometry, points, normals, create_graph=True)
    weights = {
        "interface_pressure": cfg.interface_pressure_weight,
        "interface_flux": cfg.interface_flux_weight,
    }

    print("Hard-partition XPINN scaffold")
    print(f"device={device}, dtype={dtype}")
    print(f"lambda={params.lambda_value:.6f}")
    print(f"domain R=[{params.r_min:.10f}, {params.r_max:.10f}] theta=[0, {params.theta_max:.10f}]")
    print(f"groove R=[{params.groove_r_min:.10f}, {params.groove_r_max:.10f}]")
    print(f"interface_points={len(points)}")
    print(
        format_loss_block(
            stage="geometry_inspect",
            epoch=0,
            losses=losses,
            weights=weights,
        )
    )
    if not args.inspect_geometry:
        print("Training is not wired yet. Next step: add region interior CV/PDE losses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
