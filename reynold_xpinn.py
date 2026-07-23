from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from torch_xpinn.config import XPINNConfig
from torch_xpinn.geometry import HardGrooveGeometry, Region, compute_physical_params
from torch_xpinn.logging import format_loss_block
from torch_xpinn.networks import XPINNModel
from torch_xpinn.physics import region_jfo_loss, region_reynolds_loss
from torch_xpinn.evaluation import load_checkpoint, run_final_evaluation
from torch_xpinn.trainer import XPINNTrainer, resolve_device, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hard-partition XPINN scaffold for spiral-groove Reynolds cavitation."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--eval-output-dir")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--inspect-geometry", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> XPINNConfig:
    if args.config:
        cfg = XPINNConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    else:
        cfg = XPINNConfig()
    if args.device:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.no_evaluate:
        cfg.evaluate_after_training = False
    if args.no_tqdm:
        cfg.use_tqdm = False
    if args.log_interval is not None:
        cfg.log_interval = args.log_interval
    if args.smoke:
        cfg.epochs = args.epochs if args.epochs is not None else 2
        cfg.hidden_width = 32
        cfg.hidden_layers = 2
        cfg.fourier_modes = 4
        cfg.thin_points = 64
        cfg.groove_points = 64
        cfg.boundary_points = 32
        cfg.log_interval = 1
        cfg.checkpoint_interval = 0
    return cfg


def evaluate_checkpoint(
    checkpoint: Path,
    cfg: XPINNConfig,
    output_dir: Path | None,
) -> None:
    root = Path(__file__).resolve().parent
    checkpoint_path = checkpoint if checkpoint.is_absolute() else root / checkpoint
    device = resolve_device(cfg.device)
    model, loaded_cfg, _ = load_checkpoint(checkpoint_path, device)
    loaded_cfg.device = cfg.device
    eval_dir = output_dir or checkpoint_path.parent.parent / "evaluation"
    run_final_evaluation(model, loaded_cfg, root, eval_dir, device)


def inspect_geometry(cfg: XPINNConfig) -> None:
    dtype = resolve_dtype(cfg.dtype)
    device = resolve_device(cfg.device)
    torch.set_default_dtype(dtype)

    geometry = HardGrooveGeometry(cfg)
    params = compute_physical_params(cfg)
    model = XPINNModel(cfg, params).to(device=device, dtype=dtype)
    thin_points = geometry.sample_region(
        Region.THIN, cfg.thin_points, device, dtype
    )
    groove_points = geometry.sample_region(
        Region.GROOVE, cfg.groove_points, device, dtype
    )
    losses = {
        "reynolds": torch.stack(
            [
                region_reynolds_loss(model, geometry, thin_points, Region.THIN),
                region_reynolds_loss(model, geometry, groove_points, Region.GROOVE),
            ]
        ).mean(),
        "jfo": torch.stack(
            [
                region_jfo_loss(model, thin_points, Region.THIN),
                region_jfo_loss(model, groove_points, Region.GROOVE),
            ]
        ).mean(),
    }
    weights = {
        "reynolds": cfg.reynolds_weight,
        "jfo": cfg.jfo_weight,
    }

    print("Hard-partition XPINN scaffold")
    print(f"device={device}, dtype={dtype}")
    print(f"lambda={params.lambda_value:.6f}")
    print(f"domain R=[{params.r_min:.10f}, {params.r_max:.10f}] theta=[0, {params.theta_max:.10f}]")
    print(f"groove R=[{params.groove_r_min:.10f}, {params.groove_r_max:.10f}]")
    print(f"thin_points={len(thin_points)}")
    print(f"groove_points={len(groove_points)}")
    print(
        format_loss_block(
            stage="geometry_inspect",
            epoch=0,
            losses=losses,
            weights=weights,
        )
    )


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    if args.inspect_geometry:
        inspect_geometry(cfg)
        return 0
    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint")
        evaluate_checkpoint(args.checkpoint, cfg, Path(args.eval_output_dir) if args.eval_output_dir else None)
        return 0

    trainer = XPINNTrainer(cfg, Path(__file__).resolve().parent)
    summary = trainer.train()
    print(
        "Training finished: "
        f"best={summary['best']:.8e} at epoch={summary['best_epoch']}, "
        f"final={summary['final']:.8e}"
    )
    print(f"output_dir={summary['output_dir']}")
    print(f"final_checkpoint={summary['final_checkpoint']}")
    if cfg.evaluate_after_training:
        checkpoint = Path(summary["best_checkpoint"])
        eval_dir = (
            Path(args.eval_output_dir)
            if args.eval_output_dir
            else Path(summary["output_dir"]) / "evaluation_best"
        )
        evaluate_checkpoint(checkpoint, cfg, eval_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
