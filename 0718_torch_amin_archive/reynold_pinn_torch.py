from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from torch_pinn.config import ExperimentConfig
from torch_pinn.diagnostics import Tee
from torch_pinn.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the control-volume augmented-Lagrangian Torch PINN."
    )
    parser.add_argument("--config", type=Path, help="Experiment JSON configuration")
    parser.add_argument("--output-dir", help="Override the output directory")
    parser.add_argument("--device", help="Override device, e.g. cuda, cuda:0, cpu")
    parser.add_argument(
        "--smoke", action="store_true", help="Run a tiny end-to-end verification"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip the two automatic FEM evaluations",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.smoke:
        cfg = ExperimentConfig.smoke(args.output_dir or "output_torch_smoke")
    elif args.config:
        with args.config.open("r", encoding="utf-8") as handle:
            cfg = ExperimentConfig.from_dict(json.load(handle))
    else:
        cfg = ExperimentConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.device = args.device
    if args.skip_evaluation:
        cfg.auto_run_evaluations = False
    return cfg


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    cfg = build_config(args)
    output_dir = cfg.resolve_path(project_root, cfg.output_dir)
    tee = Tee(output_dir / "log" / "train.log", sys.stdout)
    original_stdout = sys.stdout
    sys.stdout = tee
    try:
        Trainer(cfg, project_root).run()
    finally:
        sys.stdout = original_stdout
        tee.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
