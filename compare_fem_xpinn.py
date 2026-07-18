from __future__ import annotations

import argparse
from pathlib import Path

from torch_xpinn.config import XPINNConfig
from torch_xpinn.evaluation import load_checkpoint, run_final_evaluation
from torch_xpinn.trainer import resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained Torch XPINN")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output_xpinn/checkpoints/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    device = resolve_device(args.device)
    model, cfg, _ = load_checkpoint(checkpoint, device)
    cfg.device = args.device
    output = args.output_dir or checkpoint.parent.parent / "evaluation"
    run_final_evaluation(model, cfg, root, output, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
