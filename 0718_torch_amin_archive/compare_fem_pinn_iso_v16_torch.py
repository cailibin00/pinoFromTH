from __future__ import annotations

import argparse
from pathlib import Path

from torch_pinn.checkpoint import load_checkpoint
from torch_pinn.evaluation import prepare_evaluation, run_isoline_evaluation
from torch_pinn.trainer import resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Torch CV-AL PINN isoline analysis")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output_torch_cv_al/checkpoints/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    device = resolve_device(args.device)
    model, cfg, _ = load_checkpoint(checkpoint, device)
    output = args.output_dir or checkpoint.parent.parent / "isoline_results"
    bundle = prepare_evaluation(model, cfg, root, device)
    metrics = run_isoline_evaluation(
        model, cfg, root, output, device, bundle=bundle
    )
    print(f"Saved {len(metrics)} isoline metrics and figures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
