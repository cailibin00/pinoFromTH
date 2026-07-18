# Torch CV-AL PINN

`0504/` is now historical reference material. The active Torch implementation is a
new model and training formulation; it does not import or reuse the old TensorFlow
network, PIKAN, PCGrad, RAD, shared trunk, or second-order strong residual.

## Run

```powershell
python reynold_pinn_torch.py
```

The default run trains five continuation stages with Adam, finishes the true-physics
stage with full-grid L-BFGS, restores the best checkpoint selected on a fixed control-
volume validation grid, and automatically runs both FEM evaluation pipelines.

Useful development commands:

```powershell
python reynold_pinn_torch.py --smoke
python reynold_pinn_torch.py --config experiment.json
python reynold_pinn_torch.py --device cuda:0
python -m unittest tests.test_torch_cv_al -v
```

`--skip-evaluation` exists only for debugging. A normal completed training run does
not require either comparison script to be launched manually.

## Outputs

The default directory is `output_torch_cv_al/`:

- `checkpoints/best.pt`: fixed-validation best during the true-physics stage.
- `checkpoints/final_best_restored.pt`: the same best weights after final restore.
- `diagnostics/snapshots/*.json`: field, residual, and gradient statistics.
- `diagnostics/snapshots/*.npz`: raw validation arrays for later analysis.
- `diagnostics/post_training/`: full best-model conservation and JFO metrics.
- `log/history.csv` and `history.json`: training and validation history.
- `figures/`: training history and best solution fields.
- `comparison_results/`: FEM metrics, eight figures, and complete TXT/NPZ fields.
- `isoline_results/`: multi-threshold IoU/Dice and cavitation-isoline errors.

The two root comparison scripts remain usable for re-evaluating an existing
checkpoint, but `reynold_pinn_torch.py` invokes the same functions automatically.

## Windows OpenMP environment note

The current Anaconda base environment on this machine exposes one
`libiomp5md.dll` from `Anaconda/Library/bin` and another from `torch/lib`. Intel
OpenMP may abort when PyTorch starts numerical work. The durable fix is to run in a
clean conda environment containing one consistent PyTorch/Intel OpenMP stack.

For a short diagnostic run only, the commonly used process-local workaround is:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python reynold_pinn_torch.py --smoke
```

It is intentionally not set inside the source code because it can hide a binary
dependency conflict in a long production training run.

## Active formulation

`PressureNet` and `GammaNet` are independent SiLU MLPs with fixed periodic Fourier
features. Pressure non-negativity, radial pressure values, gamma radial values, and
theta periodicity are imposed by the output representation. The loss combines:

1. Control-volume conservation of the directly computed first-order fluxes.
2. A smoothed Fischer-Burmeister constraint handled as an augmented Lagrangian.
3. Physics continuation in speed, film-transition width, and FB smoothing.

FEM values are never used for optimization or checkpoint selection. They are only
loaded after the best physics checkpoint has been restored.
