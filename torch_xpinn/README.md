# Torch XPINN Hard-Partition Version

This is the active PyTorch implementation for the spiral-groove Reynolds
cavitation problem.

Core decisions:

```text
Two experts, one for each film-thickness region.
No sigmoid film-thickness transition.
No explicit H derivative.
No interface, periodic, or Fischer-Burmeister loss in training.
```

The model uses two independent experts:

```text
Thin expert   -> H = 1 region
Groove expert -> H = 4 region
```

The hard groove mask only chooses which expert evaluates a point. Inside each
region the film thickness is constant, so the Reynolds residual does not contain
a differentiated transition layer.

Training loss is intentionally reduced to three terms:

```text
L = w_R * L_Reynolds + w_JFO * L_JFO + w_BC * L_BC
```

- `L_Reynolds`: raw constant-H Reynolds residual, without residual scaling or
  stabilization variants.
- `L_JFO`: direct complementarity loss `mean((p * gamma)^2)`.
- `L_BC`: inner and outer radial pressure boundary loss.

Pressure is not hard-wired to the radial boundary values in the network output.
It is only constrained by `L_BC`, which keeps the three training terms explicit.

Optimisation is Adam with cosine annealing only.

Current modules:

- `geometry.py`: hard spiral-groove mask and fixed `H=1/H=4` film values.
- `networks.py`: two independent pressure/gamma experts.
- `physics.py`: raw Reynolds residual and direct JFO complementarity loss.
- `logging.py`: per-term logging for the three simplified loss terms.
- `trainer.py`: Adam training loop with cosine annealing.
- `evaluation.py`: FEM loading, hard-mask XPINN prediction, metrics, and figures.
- `../reynold_xpinn.py`: active training entry.
- `../compare_fem_xpinn.py`: standalone checkpoint evaluation entry.

Quick checks:

```powershell
python reynold_xpinn.py --inspect-geometry --device cpu --smoke
python reynold_xpinn.py --device cpu --smoke --output-dir output_xpinn_smoke --no-evaluate
python compare_fem_xpinn.py --checkpoint output_xpinn_smoke/checkpoints/best.pt --device cpu
```

Normal training starts with:

```powershell
python reynold_xpinn.py --device cuda
```

By default, training evaluates the best checkpoint against `p_FBNS.txt` and
`g_FBNS.txt`.
