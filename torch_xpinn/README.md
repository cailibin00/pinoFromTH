# Torch XPINN Hard-Partition Version

This implementation keeps the clean two-expert idea while avoiding the previous
flat-solution failure mode.

Core decisions:

```text
Two experts, one for each film-thickness region.
No sigmoid film-thickness transition.
No explicit H derivative.
Reynolds loss includes conservative interface coupling.
```

The model uses two independent experts:

```text
Thin expert   -> H = 1 region
Groove expert -> H = 4 region
```

The hard groove mask chooses which expert evaluates a point. Inside each region
the film thickness is constant, so the PDE does not differentiate a transition
layer. Because discontinuous film thickness still needs conservation across the
groove edge, the Reynolds loss contains both regional residuals and pressure/flux
interface coupling.

Training reports three public loss terms:

```text
L = w_R * L_Reynolds + w_JFO * L_JFO + w_BC * L_BC
```

- `L_Reynolds`: raw constant-H regional residual plus conservative interface
  coupling.
- `L_JFO`: direct complementarity loss `mean((p * gamma)^2)`.
- `L_BC`: inner and outer radial pressure boundary loss.

The network input includes both global periodic theta features and local spiral
phase features. This keeps the architecture geometry-aware without hard-coding a
specific solution field.

Optimisation is Adam with cosine annealing only.

Quick checks:

```powershell
python reynold_xpinn.py --inspect-geometry --device cpu --smoke
python reynold_xpinn.py --device cpu --smoke --output-dir output_xpinn_smoke --no-evaluate
python compare_fem_xpinn.py --checkpoint output_xpinn_smoke/checkpoints/best.pt --device cpu
```
