# Torch XPINN Hard-Groove Version

This is the new active implementation direction.

Core decision:

```text
No sigmoid film-thickness transition.
No global single network across H=1 and H=4.
No explicit H derivative.
```

The model uses two experts:

```text
Thin expert   -> H = 1 region
Groove expert -> H = 4 region
```

The groove interface is trained only through interface conditions:

```text
p_thin = p_groove
q_n_thin = q_n_groove
```

Current status:

- `geometry.py`: hard spiral-groove mask and interface sampling.
- `networks.py`: two independent pressure/gamma experts.
- `physics.py`: constant-H regional PDE residuals, FB losses, fluxes, and interface losses.
- `logging.py`: traditional per-term loss output with raw values, weights, and contributions.
- `trainer.py`: Adam training loop, checkpoints, logs, and sampled region batches.
- `evaluation.py`: FEM loading, hard-mask XPINN prediction, metrics, and comparison figures.
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

By default, training evaluates the best checkpoint against `p_FBNS.txt` and `g_FBNS.txt`.
The training log prints each raw loss, weight, and weighted contribution.
