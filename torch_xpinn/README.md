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
- `physics.py`: constant-H regional fluxes and interface losses.
- `logging.py`: traditional per-term loss output with raw values, weights, and contributions.
- `../reynold_xpinn.py`: scaffold inspection entry.

Next implementation step:

```text
Add region interior physics losses and a staged trainer.
```
