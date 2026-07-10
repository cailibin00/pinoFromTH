"""
FEM vs PINN Comparison Script with Isoline Analysis (PyTorch version).
Faithful port from 0504/compare_fem_pinn_iso_v16.py.

Extends compare_fem_pinn_final_torch.py with:
- 2x2 combined field comparison (fig1)
- hot_r error maps (fig2)
- Cavitation isoline position analysis (fig9)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

def _p(filename):
    return os.path.join(_HERE, filename)


class Args:
    model_path = os.path.join(_HERE, "output_torch", "checkpoints", "epochs_best_model.pt")
    fem_p      = _p("p_FBNS.txt")
    fem_g      = _p("g_FBNS.txt")
    out_dir    = os.path.join(_HERE, "output_torch", "comparison_results_iso_v16")
    n_grid     = 201
    dpi        = 300


from reynold_pinn_torch import (
    Config, compute_physical_params, create_H_func,
    create_pde_models, generate_groove_points,
)
from torch_pinn.models import CollocationSolverND
from torch_pinn.boundaries import dirichletBC
from torch_pinn.domains import DomainND


# =============================================================================
# 1. Load FEM
# =============================================================================
def load_fem(fem_p_path, fem_g_path):
    raw_p = np.loadtxt(fem_p_path)
    raw_g = np.loadtxt(fem_g_path)
    R_pts, T_pts = raw_p[:, 0], raw_p[:, 1]
    P_fem, G_fem = raw_p[:, 2], raw_g[:, 2]
    n_R, n_T = len(np.unique(R_pts)), len(np.unique(T_pts))
    P_grid = P_fem.reshape(n_R, n_T)
    G_grid = G_fem.reshape(n_R, n_T)
    R_unique, T_unique = np.unique(R_pts), np.unique(T_pts)
    print(f"[FEM] Loaded | R: {R_unique[0]:.5f} -> {R_unique[-1]:.5f}"
          f"  |  theta: {T_unique[0]:.5f} -> {T_unique[-1]:.5f}")
    print(f"[FEM] P: min={P_fem.min():.6f}  max={P_fem.max():.6f}")
    print(f"[FEM] g: min={G_fem.min():.6f}  max={G_fem.max():.6f}")
    return R_pts, T_pts, P_fem, G_fem, P_grid, G_grid, R_unique, T_unique


# =============================================================================
# 2. Load PINN
# =============================================================================
def load_pinn_and_predict(model_path, coords, cfg, params):
    H_func, theta_sym = create_H_func(params, cfg)
    f_model_FBNS, f_model_FB = create_pde_models(H_func, params)

    Domain = DomainND(["R", "theta"])
    Domain.add("R", params["R_lim"], cfg.domain_fidelity)
    Domain.add("theta", params["theta_lim"], cfg.domain_fidelity)
    Domain.X_f = Domain.generate_collocation_points(cfg.N_f, 1)

    add_R, add_theta = generate_groove_points(theta_sym, params, cfg)
    Domain.X_f = np.concatenate([Domain.X_f, np.concatenate([add_R, add_theta], 1)], 0)

    lower_bc = dirichletBC(Domain, val=params["P_i"], var="R", target="lower")
    upper_bc = dirichletBC(Domain, val=params["P_o"], var="R", target="upper")

    model = CollocationSolverND()
    if cfg.core == 'pikan' and cfg.pikan_layer_sizes is not None:
        effective_sizes = cfg.pikan_layer_sizes
    else:
        effective_sizes = cfg.layer_sizes
    model.compile(
        effective_sizes, [f_model_FBNS], Domain, [lower_bc, upper_bc],
        u_model_switch=8, two_output=True, none_zero=False, adapt_True=False,
        isAdaptive=False, MTL_adapt=False, PCGrad_true=True, Boundary_true=False,
        R_range=params["R_lim"], theta_range=params["theta_lim"],
        core=cfg.core, kan_grid_size=cfg.kan_grid_size,
        kan_spline_order=cfg.kan_spline_order,
        output_head_dim=cfg.output_head_dim,
    )
    model.load_weights(model_path)
    print(f"[PINN] Weights loaded from: {model_path}")

    u_pred = model.predict(coords)
    P_pinn, G_pinn = u_pred[0].flatten(), u_pred[1].flatten()

    print(f"[PINN] P: min={P_pinn.min():.6f}  max={P_pinn.max():.6f}")
    print(f"[PINN] g: min={G_pinn.min():.6f}  max={G_pinn.max():.6f}")
    return P_pinn, G_pinn


# =============================================================================
# 3. Metrics
# =============================================================================
def compute_metrics(P_fem, G_fem, P_pinn, G_pinn):
    eps = 1e-16
    def rel_l2(ref, pred):
        return np.linalg.norm(ref - pred) / (np.linalg.norm(ref) + eps)
    def rel_linf(ref, pred):
        return np.max(np.abs(ref - pred)) / (np.max(np.abs(ref)) + eps)

    metrics = {
        "P_rel_L2": rel_l2(P_fem, P_pinn), "P_rel_Linf": rel_linf(P_fem, P_pinn),
        "G_rel_L2": rel_l2(G_fem, G_pinn), "G_rel_Linf": rel_linf(G_fem, G_pinn),
        "P_MAE": np.mean(np.abs(P_fem - P_pinn)), "P_RMSE": np.sqrt(np.mean((P_fem - P_pinn)**2)),
        "G_MAE": np.mean(np.abs(G_fem - G_pinn)), "G_RMSE": np.sqrt(np.mean((G_fem - G_pinn)**2)),
    }
    mask_cav, mask_full = G_fem > 1e-6, ~(G_fem > 1e-6)
    if mask_cav.any():
        metrics["P_rel_L2_cavRegion"] = rel_l2(P_fem[mask_cav], P_pinn[mask_cav])
        metrics["G_rel_L2_cavRegion"] = rel_l2(G_fem[mask_cav], G_pinn[mask_cav])
    if mask_full.any():
        metrics["P_rel_L2_fullRegion"] = rel_l2(P_fem[mask_full], P_pinn[mask_full])
        metrics["G_rel_L2_fullRegion"] = rel_l2(G_fem[mask_full], G_pinn[mask_full])
    metrics["complementarity_violation"] = float(np.max(P_pinn * G_pinn))
    mask_cav_pinn = G_pinn > 1e-6
    intersection = (mask_cav & mask_cav_pinn).sum()
    metrics["cavitation_IoU"] = intersection / ((mask_cav | mask_cav_pinn).sum() + eps)
    metrics["cavitation_Dice"] = 2 * intersection / (mask_cav.sum() + mask_cav_pinn.sum() + eps)
    return metrics


def print_metrics(metrics):
    sep = "-" * 52
    print(f"\n{sep}\n  Error Metrics Summary\n{sep}")
    for title, keys in [
        ("Global - P", ["P_rel_L2","P_rel_Linf","P_MAE","P_RMSE"]),
        ("Global - g", ["G_rel_L2","G_rel_Linf","G_MAE","G_RMSE"]),
        ("Per-Region", ["P_rel_L2_fullRegion","P_rel_L2_cavRegion","G_rel_L2_fullRegion","G_rel_L2_cavRegion"]),
        ("Cavitation", ["cavitation_IoU","cavitation_Dice"]),
        ("JFO", ["complementarity_violation"]),
    ]:
        print(f"\n  {title}")
        for k in keys:
            if k in metrics: print(f"    {k:<35s} = {metrics[k]:.6e}")
    print(f"{sep}\n")


# =============================================================================
# 4. Visualization
# =============================================================================
CMAP = cm.RdYlBu_r
FS, FS_T = 14, 12

def _savefig(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  -> Saved: {os.path.basename(path)}")


def plot_field_comparison(R_uniq, T_uniq, P_fem_grid, G_fem_grid, P_pinn_grid, G_pinn_grid, out_dir, dpi):
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("FEM vs PINN - field comparison", fontsize=FS + 2)
    for data, title, ax in [
        (P_fem_grid, r"FEM  $P(R,\theta)$", axes[0,0]),
        (P_pinn_grid, r"PINN $\hat{P}(R,\theta)$", axes[0,1]),
        (G_fem_grid, r"FEM  $g(R,\theta)$", axes[1,0]),
        (G_pinn_grid, r"PINN $\hat{g}(R,\theta)$", axes[1,1]),
    ]:
        pc = ax.pcolormesh(RR, TT, data, shading="auto", cmap=CMAP)
        fig.colorbar(pc, ax=ax, pad=0.02)
        ax.set_title(title, fontsize=FS)
        ax.set_xlabel(r"$R$", fontsize=FS)
        ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12)
        ax.tick_params(labelsize=FS_T)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig1_field_comparison.png"), dpi)


def plot_error_maps(R_uniq, T_uniq, P_fem_grid, G_fem_grid, P_pinn_grid, G_pinn_grid, out_dir, dpi):
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")
    err_P, err_G = np.abs(P_fem_grid - P_pinn_grid), np.abs(G_fem_grid - G_pinn_grid)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Pointwise absolute error", fontsize=FS + 2)
    for err, label, ax in [(err_P, r"$|P_{FEM}-\hat{P}_{PINN}|$", axes[0]),
                            (err_G, r"$|g_{FEM}-\hat{g}_{PINN}|$", axes[1])]:
        pc = ax.pcolormesh(RR, TT, err, shading="auto", cmap="hot_r")
        cb = fig.colorbar(pc, ax=ax, pad=0.02)
        cb.ax.tick_params(labelsize=FS_T)
        ax.set_title(label, fontsize=FS)
        ax.set_xlabel(r"$R$", fontsize=FS)
        ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12)
        ax.tick_params(labelsize=FS_T)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig2_error_maps.png"), dpi)


def plot_cavitation_boundary(R_uniq, T_uniq, G_fem_grid, G_pinn_grid, out_dir, dpi):
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Cavitation boundary - FEM vs PINN", fontsize=FS + 2)
    pc = ax.pcolormesh(RR, TT, G_pinn_grid, shading="auto", cmap="Blues", alpha=0.5)
    fig.colorbar(pc, ax=ax, label=r"PINN $\hat{g}$", pad=0.02)
    ax.contour(RR, TT, G_fem_grid, levels=[1e-6], colors=["red"], linewidths=1.8, linestyles="-")
    ax.contour(RR, TT, G_pinn_grid, levels=[1e-6], colors=["blue"], linewidths=1.8, linestyles="--")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color="red", lw=1.8, ls="-", label="FEM boundary"),
        Line2D([0],[0], color="blue", lw=1.8, ls="--", label="PINN boundary"),
    ], fontsize=FS_T)
    ax.set_xlabel(r"$R$", fontsize=FS)
    ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12)
    ax.tick_params(labelsize=FS_T)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig3_cavitation_boundary.png"), dpi)


def plot_pressure_contour_overlay(R_uniq, T_uniq, P_fem_grid, P_pinn_grid, out_dir, dpi):
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")
    p_min = min(P_fem_grid.min(), P_pinn_grid.min())
    p_max = max(P_fem_grid.max(), P_pinn_grid.max())
    levels = np.linspace(p_min, p_max, 15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Pressure contour overlay", fontsize=FS + 2)
    cs_fem = ax.contour(RR, TT, P_fem_grid, levels=levels, colors="red", linewidths=1.2, linestyles="-")
    cs_pinn = ax.contour(RR, TT, P_pinn_grid, levels=levels, colors="blue", linewidths=1.2, linestyles="--")
    ax.clabel(cs_fem, fontsize=8, fmt="%.3f")
    ax.set_xlabel(r"$R$", fontsize=FS)
    ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12)
    ax.tick_params(labelsize=FS_T)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color="red", lw=1.2, ls="-", label="FEM"),
        Line2D([0],[0], color="blue", lw=1.2, ls="--", label="PINN"),
    ], fontsize=FS_T)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig4_pressure_contour_overlay.png"), dpi)


def plot_profiles(R_uniq, T_uniq, P_fem_grid, G_fem_grid, P_pinn_grid, G_pinn_grid, out_dir, dpi):
    nR, nT = len(R_uniq), len(T_uniq)
    idx_R, idx_T = nR // 2, nT // 2
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Profile comparison", fontsize=FS + 2)
    configs = [
        (axes[0,0], T_uniq, P_fem_grid[idx_R,:], P_pinn_grid[idx_R,:], r"$\theta$", r"$P$", f"P at R={R_uniq[idx_R]:.4f}"),
        (axes[0,1], T_uniq, G_fem_grid[idx_R,:], G_pinn_grid[idx_R,:], r"$\theta$", r"$g$", f"g at R={R_uniq[idx_R]:.4f}"),
        (axes[1,0], R_uniq, P_fem_grid[:,idx_T], P_pinn_grid[:,idx_T], r"$R$", r"$P$", f"P at theta={T_uniq[idx_T]:.4f}"),
        (axes[1,1], R_uniq, G_fem_grid[:,idx_T], G_pinn_grid[:,idx_T], r"$R$", r"$g$", f"g at theta={T_uniq[idx_T]:.4f}"),
    ]
    for ax, x, y_fem, y_pinn, xlabel, ylabel, title in configs:
        ax.plot(x, y_fem, "r-", lw=1.5, label="FEM")
        ax.plot(x, y_pinn, "b--", lw=1.5, label="PINN")
        ax.set_title(title, fontsize=FS)
        ax.set_xlabel(xlabel, fontsize=FS); ax.set_ylabel(ylabel, fontsize=FS)
        ax.legend(fontsize=FS_T); ax.tick_params(labelsize=FS_T)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig5_profiles.png"), dpi)


def plot_periodic_check(R_uniq, P_fem_grid, P_pinn_grid, G_fem_grid, G_pinn_grid, out_dir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"Periodic boundary check ($\theta=0$ vs $\theta_{max}$)", fontsize=FS + 2)
    for ax, (ff, fp, label) in zip(axes, [(P_fem_grid, P_pinn_grid, "P"), (G_fem_grid, G_pinn_grid, "g")]):
        ax.plot(R_uniq, ff[:,0], "r-", lw=1.5, label=r"FEM $\theta=0$")
        ax.plot(R_uniq, ff[:,-1], "r--", lw=1.2, label=r"FEM $\theta_{max}$")
        ax.plot(R_uniq, fp[:,0], "b-", lw=1.5, label=r"PINN $\theta=0$")
        ax.plot(R_uniq, fp[:,-1], "b--", lw=1.2, label=r"PINN $\theta_{max}$")
        ax.set_title(f"Field: {label}", fontsize=FS)
        ax.set_xlabel(r"$R$", fontsize=FS); ax.set_ylabel(label, fontsize=FS)
        ax.legend(fontsize=FS_T-1); ax.tick_params(labelsize=FS_T)
        print(f"  Periodic [{label}] FEM max diff: {np.max(np.abs(ff[:,0]-ff[:,-1])):.2e}  PINN: {np.max(np.abs(fp[:,0]-fp[:,-1])):.2e}")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig6_periodic_check.png"), dpi)


def plot_scatter_correlation(P_fem, G_fem, P_pinn, G_pinn, out_dir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Correlation", fontsize=FS + 2)
    rng = np.random.default_rng(42)
    n_s = min(5000, len(P_fem))
    idx = rng.choice(len(P_fem), n_s, replace=False)
    for ax, (ref, pred, label) in zip(axes, [(P_fem[idx], P_pinn[idx], "P"), (G_fem[idx], G_pinn[idx], "g")]):
        ax.scatter(ref, pred, s=4, alpha=0.4, color="steelblue")
        lo, hi = min(ref.min(), pred.min()), max(ref.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="ideal")
        ax.set_xlabel(f"FEM {label}", fontsize=FS); ax.set_ylabel(f"PINN {label}", fontsize=FS)
        ax.set_title(f"Scatter - {label}", fontsize=FS)
        ax.legend(fontsize=FS_T); ax.tick_params(labelsize=FS_T)
        corr = np.corrcoef(ref, pred)[0,1]**2
        ax.text(0.05, 0.92, f"$R^2$ = {corr:.6f}", transform=ax.transAxes, fontsize=FS_T,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig7_scatter_correlation.png"), dpi)


def plot_jfo_complement(P_pinn, G_pinn, P_fem, G_fem, out_dir, dpi):
    rng = np.random.default_rng(0)
    n_s = min(8000, len(P_fem))
    idx = rng.choice(len(P_fem), n_s, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("JFO complementarity ($P \\cdot g = 0$)", fontsize=FS + 2)
    for ax, (pp, gg, title) in zip(axes, [(P_fem[idx], G_fem[idx], "FEM"), (P_pinn[idx], G_pinn[idx], "PINN")]):
        sc = ax.scatter(pp, gg, s=4, alpha=0.35, c=pp*gg, cmap="hot_r", vmin=0)
        fig.colorbar(sc, ax=ax, label=r"$P \cdot g$")
        ax.axvline(0, color="gray", lw=0.8, ls="--"); ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(r"$P$", fontsize=FS); ax.set_ylabel(r"$g$", fontsize=FS)
        ax.set_title(title, fontsize=FS); ax.tick_params(labelsize=FS_T)
        ax.text(0.05, 0.92, f"max(P*g) = {np.max(pp*gg):.2e}", transform=ax.transAxes, fontsize=FS_T,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig8_jfo_complement.png"), dpi)


def plot_cavitation_isoline(R_uniq, T_uniq, G_fem_grid, G_pinn_grid, out_dir, dpi):
    """Fig9: Cavitation isoline position analysis."""
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")

    # Adaptive iso_level
    g_nonzero = G_pinn_grid[G_pinn_grid > 1e-6]
    iso_level_auto = (g_nonzero.max() + g_nonzero.min()) / 2
    print(f"  Adaptive iso_level = {iso_level_auto:.4f}")

    fig1, ax1 = plt.subplots(figsize=(7, 6))
    vmax_g = max(G_fem_grid.max(), G_pinn_grid.max())
    pc = ax1.pcolormesh(RR, TT, G_pinn_grid, shading="auto", cmap=CMAP, vmin=0, vmax=vmax_g)
    ax1.contourf(RR, TT, G_pinn_grid, levels=[-1e10, iso_level_auto], colors=["#313695"], alpha=1.0)
    ax1.contourf(RR, TT, G_pinn_grid, levels=[iso_level_auto, 1e10], colors=["#FDAE6B"], alpha=1.0)
    ax1.contour(RR, TT, G_fem_grid, levels=[iso_level_auto], colors=["red"], linewidths=2.0, linestyles="-")
    ax1.contour(RR, TT, G_pinn_grid, levels=[iso_level_auto], colors=["blue"], linewidths=2.0, linestyles="--")
    from matplotlib.lines import Line2D
    ax1.legend(handles=[
        Line2D([0],[0], color="red", lw=2.0, ls="-", label="FEM"),
        Line2D([0],[0], color="blue", lw=2.0, ls="--", label="PINN"),
    ], fontsize=FS_T, loc='upper right')
    ax1.set_title("Cavitation region", fontsize=FS)
    ax1.set_xlabel(r"$R$", fontsize=FS); ax1.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12)
    ax1.tick_params(labelsize=FS_T)
    fig1.tight_layout()
    _savefig(fig1, os.path.join(out_dir, f"fig9_isoline_left_g{int(iso_level_auto*100):02d}.png"), dpi)

    # Right plot: theta position deviation
    fig2, ax2 = plt.subplots(figsize=(7, 6))

    def find_crossings(G_grid, level):
        R_valid, t_in, t_out = [], [], []
        for i in range(len(R_uniq)):
            g_row = G_grid[i, :]
            above = g_row >= level
            cross = np.where(np.diff(above.astype(int)) != 0)[0]
            if len(cross) < 2: continue
            def interp(idx):
                t0, t1 = T_uniq[idx], T_uniq[idx+1]
                g0, g1 = g_row[idx], g_row[idx+1]
                return t0 + (level - g0)/(g1 - g0) * (t1 - t0) if abs(g1-g0) > 1e-12 else (t0+t1)/2
            R_valid.append(R_uniq[i])
            t_in.append(interp(cross[0]))
            t_out.append(interp(cross[-1]))
        return np.array(R_valid), np.array(t_in), np.array(t_out)

    R_f, t_f_in, t_f_out = find_crossings(G_fem_grid, iso_level_auto)
    R_p, t_p_in, t_p_out = find_crossings(G_pinn_grid, iso_level_auto)

    R_common = np.intersect1d(np.round(R_f, 8), np.round(R_p, 8))
    if len(R_common) > 0:
        idx_f = [np.argmin(np.abs(np.round(R_f, 8) - r)) for r in R_common]
        idx_p = [np.argmin(np.abs(np.round(R_p, 8) - r)) for r in R_common]
        diff_in  = (t_p_in[idx_p]  - t_f_in[idx_f])  * 1e3
        diff_out = (t_p_out[idx_p] - t_f_out[idx_f]) * 1e3
        ax2.plot(diff_in,  R_common, "b-",  lw=1.8, label="Entry side")
        ax2.plot(diff_out, R_common, "r--", lw=1.8, label="Exit side")
        ax2.axvline(0, color="k", lw=0.8, ls=":")
        ax2.fill_betweenx(R_common, diff_in,  0, alpha=0.15, color="blue")
        ax2.fill_betweenx(R_common, diff_out, 0, alpha=0.15, color="red")
        ax2.set_xlabel(r"$\Delta\theta \times 10^{-3}$ rad (PINN - FEM)", fontsize=FS)
        ax2.set_ylabel(r"$R$", fontsize=FS, rotation=0, labelpad=12)
        ax2.set_title(f"Isoline position error (g={iso_level_auto:.2f})", fontsize=FS)
        ax2.legend(fontsize=FS_T); ax2.tick_params(labelsize=FS_T); ax2.grid(True, alpha=0.3)
        print(f"  Isoline position error (g={iso_level_auto:.2f}):")
        print(f"    Entry: mean={diff_in.mean():+.3f}e-3  std={diff_in.std():.3f}e-3  max|err|={np.abs(diff_in).max():.3f}e-3")
        print(f"    Exit:  mean={diff_out.mean():+.3f}e-3  std={diff_out.std():.3f}e-3  max|err|={np.abs(diff_out).max():.3f}e-3")
    fig2.tight_layout()
    _savefig(fig2, os.path.join(out_dir, f"fig9_isoline_right_g{int(iso_level_auto*100):02d}.png"), dpi)


def save_metrics(metrics, out_dir):
    path = os.path.join(out_dir, "metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("FEM vs PINN - Error Metrics\n" + "=" * 52 + "\n")
        for k, v in metrics.items():
            f.write(f"  {k:<40s} = {v:.6e}\n")
    print(f"  -> Metrics saved: {os.path.basename(path)}")


# =============================================================================
# 5. Main
# =============================================================================
def main():
    args = Args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config()
    params = compute_physical_params(cfg)

    print("\n[1/4] Loading FEM data...")
    (R_pts, T_pts, P_fem, G_fem, P_fem_grid, G_fem_grid, R_uniq, T_uniq) = load_fem(args.fem_p, args.fem_g)

    print("\n[2/4] Loading PINN and predicting...")
    coords = np.stack([R_pts, T_pts], axis=1).astype(np.float32)
    P_pinn, G_pinn = load_pinn_and_predict(args.model_path, coords, cfg, params)

    nR, nT = len(R_uniq), len(T_uniq)
    P_pinn_grid = P_pinn.reshape(nR, nT)
    G_pinn_grid = G_pinn.reshape(nR, nT)

    print("\n[3/4] Computing metrics...")
    metrics = compute_metrics(P_fem, G_fem, P_pinn, G_pinn)
    print_metrics(metrics)
    save_metrics(metrics, args.out_dir)

    print("\n[4/4] Generating plots...")
    dpi = args.dpi

    plot_field_comparison(R_uniq, T_uniq, P_fem_grid, G_fem_grid, P_pinn_grid, G_pinn_grid, args.out_dir, dpi)
    plot_error_maps(R_uniq, T_uniq, P_fem_grid, G_fem_grid, P_pinn_grid, G_pinn_grid, args.out_dir, dpi)
    plot_cavitation_boundary(R_uniq, T_uniq, G_fem_grid, G_pinn_grid, args.out_dir, dpi)
    plot_pressure_contour_overlay(R_uniq, T_uniq, P_fem_grid, P_pinn_grid, args.out_dir, dpi)
    plot_profiles(R_uniq, T_uniq, P_fem_grid, G_fem_grid, P_pinn_grid, G_pinn_grid, args.out_dir, dpi)
    plot_periodic_check(R_uniq, P_fem_grid, P_pinn_grid, G_fem_grid, G_pinn_grid, args.out_dir, dpi)
    plot_scatter_correlation(P_fem, G_fem, P_pinn, G_pinn, args.out_dir, dpi)
    plot_jfo_complement(P_pinn, G_pinn, P_fem, G_fem, args.out_dir, dpi)
    plot_cavitation_isoline(R_uniq, T_uniq, G_fem_grid, G_pinn_grid, args.out_dir, dpi)

    print(f"\nDone! All results saved to: {os.path.abspath(args.out_dir)}\n")


if __name__ == "__main__":
    main()
