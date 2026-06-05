"""
FEM vs PINN 对比分析脚本 (PyTorch版, 完整9图)
================================================
直接运行即可，无需命令行参数。
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

from reynold_pinn_torch import (
    Config, compute_physical_params, create_H_func,
    create_pde_models, generate_groove_points,
)
from torch_pinn import TorchCollocationSolver, DomainND, dirichletBC

# =============================================================================
# 0. 路径配置
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))

def _p(filename):
    return os.path.join(_HERE, filename)

class Args:
    model_path = _p("output_torch/checkpoints/epochs_best_model.pt")
    fem_p      = _p("p_FBNS.txt")
    fem_g      = _p("g_FBNS.txt")
    out_dir    = _p("output_torch/comparison_results_iso_v16")
    n_grid     = 201
    dpi        = 300

def parse_args():
    return Args()


# =============================================================================
# 1. 读取 FEM 数据
# =============================================================================
def load_fem(fem_p_path, fem_g_path):
    raw_p = np.loadtxt(fem_p_path)
    raw_g = np.loadtxt(fem_g_path)
    R_pts, T_pts = raw_p[:, 0], raw_p[:, 1]
    P_fem, G_fem = raw_p[:, 2], raw_g[:, 2]
    n_R, n_T = len(np.unique(R_pts)), len(np.unique(T_pts))
    P_grid = P_fem.reshape(n_R, n_T)
    G_grid = G_fem.reshape(n_R, n_T)
    R_unique = np.unique(R_pts)
    T_unique = np.unique(T_pts)
    print(f"[FEM] 读取完成  |  R: {R_unique[0]:.5f} → {R_unique[-1]:.5f}"
          f"  |  θ: {T_unique[0]:.5f} → {T_unique[-1]:.5f}")
    print(f"[FEM] P: min={P_fem.min():.6f}  max={P_fem.max():.6f}")
    print(f"[FEM] g: min={G_fem.min():.6f}  max={G_fem.max():.6f}")
    print(f"[FEM] 空化区占比: {(G_fem > 1e-6).mean() * 100:.1f}%")
    print(f"[FEM] P·g 最大值: {(P_fem * G_fem).max():.2e}")
    return R_pts, T_pts, P_fem, G_fem, P_grid, G_grid, R_unique, T_unique


# =============================================================================
# 2. 加载 PINN 模型并推理
# =============================================================================
def load_pinn_and_predict(model_path, coords, cfg, params):
    H_func, theta_sym = create_H_func(params, cfg)
    f_model_FBNS, f_model_FB, set_w_wedge = create_pde_models(H_func, params, cfg)

    Domain = DomainND(["R", "theta"])
    Domain.add("R", params["R_lim"], cfg.domain_fidelity)
    Domain.add("theta", params["theta_lim"], cfg.domain_fidelity)
    Domain.X_f = Domain.generate_collocation_points(cfg.N_f, 1)
    add_R, add_theta = generate_groove_points(theta_sym, params, cfg)
    Domain.X_f = np.concatenate([Domain.X_f, np.concatenate([add_R, add_theta], 1)], 0)

    lower_bc = dirichletBC(Domain, val=params["P_i"], var="R", target="lower")
    upper_bc = dirichletBC(Domain, val=params["P_o"], var="R", target="upper")

    model = TorchCollocationSolver()
    model.compile(
        cfg.layer_sizes, [f_model_FBNS], Domain, [lower_bc, upper_bc],
        u_model_switch=cfg.u_model_switch, two_output=True, none_zero=False,
        adapt_True=False, isAdaptive=False, MTL_adapt=False, PCGrad_true=True,
        Boundary_true=False,
        R_range=params["R_lim"], theta_range=params["theta_lim"],
        bc_switch=cfg.bc_switch, num_freq=cfg.num_fourier_freq,
        embed_dim=cfg.embed_dim
    )

    model.load_weights(model_path)
    print(f"[PINN] 权重加载自: {model_path}")

    P_pinn, G_pinn = model.predict(coords)
    P_pinn = P_pinn.flatten()
    G_pinn = G_pinn.flatten()

    print(f"[PINN] P: min={P_pinn.min():.6f}  max={P_pinn.max():.6f}")
    print(f"[PINN] g: min={G_pinn.min():.6f}  max={G_pinn.max():.6f}")
    print(f"[PINN] 空化区占比: {(G_pinn > 1e-6).mean() * 100:.1f}%")
    print(f"[PINN] P·g 最大值: {(P_pinn * G_pinn).max():.2e}")
    return P_pinn, G_pinn


# =============================================================================
# 3. 误差指标计算
# =============================================================================
def compute_metrics(P_fem, G_fem, P_pinn, G_pinn):
    eps = 1e-16
    def rel_l2(ref, pred): return np.linalg.norm(ref - pred) / (np.linalg.norm(ref) + eps)
    def rel_linf(ref, pred): return np.max(np.abs(ref - pred)) / (np.max(np.abs(ref)) + eps)

    metrics = {
        "P_rel_L2": rel_l2(P_fem, P_pinn), "P_rel_Linf": rel_linf(P_fem, P_pinn),
        "G_rel_L2": rel_l2(G_fem, G_pinn), "G_rel_Linf": rel_linf(G_fem, G_pinn),
        "P_MAE": np.mean(np.abs(P_fem - P_pinn)), "P_RMSE": np.sqrt(np.mean((P_fem - P_pinn) ** 2)),
        "G_MAE": np.mean(np.abs(G_fem - G_pinn)), "G_RMSE": np.sqrt(np.mean((G_fem - G_pinn) ** 2)),
    }
    mask_cav = G_fem > 1e-6; mask_full = ~mask_cav
    if mask_cav.any():
        metrics["P_rel_L2_cavRegion"] = rel_l2(P_fem[mask_cav], P_pinn[mask_cav])
        metrics["G_rel_L2_cavRegion"] = rel_l2(G_fem[mask_cav], G_pinn[mask_cav])
    if mask_full.any():
        metrics["P_rel_L2_fullRegion"] = rel_l2(P_fem[mask_full], P_pinn[mask_full])
        metrics["G_rel_L2_fullRegion"] = rel_l2(G_fem[mask_full], G_pinn[mask_full])
    metrics["complementarity_violation"] = float(np.max(P_pinn * G_pinn))
    mask_cav_pinn = G_pinn > 1e-6
    intersection = (mask_cav & mask_cav_pinn).sum(); union = (mask_cav | mask_cav_pinn).sum()
    metrics["cavitation_IoU"] = intersection / (union + eps)
    metrics["cavitation_Dice"] = 2 * intersection / (mask_cav.sum() + mask_cav_pinn.sum() + eps)
    return metrics


def print_metrics(metrics):
    sep = "─" * 52
    print(f"\n{sep}\n  误差指标汇总\n{sep}")
    groups = [
        ("全局 – 压力 P", ["P_rel_L2", "P_rel_Linf", "P_MAE", "P_RMSE"]),
        ("全局 – 空化率 g", ["G_rel_L2", "G_rel_Linf", "G_MAE", "G_RMSE"]),
        ("分区域 – P", ["P_rel_L2_fullRegion", "P_rel_L2_cavRegion"]),
        ("分区域 – g", ["G_rel_L2_fullRegion", "G_rel_L2_cavRegion"]),
        ("空化区域形状", ["cavitation_IoU", "cavitation_Dice"]),
        ("JFO 互补条件", ["complementarity_violation"]),
    ]
    for title, keys in groups:
        print(f"\n  {title}")
        for k in keys:
            if k in metrics: print(f"    {k:<35s} = {metrics[k]:.6e}")
    print(f"{sep}\n")


# =============================================================================
# 4. 可视化
# =============================================================================
CMAP = cm.RdYlBu_r; FS = 14; FS_T = 12

def _savefig(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  → 保存: {os.path.basename(path)}")

def plot_field_comparison(R_uniq, T_uniq, P_fem_grid, G_fem_grid, P_pinn_grid, G_pinn_grid, out_dir, dpi):
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("FEM vs PINN — field comparison", fontsize=FS + 2)
    for data, title, ax in [
        (P_fem_grid, r"FEM  $P(R,\theta)$", axes[0,0]), (P_pinn_grid, r"PINN $\hat{P}(R,\theta)$", axes[0,1]),
        (G_fem_grid, r"FEM  $g(R,\theta)$", axes[1,0]), (G_pinn_grid, r"PINN $\hat{g}(R,\theta)$", axes[1,1]),
    ]:
        pc = ax.pcolormesh(RR, TT, data, shading="auto", cmap=CMAP)
        fig.colorbar(pc, ax=ax, pad=0.02)
        ax.set_title(title, fontsize=FS)
        ax.set_xlabel(r"$R$", fontsize=FS); ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12)
        ax.tick_params(labelsize=FS_T)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig1_field_comparison.png"), dpi)

def plot_error_maps(R_uniq, T_uniq, P_fem_grid, G_fem_grid, P_pinn_grid, G_pinn_grid, out_dir, dpi):
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")
    err_P = np.abs(P_fem_grid - P_pinn_grid); err_G = np.abs(G_fem_grid - G_pinn_grid)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Pointwise absolute error", fontsize=FS + 2)
    for err, label, ax in [(err_P, r"$|P_{FEM}-\hat{P}_{PINN}|$", axes[0]), (err_G, r"$|g_{FEM}-\hat{g}_{PINN}|$", axes[1])]:
        pc = ax.pcolormesh(RR, TT, err, shading="auto", cmap="hot_r")
        cb = fig.colorbar(pc, ax=ax, pad=0.02); cb.ax.tick_params(labelsize=FS_T)
        ax.set_title(label, fontsize=FS)
        ax.set_xlabel(r"$R$", fontsize=FS); ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12)
        ax.tick_params(labelsize=FS_T)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig2_error_maps.png"), dpi)

def plot_cavitation_boundary(R_uniq, T_uniq, G_fem_grid, G_pinn_grid, out_dir, dpi):
    from matplotlib.lines import Line2D
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Cavitation boundary — FEM vs PINN", fontsize=FS + 2)
    pc = ax.pcolormesh(RR, TT, G_pinn_grid, shading="auto", cmap="Blues", alpha=0.5)
    fig.colorbar(pc, ax=ax, label=r"PINN $\hat{g}$", pad=0.02)
    ax.contour(RR, TT, G_fem_grid, levels=[1e-6], colors=["red"], linewidths=1.8, linestyles="-")
    ax.contour(RR, TT, G_pinn_grid, levels=[1e-6], colors=["blue"], linewidths=1.8, linestyles="--")
    ax.legend(handles=[Line2D([0],[0], color="red", lw=1.8, ls="-", label="FEM"), Line2D([0],[0], color="blue", lw=1.8, ls="--", label="PINN")], fontsize=FS_T)
    ax.set_xlabel(r"$R$", fontsize=FS); ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12); ax.tick_params(labelsize=FS_T)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig3_cavitation_boundary.png"), dpi)

def plot_pressure_contour_overlay(R_uniq, T_uniq, P_fem_grid, P_pinn_grid, out_dir, dpi):
    from matplotlib.lines import Line2D
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")
    p_min, p_max = min(P_fem_grid.min(), P_pinn_grid.min()), max(P_fem_grid.max(), P_pinn_grid.max())
    levels = np.linspace(p_min, p_max, 15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Pressure contour overlay", fontsize=FS + 2)
    cs_fem = ax.contour(RR, TT, P_fem_grid, levels=levels, colors="red", linewidths=1.2, linestyles="-")
    cs_pinn = ax.contour(RR, TT, P_pinn_grid, levels=levels, colors="blue", linewidths=1.2, linestyles="--")
    ax.clabel(cs_fem, fontsize=8, fmt="%.3f")
    ax.set_xlabel(r"$R$", fontsize=FS); ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12); ax.tick_params(labelsize=FS_T)
    ax.legend(handles=[Line2D([0],[0], color="red", lw=1.2, ls="-", label="FEM"), Line2D([0],[0], color="blue", lw=1.2, ls="--", label="PINN")], fontsize=FS_T)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig4_pressure_contour_overlay.png"), dpi)

def plot_profiles(R_uniq, T_uniq, P_fem_grid, G_fem_grid, P_pinn_grid, G_pinn_grid, out_dir, dpi):
    nR, nT = len(R_uniq), len(T_uniq); idx_R, idx_T = nR // 2, nT // 2
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Profile comparison — mid-R and mid-θ slices", fontsize=FS + 2)
    setups = [
        (axes[0,0], T_uniq, P_fem_grid[idx_R,:], P_pinn_grid[idx_R,:], f"P  at  R = {R_uniq[idx_R]:.4f}"),
        (axes[0,1], T_uniq, G_fem_grid[idx_R,:], G_pinn_grid[idx_R,:], f"g  at  R = {R_uniq[idx_R]:.4f}"),
        (axes[1,0], R_uniq, P_fem_grid[:,idx_T], P_pinn_grid[:,idx_T], fr"P  at  $\theta$ = {T_uniq[idx_T]:.4f}"),
        (axes[1,1], R_uniq, G_fem_grid[:,idx_T], G_pinn_grid[:,idx_T], fr"g  at  $\theta$ = {T_uniq[idx_T]:.4f}"),
    ]
    for ax, x, fem, pinn, title in setups:
        ax.plot(x, fem, "r-", lw=1.5, label="FEM"); ax.plot(x, pinn, "b--", lw=1.5, label="PINN")
        ax.set_title(title, fontsize=FS); ax.legend(fontsize=FS_T); ax.tick_params(labelsize=FS_T)
    axes[0,0].set_xlabel(r"$\theta$"); axes[0,1].set_xlabel(r"$\theta$"); axes[1,0].set_xlabel(r"$R$"); axes[1,1].set_xlabel(r"$R$")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig5_profiles.png"), dpi)

def plot_periodic_check(R_uniq, T_uniq, P_fem_grid, P_pinn_grid, G_fem_grid, G_pinn_grid, out_dir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"Periodic boundary check  ($\theta=0$  vs  $\theta_{max}$)", fontsize=FS + 2)
    for ax, (f_fem, f_pinn, label) in zip(axes, [(P_fem_grid, P_pinn_grid, "P"), (G_fem_grid, G_pinn_grid, "g")]):
        ax.plot(R_uniq, f_fem[:,0], "r-", lw=1.5, label=r"FEM $\theta=0$")
        ax.plot(R_uniq, f_fem[:,-1], "r--", lw=1.2, label=r"FEM $\theta_{max}$")
        ax.plot(R_uniq, f_pinn[:,0], "b-", lw=1.5, label=r"PINN $\theta=0$")
        ax.plot(R_uniq, f_pinn[:,-1], "b--", lw=1.2, label=r"PINN $\theta_{max}$")
        ax.set_title(f"Field: {label}", fontsize=FS); ax.set_xlabel(r"$R$", fontsize=FS); ax.legend(fontsize=FS_T - 1); ax.tick_params(labelsize=FS_T)
        print(f"  周期性偏差 [{label}]  FEM: {np.max(np.abs(f_fem[:,0] - f_fem[:,-1])):.2e}   PINN: {np.max(np.abs(f_pinn[:,0] - f_pinn[:,-1])):.2e}")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig6_periodic_check.png"), dpi)

def plot_scatter_correlation(P_fem, G_fem, P_pinn, G_pinn, out_dir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Correlation: FEM vs PINN", fontsize=FS + 2)
    rng = np.random.default_rng(42); n_s = min(5000, len(P_fem)); idx = rng.choice(len(P_fem), n_s, replace=False)
    for ax, (ref, pred, label) in zip(axes, [(P_fem[idx], P_pinn[idx], "P"), (G_fem[idx], G_pinn[idx], "g")]):
        ax.scatter(ref, pred, s=4, alpha=0.4, color="steelblue")
        lo, hi = min(ref.min(), pred.min()), max(ref.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        ax.set_xlabel(f"FEM {label}", fontsize=FS); ax.set_ylabel(f"PINN {label}", fontsize=FS)
        ax.set_title(f"Scatter — {label}", fontsize=FS); ax.tick_params(labelsize=FS_T)
        ax.text(0.05, 0.92, f"$R^2$ = {np.corrcoef(ref, pred)[0,1]**2:.6f}", transform=ax.transAxes, fontsize=FS_T, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig7_scatter_correlation.png"), dpi)

def plot_jfo_complement(P_pinn, G_pinn, P_fem, G_fem, out_dir, dpi):
    rng = np.random.default_rng(0); n_s = min(8000, len(P_fem)); idx = rng.choice(len(P_fem), n_s, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("JFO complementarity condition  ($P \\cdot g = 0$)", fontsize=FS + 2)
    for ax, (pp, gg, title) in zip(axes, [(P_fem[idx], G_fem[idx], "FEM"), (P_pinn[idx], G_pinn[idx], "PINN")]):
        sc = ax.scatter(pp, gg, s=4, alpha=0.35, c=pp * gg, cmap="hot_r", vmin=0)
        fig.colorbar(sc, ax=ax, label=r"$P \cdot g$")
        ax.axvline(0, color="gray", lw=0.8, ls="--"); ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(r"$P$", fontsize=FS); ax.set_ylabel(r"$g$", fontsize=FS); ax.set_title(title, fontsize=FS); ax.tick_params(labelsize=FS_T)
        ax.text(0.05, 0.92, f"max($P·g$) = {np.max(pp*gg):.2e}", transform=ax.transAxes, fontsize=FS_T, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig8_jfo_complement.png"), dpi)

def plot_cavitation_isoline(R_uniq, T_uniq, G_fem_grid, G_pinn_grid, out_dir, dpi, iso_level=0.4):
    from matplotlib.lines import Line2D
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")
    g_nonzero = G_pinn_grid[G_pinn_grid > 1e-6]
    iso_auto = (g_nonzero.max() + g_nonzero.min()) / 2 if len(g_nonzero) > 1 else iso_level
    print(f"  自适应 iso_level = {iso_auto:.4f}")

    fig1, ax1 = plt.subplots(figsize=(7, 6))
    vmax_g = max(G_fem_grid.max(), G_pinn_grid.max())
    pc = ax1.pcolormesh(RR, TT, G_pinn_grid, shading="auto", cmap=CMAP, vmin=0, vmax=vmax_g)
    ax1.contourf(RR, TT, G_pinn_grid, levels=[-1e10, iso_auto], colors=["#313695"], alpha=1.0)
    ax1.contourf(RR, TT, G_pinn_grid, levels=[iso_auto, 1e10], colors=["#FDAE6B"], alpha=1.0)
    ax1.contour(RR, TT, G_fem_grid, levels=[iso_auto], colors=["red"], linewidths=2.0, linestyles="-")
    ax1.contour(RR, TT, G_pinn_grid, levels=[iso_auto], colors=["blue"], linewidths=2.0, linestyles="--")
    ax1.legend(handles=[Line2D([0],[0], color="red", lw=2.0, ls="-", label="FEM"), Line2D([0],[0], color="blue", lw=2.0, ls="--", label="PINN")], fontsize=FS_T, loc='upper right')
    ax1.set_title("Cavitation region", fontsize=FS)
    ax1.set_xlabel(r"$R$", fontsize=FS); ax1.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12); ax1.tick_params(labelsize=FS_T)
    fig1.tight_layout()
    _savefig(fig1, os.path.join(out_dir, f"fig9a_isoline_left_g{int(iso_auto*100):02d}.png"), dpi)

    # Isoline position error (right panel)
    def find_crossings(G_grid, level, R_u, T_u):
        R_valid, t_in, t_out = [], [], []
        for i in range(len(R_u)):
            g_row = G_grid[i, :]; above = g_row >= level
            cross = np.where(np.diff(above.astype(int)) != 0)[0]
            if len(cross) < 2: continue
            def interp(idx):
                t0, t1 = T_u[idx], T_u[idx + 1]; g0, g1 = g_row[idx], g_row[idx + 1]
                return (t0 + t1) / 2 if abs(g1 - g0) < 1e-12 else t0 + (level - g0) / (g1 - g0) * (t1 - t0)
            R_valid.append(R_u[i]); t_in.append(interp(cross[0])); t_out.append(interp(cross[-1]))
        return np.array(R_valid), np.array(t_in), np.array(t_out)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    R_f, t_f_in, t_f_out = find_crossings(G_fem_grid, iso_auto, R_uniq, T_uniq)
    R_p, t_p_in, t_p_out = find_crossings(G_pinn_grid, iso_auto, R_uniq, T_uniq)
    R_f_r, R_p_r = np.round(R_f, 8), np.round(R_p, 8); R_common = np.intersect1d(R_f_r, R_p_r)
    if len(R_common) > 0:
        idx_f = [np.argmin(np.abs(R_f_r - r)) for r in R_common]
        idx_p = [np.argmin(np.abs(R_p_r - r)) for r in R_common]
        diff_in = (t_p_in[idx_p] - t_f_in[idx_f]) * 1e3; diff_out = (t_p_out[idx_p] - t_f_out[idx_f]) * 1e3
        ax2.plot(diff_in, R_common, "b-", lw=1.8, label="入口侧偏差")
        ax2.plot(diff_out, R_common, "r--", lw=1.8, label="出口侧偏差")
        ax2.axvline(0, color="k", lw=0.8, ls=":")
        ax2.set_xlabel(r"$\Delta\theta \times 10^{-3}$ rad  (PINN $-$ FEM)", fontsize=FS)
        ax2.set_ylabel(r"$R$", fontsize=FS, rotation=0, labelpad=12)
        ax2.set_title(f"Isoline position error  (g={iso_auto:.2f})", fontsize=FS)
        ax2.legend(fontsize=FS_T); ax2.tick_params(labelsize=FS_T); ax2.grid(True, alpha=0.3)
        print(f"\n  g={iso_auto:.2f} 等值线位置偏差（PINN − FEM）：")
        print(f"    入口侧 mean={diff_in.mean():+.3f}e-3 rad  std={diff_in.std():.3f}e-3  max={np.abs(diff_in).max():.3f}e-3")
        print(f"    出口侧 mean={diff_out.mean():+.3f}e-3 rad  std={diff_out.std():.3f}e-3  max={np.abs(diff_out).max():.3f}e-3")
    fig2.tight_layout()
    _savefig(fig2, os.path.join(out_dir, f"fig9b_isoline_right_g{int(iso_auto*100):02d}.png"), dpi)


def save_metrics(metrics, out_dir):
    path = os.path.join(out_dir, "metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("FEM vs PINN — 误差指标 (PyTorch)\n" + "=" * 52 + "\n")
        for k, v in metrics.items(): f.write(f"  {k:<40s} = {v:.6e}\n")
    print(f"  → 指标保存: {os.path.basename(path)}")


# =============================================================================
# 5. 主函数
# =============================================================================
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(); params = compute_physical_params(cfg)

    print("\n[1/4] 读取 FEM 数据...")
    (R_pts, T_pts, P_fem, G_fem, P_fem_grid, G_fem_grid, R_uniq, T_uniq) = load_fem(args.fem_p, args.fem_g)

    print("\n[2/4] 加载 PINN 并推理...")
    coords = np.stack([R_pts, T_pts], axis=1).astype(np.float32)
    P_pinn, G_pinn = load_pinn_and_predict(args.model_path, coords, cfg, params)

    nR, nT = len(R_uniq), len(T_uniq)
    P_pinn_grid = P_pinn.reshape(nR, nT); G_pinn_grid = G_pinn.reshape(nR, nT)

    print("\n[3/4] 计算误差指标...")
    metrics = compute_metrics(P_fem, G_fem, P_pinn, G_pinn)
    print_metrics(metrics); save_metrics(metrics, args.out_dir)

    print("\n[4/4] 生成对比图...")
    dpi = args.dpi
    plot_field_comparison(R_uniq, T_uniq, P_fem_grid, G_fem_grid, P_pinn_grid, G_pinn_grid, args.out_dir, dpi)
    plot_error_maps(R_uniq, T_uniq, P_fem_grid, G_fem_grid, P_pinn_grid, G_pinn_grid, args.out_dir, dpi)
    plot_cavitation_boundary(R_uniq, T_uniq, G_fem_grid, G_pinn_grid, args.out_dir, dpi)
    plot_pressure_contour_overlay(R_uniq, T_uniq, P_fem_grid, P_pinn_grid, args.out_dir, dpi)
    plot_profiles(R_uniq, T_uniq, P_fem_grid, G_fem_grid, P_pinn_grid, G_pinn_grid, args.out_dir, dpi)
    plot_periodic_check(R_uniq, T_uniq, P_fem_grid, P_pinn_grid, G_fem_grid, G_pinn_grid, args.out_dir, dpi)
    plot_scatter_correlation(P_fem, G_fem, P_pinn, G_pinn, args.out_dir, dpi)
    plot_jfo_complement(P_pinn, G_pinn, P_fem, G_fem, args.out_dir, dpi)
    plot_cavitation_isoline(R_uniq, T_uniq, G_fem_grid, G_pinn_grid, args.out_dir, dpi)
    print(f"\n完成！所有结果保存至: {os.path.abspath(args.out_dir)}\n")


if __name__ == "__main__":
    main()
