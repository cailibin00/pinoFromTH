"""
FEM vs PINN 对比分析脚本 (PyTorch版)
================================================
直接运行即可，无需命令行参数。
所有路径自动根据本脚本所在目录确定。

依赖：reynold_pinn_torch.py 与本脚本放在同一目录。
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# 选择显卡: 改数字即可, 0=第一张, 1=第二张, 'cpu'=不用显卡
CUDA_DEVICE = 0
torch.cuda.set_device(CUDA_DEVICE) if torch.cuda.is_available() else None
import matplotlib.ticker as ticker
from matplotlib import cm

# ── 从 PyTorch PINN 文件导入配置与辅助函数 ──
from reynold_pinn_torch import (
    Config,
    compute_physical_params,
    create_H_func,
    create_pde_models,
    generate_groove_points,
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
    out_dir    = _p("output_torch/comparison_results")
    n_grid     = 201
    dpi        = 300

def parse_args():
    return Args()


# =============================================================================
# 1. 读取 FEM 数据
# =============================================================================
def load_fem(fem_p_path: str, fem_g_path: str):
    raw_p = np.loadtxt(fem_p_path)
    raw_g = np.loadtxt(fem_g_path)

    R_pts = raw_p[:, 0]
    T_pts = raw_p[:, 1]
    P_fem = raw_p[:, 2]
    G_fem = raw_g[:, 2]

    n_R = len(np.unique(R_pts))
    n_T = len(np.unique(T_pts))

    P_grid = P_fem.reshape(n_R, n_T)
    G_grid = G_fem.reshape(n_R, n_T)

    R_unique = np.unique(R_pts)
    T_unique = np.unique(T_pts)

    print(f"[FEM] 读取完成  |  R: {R_unique[0]:.5f} → {R_unique[-1]:.5f}"
          f"  |  θ: {T_unique[0]:.5f} → {T_unique[-1]:.5f}")
    print(f"[FEM] P: min={P_fem.min():.6f}  max={P_fem.max():.6f}")
    print(f"[FEM] g: min={G_fem.min():.6f}  max={G_fem.max():.6f}")
    cav_frac = (G_fem > 1e-6).mean() * 100
    print(f"[FEM] 空化区占比: {cav_frac:.1f}%")
    print(f"[FEM] P·g 最大值（应≈0）: {(P_fem * G_fem).max():.2e}")

    return R_pts, T_pts, P_fem, G_fem, P_grid, G_grid, R_unique, T_unique


# =============================================================================
# 2. 加载 PINN 模型并推理
# =============================================================================
def load_pinn_and_predict(model_path: str, coords: np.ndarray, cfg: Config, params: dict):
    """加载 PyTorch PINN 权重并在 coords 上推理。"""
    from reynold_pinn_torch import create_pde_models, generate_groove_points
    from torch_pinn import DomainND, dirichletBC

    H_func, theta_sym = create_H_func(params, cfg)
    f_model_FBNS, f_model_FB, set_w_wedge = create_pde_models(H_func, params, cfg)

    Domain = DomainND(["R", "theta"])
    Domain.add("R",     params["R_lim"],     cfg.domain_fidelity)
    Domain.add("theta", params["theta_lim"], cfg.domain_fidelity)
    Domain.X_f = Domain.generate_collocation_points(cfg.N_f, 1)

    add_R, add_theta = generate_groove_points(theta_sym, params, cfg)
    Domain.X_f = np.concatenate(
        [Domain.X_f, np.concatenate([add_R, add_theta], 1)], 0
    )

    lower_bc = dirichletBC(Domain, val=params["P_i"], var="R", target="lower")
    upper_bc = dirichletBC(Domain, val=params["P_o"], var="R", target="upper")

    # 推理：先尝试 GPU，OOM 时自动回退 CPU
    for dev in ["cuda", "cpu"]:
        try:
            model = TorchCollocationSolver(device=dev)
            model.compile(
                cfg.layer_sizes, [f_model_FBNS], Domain, [lower_bc, upper_bc],
                u_model_switch=cfg.u_model_switch, two_output=True, none_zero=False,
                adapt_True=False, isAdaptive=False, MTL_adapt=False, PCGrad_true=True,
                Boundary_true=False,
                R_range=params["R_lim"], theta_range=params["theta_lim"],
                bc_switch=cfg.bc_switch, num_freq=cfg.num_fourier_freq,
                embed_dim=cfg.embed_dim
            )
            # 载入权重
            model.load_weights(model_path)
            break
        except RuntimeError as e:
            if "out of memory" in str(e) and dev == "cuda":
                torch.cuda.empty_cache()
                print(f"[PINN] GPU OOM，回退 CPU ...")
                continue
            raise
    print(f"[PINN] 权重加载自: {model_path}")

    # 推理
    p_pred, g_pred = model.predict(coords)
    P_pinn = p_pred.flatten()
    G_pinn = g_pred.flatten()

    print(f"[PINN] P: min={P_pinn.min():.6f}  max={P_pinn.max():.6f}")
    print(f"[PINN] g: min={G_pinn.min():.6f}  max={G_pinn.max():.6f}")
    cav_frac = (G_pinn > 1e-6).mean() * 100
    print(f"[PINN] 空化区占比: {cav_frac:.1f}%")
    print(f"[PINN] P·g 最大值（互补条件违反量）: {(P_pinn * G_pinn).max():.2e}")

    return P_pinn, G_pinn


# =============================================================================
# 3. 误差指标计算
# =============================================================================
def compute_metrics(P_fem, G_fem, P_pinn, G_pinn):
    eps = 1e-16

    def rel_l2(ref, pred):
        return np.linalg.norm(ref - pred) / (np.linalg.norm(ref) + eps)

    def rel_linf(ref, pred):
        return np.max(np.abs(ref - pred)) / (np.max(np.abs(ref)) + eps)

    metrics = {
        "P_rel_L2":   rel_l2(P_fem, P_pinn),
        "P_rel_Linf": rel_linf(P_fem, P_pinn),
        "G_rel_L2":   rel_l2(G_fem, G_pinn),
        "G_rel_Linf": rel_linf(G_fem, G_pinn),
        "P_MAE":      np.mean(np.abs(P_fem - P_pinn)),
        "P_RMSE":     np.sqrt(np.mean((P_fem - P_pinn) ** 2)),
        "G_MAE":      np.mean(np.abs(G_fem - G_pinn)),
        "G_RMSE":     np.sqrt(np.mean((G_fem - G_pinn) ** 2)),
    }

    mask_cav  = G_fem > 1e-6
    mask_full = ~mask_cav

    if mask_cav.any():
        metrics["P_rel_L2_cavRegion"]  = rel_l2(P_fem[mask_cav], P_pinn[mask_cav])
        metrics["G_rel_L2_cavRegion"]  = rel_l2(G_fem[mask_cav], G_pinn[mask_cav])
    if mask_full.any():
        metrics["P_rel_L2_fullRegion"] = rel_l2(P_fem[mask_full], P_pinn[mask_full])
        metrics["G_rel_L2_fullRegion"] = rel_l2(G_fem[mask_full], G_pinn[mask_full])

    metrics["complementarity_violation"] = float(np.max(P_pinn * G_pinn))

    mask_cav_pinn = G_pinn > 1e-6
    intersection  = (mask_cav & mask_cav_pinn).sum()
    union         = (mask_cav | mask_cav_pinn).sum()
    sum_both      = mask_cav.sum() + mask_cav_pinn.sum()
    metrics["cavitation_IoU"]  = intersection / (union + eps)
    metrics["cavitation_Dice"] = 2 * intersection / (sum_both + eps)

    return metrics


def print_metrics(metrics: dict):
    sep = "─" * 52
    print(f"\n{sep}")
    print("  误差指标汇总")
    print(sep)
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
            if k in metrics:
                print(f"    {k:<35s} = {metrics[k]:.6e}")
    print(f"{sep}\n")


# =============================================================================
# 4. 可视化
# =============================================================================
CMAP = cm.RdYlBu_r
FS   = 14
FS_T = 12


def _savefig(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  → 保存: {os.path.basename(path)}")


def plot_field_comparison(R_uniq, T_uniq,
                          P_fem_grid,  G_fem_grid,
                          P_pinn_grid, G_pinn_grid,
                          out_dir, dpi):
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")
    for data, title, fname in [
        (P_fem_grid,  r"FEM  $P(R,\theta)$",        "fig1a_P_fem.png"),
        (P_pinn_grid, r"PINN $\hat{P}(R,\theta)$",  "fig1b_P_pinn.png"),
        (G_fem_grid,  r"FEM  $g(R,\theta)$",         "fig1c_G_fem.png"),
        (G_pinn_grid, r"PINN $\hat{g}(R,\theta)$",  "fig1d_G_pinn.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        pc = ax.pcolormesh(RR, TT, data, shading="auto", cmap=CMAP)
        fig.colorbar(pc, ax=ax, pad=0.02)
        ax.set_title(title, fontsize=FS)
        ax.set_xlabel(r"$R$", fontsize=FS)
        ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12)
        ax.tick_params(labelsize=FS_T)
        fig.tight_layout()
        _savefig(fig, os.path.join(out_dir, fname), dpi)


def plot_error_maps(R_uniq, T_uniq,
                    P_fem_grid,  G_fem_grid,
                    P_pinn_grid, G_pinn_grid,
                    out_dir, dpi):
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")
    err_P = np.abs(P_fem_grid - P_pinn_grid)
    err_G = np.abs(G_fem_grid - G_pinn_grid)
    mask_cav = G_fem_grid > 0.01
    err_G_masked = np.where(mask_cav, err_G, np.nan)
    for err, label, fname, cmap_use in [
        (err_P,        r"$|P_{FEM}-\hat{P}_{PINN}|$",              "fig2a_err_P.png",  CMAP),
        (err_G_masked, r"$|g_{FEM}-\hat{g}_{PINN}|$  (FEM空化区)", "fig2b_err_G.png",  "hot_r"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        pc = ax.pcolormesh(RR, TT, err, shading="auto", cmap=cmap_use)
        cb = fig.colorbar(pc, ax=ax, pad=0.02)
        cb.ax.tick_params(labelsize=FS_T)
        ax.set_title(label, fontsize=FS)
        ax.set_xlabel(r"$R$", fontsize=FS)
        ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12)
        ax.tick_params(labelsize=FS_T)
        fig.tight_layout()
        _savefig(fig, os.path.join(out_dir, fname), dpi)


def plot_cavitation_boundary(R_uniq, T_uniq,
                              G_fem_grid, G_pinn_grid,
                              out_dir, dpi):
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Cavitation boundary — FEM vs PINN", fontsize=FS + 2)
    pc = ax.pcolormesh(RR, TT, G_pinn_grid, shading="auto", cmap="Blues", alpha=0.5)
    fig.colorbar(pc, ax=ax, label=r"PINN $\hat{g}$", pad=0.02)
    ax.contour(RR, TT, G_fem_grid,  levels=[1e-6], colors=["red"],  linewidths=1.8, linestyles="-")
    ax.contour(RR, TT, G_pinn_grid, levels=[1e-6], colors=["blue"], linewidths=1.8, linestyles="--")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="red",  linewidth=1.8, linestyle="-",  label="FEM  boundary"),
        Line2D([0], [0], color="blue", linewidth=1.8, linestyle="--", label="PINN boundary"),
    ]
    ax.legend(handles=legend_elements, fontsize=FS_T)
    ax.set_xlabel(r"$R$", fontsize=FS)
    ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12)
    ax.tick_params(labelsize=FS_T)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig3_cavitation_boundary.png"), dpi)


def plot_profiles(R_uniq, T_uniq,
                  P_fem_grid, G_fem_grid,
                  P_pinn_grid, G_pinn_grid,
                  out_dir, dpi):
    nR = len(R_uniq); nT = len(T_uniq)
    idx_R = nR // 2; idx_T = nT // 2
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Profile comparison — mid-R and mid-θ slices", fontsize=FS + 2)
    for ax, data_fem, data_pinn, title in [
        (axes[0,0], G_fem_grid[idx_R,:], G_pinn_grid[idx_R,:], f"P  at  R = {R_uniq[idx_R]:.4f}"),
        (axes[0,1], G_fem_grid[idx_R,:], G_pinn_grid[idx_R,:], f"g  at  R = {R_uniq[idx_R]:.4f}"),
        (axes[1,0], G_fem_grid[:,idx_T], G_pinn_grid[:,idx_T], fr"P  at  $\theta$ = {T_uniq[idx_T]:.4f}"),
        (axes[1,1], G_fem_grid[:,idx_T], G_pinn_grid[:,idx_T], fr"g  at  $\theta$ = {T_uniq[idx_T]:.4f}"),
    ]:
        ax.plot(T_uniq if idx_R else R_uniq, data_fem,  "r-",  lw=1.5, label="FEM")
        ax.plot(T_uniq if idx_R else R_uniq, data_pinn, "b--", lw=1.5, label="PINN")
        ax.set_title(title, fontsize=FS)
        ax.legend(fontsize=FS_T)
        ax.tick_params(labelsize=FS_T)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig5_profiles.png"), dpi)


def plot_scatter_correlation(P_fem, G_fem, P_pinn, G_pinn, out_dir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Correlation: FEM (reference) vs PINN (predicted)", fontsize=FS + 2)
    rng = np.random.default_rng(42)
    n_s = min(5000, len(P_fem))
    idx = rng.choice(len(P_fem), n_s, replace=False)
    for ax, (ref, pred, label) in zip(
        axes, [(P_fem[idx], P_pinn[idx], "P"), (G_fem[idx], G_pinn[idx], "g")]
    ):
        ax.scatter(ref, pred, s=4, alpha=0.4, color="steelblue")
        lo, hi = min(ref.min(), pred.min()), max(ref.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="ideal")
        ax.set_xlabel(f"FEM  {label}", fontsize=FS)
        ax.set_ylabel(f"PINN {label}", fontsize=FS)
        ax.set_title(f"Scatter — {label}", fontsize=FS)
        ax.legend(fontsize=FS_T)
        ax.tick_params(labelsize=FS_T)
        corr = np.corrcoef(ref, pred)[0, 1] ** 2
        ax.text(0.05, 0.92, f"$R^2$ = {corr:.6f}",
                transform=ax.transAxes, fontsize=FS_T,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig7_scatter_correlation.png"), dpi)


# =============================================================================
# 5. 保存指标
# =============================================================================
def save_metrics(metrics: dict, out_dir: str):
    path = os.path.join(out_dir, "metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("FEM vs PINN — 误差指标 (PyTorch)\n")
        f.write("=" * 52 + "\n")
        for k, v in metrics.items():
            f.write(f"  {k:<40s} = {v:.6e}\n")
    print(f"  → 指标保存: {os.path.basename(path)}")


# =============================================================================
# 6. 主函数
# =============================================================================
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg    = Config()
    params = compute_physical_params(cfg)

    print("\n[1/4] 读取 FEM 数据...")
    (R_pts, T_pts, P_fem, G_fem,
     P_fem_grid, G_fem_grid,
     R_uniq, T_uniq) = load_fem(args.fem_p, args.fem_g)

    print("\n[2/4] 加载 PINN 并推理...")
    coords = np.stack([R_pts, T_pts], axis=1).astype(np.float32)
    torch.cuda.empty_cache()
    with torch.no_grad():
        P_pinn, G_pinn = load_pinn_and_predict(args.model_path, coords, cfg, params)

    nR, nT = len(R_uniq), len(T_uniq)
    P_pinn_grid = P_pinn.reshape(nR, nT)
    G_pinn_grid = G_pinn.reshape(nR, nT)

    print("\n[3/4] 计算误差指标...")
    metrics = compute_metrics(P_fem, G_fem, P_pinn, G_pinn)
    print_metrics(metrics)
    save_metrics(metrics, args.out_dir)

    print("\n[4/4] 生成对比图...")
    dpi = args.dpi
    plot_field_comparison(R_uniq, T_uniq, P_fem_grid, G_fem_grid,
                          P_pinn_grid, G_pinn_grid, args.out_dir, dpi)
    plot_error_maps(R_uniq, T_uniq, P_fem_grid, G_fem_grid,
                    P_pinn_grid, G_pinn_grid, args.out_dir, dpi)
    plot_cavitation_boundary(R_uniq, T_uniq, G_fem_grid, G_pinn_grid, args.out_dir, dpi)
    print(f"\n完成！所有结果保存至: {os.path.abspath(args.out_dir)}\n")


if __name__ == "__main__":
    main()
