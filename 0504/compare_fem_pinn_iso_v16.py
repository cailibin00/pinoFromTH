"""
FEM vs PINN 对比分析脚本
================================================
用法:
    python compare_fem_pinn_iso_v16.py          # 默认使用 config_id=1
    python compare_fem_pinn_iso_v16.py 3        # 使用 config/c3.py 的配置

所有路径自动根据本脚本所在目录确定。

依赖：reynolds_pinn.py 与本脚本放在同一目录。
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import tensorflow as tf

# ── 从主 PINN 文件导入辅助函数，从 config 加载配置 ────────────────────────────
from config import get_config
from reynold_pinn import (
    compute_physical_params,
    create_H_func,
)
from tensordiffeq.models import CollocationSolverND

# =============================================================================
# 0. 路径配置（自动定位到脚本所在目录，无需手动修改）
# =============================================================================
# 脚本所在目录
_HERE = os.path.dirname(os.path.abspath(__file__))

def _p(filename):
    """拼接脚本目录与文件名"""
    return os.path.join(_HERE, filename)

def parse_args():
    parser = argparse.ArgumentParser(description="FEM vs PINN 对比分析")
    parser.add_argument("config_id", nargs="?", type=int, default=1,
                        help="配置序号 (对应 config/cN.py)，默认 1")
    parser.add_argument("--dpi", type=int, default=300, help="图片 DPI")
    return parser.parse_args()


# =============================================================================
# 1. 读取 FEM 数据
# =============================================================================
def load_fem(fem_p_path: str, fem_g_path: str):
    """
    读取 p_FBNS.txt / g_FBNS.txt
    格式：每行 [R  theta  value]，共 201×201 = 40401 行
    返回：
        R_pts, T_pts  : (40401,) 坐标向量
        P_fem, G_fem  : (40401,) 场值向量
        P_grid, G_grid: (201, 201) 网格矩阵（行=R轴，列=theta轴）
        R_unique, T_unique: (201,) 唯一坐标
    """
    raw_p = np.loadtxt(fem_p_path)   # (40401, 3)
    raw_g = np.loadtxt(fem_g_path)   # (40401, 3)

    R_pts = raw_p[:, 0]
    T_pts = raw_p[:, 1]
    P_fem = raw_p[:, 2]
    G_fem = raw_g[:, 2]

    n_R = len(np.unique(R_pts))      # 应为 201
    n_T = len(np.unique(T_pts))      # 应为 201

    # FEM 按行优先排列：R 固定，theta 递增
    P_grid = P_fem.reshape(n_R, n_T)   # shape (201, 201)
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
def load_pinn_and_predict(model_path: str, coords: np.ndarray, cfg, params: dict):
    """
    加载已保存的 PINN 权重，在 coords (N,2) 上推理。

    策略：用与训练完全相同的方式重建 CollocationSolverND，
    再 load_weights，这样网络结构、归一化层等完全一致。

    coords 列顺序：[:, 0]=R, [:, 1]=theta
    返回 P_pinn (N,), G_pinn (N,)
    """
    from reynold_pinn import create_pde_models, generate_groove_points
    from tensordiffeq.boundaries import dirichletBC
    from tensordiffeq.domains import DomainND

    # 重建与训练时完全一致的 model 结构
    H_func, theta_sym = create_H_func(params, cfg)
    f_model_FBNS, f_model_FB = create_pde_models(H_func, params)

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

    model = CollocationSolverND()
    u_model_switch = 8

    model.compile(
        cfg.layer_sizes, [f_model_FBNS], Domain, [lower_bc, upper_bc],
        u_model_switch=u_model_switch, two_output=True, none_zero=False,
        adapt_True=False, isAdaptive=False, MTL_adapt=False,
        Boundary_true=False,
        R_range=params["R_lim"], theta_range=params["theta_lim"],
        # 与训练时一致
        Act=cfg.Act, use_residual=cfg.use_residual,
        output_head_dim=cfg.output_head_dim, batch_size=cfg.batch_size,
        coslayer_mode=cfg.coslayer_mode,
        gamma_output_transform=getattr(cfg, "gamma_output_transform", "tanh_square"),
    )

    # 载入权重
    model.u_model.load_weights(model_path)
    print(f"[PINN] 权重加载自: {model_path}")

    # 推理
    X_tf = tf.constant(coords, dtype=tf.float32)
    out  = model.u_model(X_tf)     # list[tensor, tensor] 或 (N,2) tensor

    # 兼容两种输出格式
    if isinstance(out, (list, tuple)):
        P_pinn = out[0].numpy().flatten()
        G_pinn = out[1].numpy().flatten()
    else:
        arr    = out.numpy()
        P_pinn = arr[:, 0]
        G_pinn = arr[:, 1]

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
    """
    计算全局与分区域误差指标，返回指标字典。
    """
    eps = 1e-16

    def rel_l2(ref, pred):
        return np.linalg.norm(ref - pred) / (np.linalg.norm(ref) + eps)

    def rel_linf(ref, pred):
        return np.max(np.abs(ref - pred)) / (np.max(np.abs(ref)) + eps)

    # ── 全局指标 ──────────────────────────────────────────────────────────────
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

    # ── 空化区 / 全膜区分区误差 ───────────────────────────────────────────────
    mask_cav  = G_fem > 1e-6   # FEM 空化区
    mask_full = ~mask_cav       # FEM 全膜区

    if mask_cav.any():
        metrics["P_rel_L2_cavRegion"]  = rel_l2(P_fem[mask_cav], P_pinn[mask_cav])
        metrics["G_rel_L2_cavRegion"]  = rel_l2(G_fem[mask_cav], G_pinn[mask_cav])
    if mask_full.any():
        metrics["P_rel_L2_fullRegion"] = rel_l2(P_fem[mask_full], P_pinn[mask_full])
        metrics["G_rel_L2_fullRegion"] = rel_l2(G_fem[mask_full], G_pinn[mask_full])

    # ── JFO 互补条件违反量 ────────────────────────────────────────────────────
    metrics["complementarity_violation"] = float(np.max(P_pinn * G_pinn))

    # ── 空化区域重叠（Dice / IoU） ────────────────────────────────────────────
    cav_thresholds = [1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
    for th in cav_thresholds:
        mask_cav_th = G_fem > th
        mask_cav_pinn = G_pinn > th
        intersection = (mask_cav_th & mask_cav_pinn).sum()
        union = (mask_cav_th | mask_cav_pinn).sum()
        sum_both = mask_cav_th.sum() + mask_cav_pinn.sum()
        suffix = f"{th:.0e}".replace("-", "m")
        metrics[f"cavitation_IoU_gt_{suffix}"] = intersection / (union + eps)
        metrics[f"cavitation_Dice_gt_{suffix}"] = 2 * intersection / (sum_both + eps)

    metrics["cavitation_IoU"] = metrics["cavitation_IoU_gt_1em06"]
    metrics["cavitation_Dice"] = metrics["cavitation_Dice_gt_1em06"]

    # ── 周期边界一致性（FEM 应自动满足，PINN 需验证） ────────────────────────
    # 检查 PINN 在 theta=0 与 theta=theta_max 行的差异
    # （需要网格形式，在 plot 函数里单独检查）

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
    groups[4] = ("Cavitation shape", [
        "cavitation_IoU_gt_1em06", "cavitation_Dice_gt_1em06",
        "cavitation_IoU_gt_1em04", "cavitation_Dice_gt_1em04",
        "cavitation_IoU_gt_1em03", "cavitation_Dice_gt_1em03",
        "cavitation_IoU_gt_1em02", "cavitation_Dice_gt_1em02",
        "cavitation_IoU_gt_5em02", "cavitation_Dice_gt_5em02",
        "cavitation_IoU_gt_1em01", "cavitation_Dice_gt_1em01",
    ])
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
FS   = 14   # 轴标签字号
FS_T = 12   # 刻度字号


def _savefig(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  → 保存: {os.path.basename(path)}")


def plot_field_comparison(R_uniq, T_uniq,
                          P_fem_grid,  G_fem_grid,
                          P_pinn_grid, G_pinn_grid,
                          out_dir, dpi):
    """
    图1：FEM vs PINN 压力与空化率并排云图（2×2）
    """
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")   # (nR, nT)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("FEM vs PINN — field comparison", fontsize=FS + 2)

    datasets = [
        (P_fem_grid,  r"FEM  $P(R,\theta)$",        axes[0, 0]),
        (P_pinn_grid, r"PINN $\hat{P}(R,\theta)$",  axes[0, 1]),
        (G_fem_grid,  r"FEM  $g(R,\theta)$",         axes[1, 0]),
        (G_pinn_grid, r"PINN $\hat{g}(R,\theta)$",  axes[1, 1]),
    ]

    for data, title, ax in datasets:
        pc = ax.pcolormesh(RR, TT, data, shading="auto", cmap=CMAP)
        fig.colorbar(pc, ax=ax, pad=0.02)
        ax.set_title(title, fontsize=FS)
        ax.set_xlabel(r"$R$", fontsize=FS)
        ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12)
        ax.tick_params(labelsize=FS_T)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig1_field_comparison.png"), dpi)


def plot_error_maps(R_uniq, T_uniq,
                    P_fem_grid,  G_fem_grid,
                    P_pinn_grid, G_pinn_grid,
                    out_dir, dpi):
    """
    图2：逐点绝对误差热力图（压力 + 空化率）
    """
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")

    err_P = np.abs(P_fem_grid - P_pinn_grid)
    err_G = np.abs(G_fem_grid - G_pinn_grid)

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


def plot_cavitation_boundary(R_uniq, T_uniq,
                              G_fem_grid, G_pinn_grid,
                              out_dir, dpi):
    """
    图3：空化边界对比（FEM轮廓 vs PINN轮廓）
    """
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Cavitation boundary — FEM vs PINN", fontsize=FS + 2)

    # 背景：PINN 空化率
    pc = ax.pcolormesh(RR, TT, G_pinn_grid, shading="auto",
                       cmap="Blues", alpha=0.5)
    fig.colorbar(pc, ax=ax, label=r"PINN $\hat{g}$", pad=0.02)

    # FEM 边界（红色等值线）
    ax.contour(RR, TT, G_fem_grid,  levels=[1e-6],
               colors=["red"],  linewidths=1.8, linestyles="-",  label="FEM boundary")
    # PINN 边界（蓝色等值线）
    ax.contour(RR, TT, G_pinn_grid, levels=[1e-6],
               colors=["blue"], linewidths=1.8, linestyles="--", label="PINN boundary")

    # 图例
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


def plot_pressure_contour_overlay(R_uniq, T_uniq,
                                   P_fem_grid, P_pinn_grid,
                                   out_dir, dpi):
    """
    图4：压力等值线叠加对比（FEM实线 vs PINN虚线）
    """
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")

    # 共享等值线水平
    p_min = min(P_fem_grid.min(), P_pinn_grid.min())
    p_max = max(P_fem_grid.max(), P_pinn_grid.max())
    levels = np.linspace(p_min, p_max, 15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Pressure contour overlay", fontsize=FS + 2)

    cs_fem  = ax.contour(RR, TT, P_fem_grid,  levels=levels,
                         colors="red",  linewidths=1.2, linestyles="-")
    cs_pinn = ax.contour(RR, TT, P_pinn_grid, levels=levels,
                         colors="blue", linewidths=1.2, linestyles="--")

    ax.clabel(cs_fem,  fontsize=8, fmt="%.3f")
    ax.set_xlabel(r"$R$", fontsize=FS)
    ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12)
    ax.tick_params(labelsize=FS_T)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="red",  linewidth=1.2, linestyle="-",  label="FEM"),
        Line2D([0], [0], color="blue", linewidth=1.2, linestyle="--", label="PINN"),
    ]
    ax.legend(handles=legend_elements, fontsize=FS_T)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig4_pressure_contour_overlay.png"), dpi)


def plot_profiles(R_uniq, T_uniq,
                  P_fem_grid, G_fem_grid,
                  P_pinn_grid, G_pinn_grid,
                  out_dir, dpi):
    """
    图5：沿固定 R 和固定 θ 的截面线图（选取中间截面）
    """
    nR = len(R_uniq)
    nT = len(T_uniq)
    idx_R = nR // 2        # 径向中间
    idx_T = nT // 2        # 周向中间

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Profile comparison — mid-R and mid-θ slices", fontsize=FS + 2)

    # (0,0) 固定 R，P 沿 θ
    ax = axes[0, 0]
    ax.plot(T_uniq, P_fem_grid[idx_R, :],  "r-",  lw=1.5, label="FEM")
    ax.plot(T_uniq, P_pinn_grid[idx_R, :], "b--", lw=1.5, label="PINN")
    ax.set_title(f"P  at  R = {R_uniq[idx_R]:.4f}", fontsize=FS)
    ax.set_xlabel(r"$\theta$", fontsize=FS)
    ax.set_ylabel(r"$P$", fontsize=FS)
    ax.legend(fontsize=FS_T)
    ax.tick_params(labelsize=FS_T)

    # (0,1) 固定 R，g 沿 θ
    ax = axes[0, 1]
    ax.plot(T_uniq, G_fem_grid[idx_R, :],  "r-",  lw=1.5, label="FEM")
    ax.plot(T_uniq, G_pinn_grid[idx_R, :], "b--", lw=1.5, label="PINN")
    ax.set_title(f"g  at  R = {R_uniq[idx_R]:.4f}", fontsize=FS)
    ax.set_xlabel(r"$\theta$", fontsize=FS)
    ax.set_ylabel(r"$g$", fontsize=FS)
    ax.legend(fontsize=FS_T)
    ax.tick_params(labelsize=FS_T)

    # (1,0) 固定 θ，P 沿 R
    ax = axes[1, 0]
    ax.plot(R_uniq, P_fem_grid[:, idx_T],  "r-",  lw=1.5, label="FEM")
    ax.plot(R_uniq, P_pinn_grid[:, idx_T], "b--", lw=1.5, label="PINN")
    ax.set_title(fr"P  at  $\theta$ = {T_uniq[idx_T]:.4f}", fontsize=FS)
    ax.set_xlabel(r"$R$", fontsize=FS)
    ax.set_ylabel(r"$P$", fontsize=FS)
    ax.legend(fontsize=FS_T)
    ax.tick_params(labelsize=FS_T)

    # (1,1) 固定 θ，g 沿 R
    ax = axes[1, 1]
    ax.plot(R_uniq, G_fem_grid[:, idx_T],  "r-",  lw=1.5, label="FEM")
    ax.plot(R_uniq, G_pinn_grid[:, idx_T], "b--", lw=1.5, label="PINN")
    ax.set_title(fr"g  at  $\theta$ = {T_uniq[idx_T]:.4f}", fontsize=FS)
    ax.set_xlabel(r"$R$", fontsize=FS)
    ax.set_ylabel(r"$g$", fontsize=FS)
    ax.legend(fontsize=FS_T)
    ax.tick_params(labelsize=FS_T)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig5_profiles.png"), dpi)


def plot_periodic_check(R_uniq, T_uniq,
                         P_fem_grid, P_pinn_grid,
                         G_fem_grid, G_pinn_grid,
                         out_dir, dpi):
    """
    图6：周期边界一致性检查
    FEM 中 theta=0 与 theta_max 的值应完全相同；检查 PINN 是否满足。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"Periodic boundary check  ($\theta=0$  vs  $\theta_{max}$)", fontsize=FS + 2)

    for ax, (field_fem, field_pinn, label) in zip(
        axes,
        [(P_fem_grid, P_pinn_grid, "P"), (G_fem_grid, G_pinn_grid, "g")]
    ):
        # FEM
        ax.plot(R_uniq, field_fem[:, 0],  "r-",  lw=1.5, label=r"FEM  $\theta=0$")
        ax.plot(R_uniq, field_fem[:, -1], "r--", lw=1.2, label=r"FEM  $\theta_{max}$")
        # PINN
        ax.plot(R_uniq, field_pinn[:, 0],  "b-",  lw=1.5, label=r"PINN $\theta=0$")
        ax.plot(R_uniq, field_pinn[:, -1], "b--", lw=1.2, label=r"PINN $\theta_{max}$")

        ax.set_title(f"Field: {label}", fontsize=FS)
        ax.set_xlabel(r"$R$", fontsize=FS)
        ax.set_ylabel(label, fontsize=FS)
        ax.legend(fontsize=FS_T - 1)
        ax.tick_params(labelsize=FS_T)

        # 打印最大偏差
        pinn_diff = np.max(np.abs(field_pinn[:, 0] - field_pinn[:, -1]))
        fem_diff  = np.max(np.abs(field_fem[:, 0]  - field_fem[:, -1]))
        print(f"  周期性偏差 [{label}]  FEM: {fem_diff:.2e}   PINN: {pinn_diff:.2e}")

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig6_periodic_check.png"), dpi)


def plot_scatter_correlation(P_fem, G_fem, P_pinn, G_pinn, out_dir, dpi):
    """
    图7：散点相关图（FEM真值 vs PINN预测）
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Correlation: FEM (reference) vs PINN (predicted)", fontsize=FS + 2)

    # 随机抽样（全部 40401 点绘图会很慢）
    rng  = np.random.default_rng(42)
    n_s  = min(5000, len(P_fem))
    idx  = rng.choice(len(P_fem), n_s, replace=False)

    for ax, (ref, pred, label) in zip(
        axes,
        [(P_fem[idx], P_pinn[idx], "P"), (G_fem[idx], G_pinn[idx], "g")]
    ):
        ax.scatter(ref, pred, s=4, alpha=0.4, color="steelblue")
        lo = min(ref.min(), pred.min())
        hi = max(ref.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="ideal")
        ax.set_xlabel(f"FEM  {label}", fontsize=FS)
        ax.set_ylabel(f"PINN {label}", fontsize=FS)
        ax.set_title(f"Scatter — {label}", fontsize=FS)
        ax.legend(fontsize=FS_T)
        ax.tick_params(labelsize=FS_T)

        # Pearson R²
        corr = np.corrcoef(ref, pred)[0, 1] ** 2
        ax.text(0.05, 0.92, f"$R^2$ = {corr:.6f}",
                transform=ax.transAxes, fontsize=FS_T,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig7_scatter_correlation.png"), dpi)


def plot_jfo_complement(P_pinn, G_pinn, P_fem, G_fem, out_dir, dpi):
    """
    图8：JFO 互补条件可视化
    绘制 P vs g 的散点（理想情况下所有点沿两轴分布，不出现在 P>0 且 g>0 的区域）
    """
    rng = np.random.default_rng(0)
    n_s = min(8000, len(P_fem))
    idx = rng.choice(len(P_fem), n_s, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("JFO complementarity condition  ($P \\cdot g = 0$)", fontsize=FS + 2)

    for ax, (pp, gg, title) in zip(
        axes,
        [(P_fem[idx],  G_fem[idx],  "FEM"),
         (P_pinn[idx], G_pinn[idx], "PINN")]
    ):
        sc = ax.scatter(pp, gg, s=4, alpha=0.35, c=pp * gg,
                        cmap="hot_r", vmin=0)
        fig.colorbar(sc, ax=ax, label=r"$P \cdot g$")
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(r"$P$", fontsize=FS)
        ax.set_ylabel(r"$g$", fontsize=FS)
        ax.set_title(title, fontsize=FS)
        ax.tick_params(labelsize=FS_T)

        violation = np.max(pp * gg)
        ax.text(0.05, 0.92, f"max($P·g$) = {violation:.2e}",
                transform=ax.transAxes, fontsize=FS_T,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig8_jfo_complement.png"), dpi)


def plot_cavitation_isoline(R_uniq, T_uniq,
                             G_fem_grid, G_pinn_grid,
                             out_dir, dpi,
                             iso_level: float = 0.4):
    """
    图9：g = iso_level 等值线位置对比（FEM vs PINN）
    左图：等值线叠加在 g 场上，直观看位置偏差
    右图：每个 R 行的入口/出口侧 θ 位置偏差曲线，定量评估精度
    """
    RR, TT = np.meshgrid(R_uniq, T_uniq, indexing="ij")

    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 6))
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
    axes = [ax1, ax2]
    #fig.suptitle(f"Cavitation isoline  g = {iso_level}  — FEM vs PINN",
                 #fontsize=FS + 2)

    # ── 左图：完全保留v7颜色，只遮住白色过渡带 ──────────────────────────────
    ax = axes[0]
    vmax_g = max(G_fem_grid.max(), G_pinn_grid.max())

    # 自适应 iso_level：取 PINN 空化区 g 值的最大值和最小非零值的均值
    g_nonzero = G_pinn_grid[G_pinn_grid > 1e-6]
    iso_level_auto = (g_nonzero.max() + g_nonzero.min()) / 2
    print(f"  自适应 iso_level = {iso_level_auto:.4f}  "
          f"(PINN g: min={g_nonzero.min():.4f}, max={g_nonzero.max():.4f})")

    # 完全不动g值，保留v7原始颜色
    pc = ax.pcolormesh(RR, TT, G_pinn_grid, shading="auto",
                       cmap=CMAP, vmin=0, vmax=vmax_g)
    #fig.colorbar(pc, ax=ax, pad=0.02, label=r"PINN $\hat{g}$")
    # 在 g<iso_level 的过渡带上盖深蓝，消除白色
    ax.contourf(RR, TT, G_pinn_grid, levels=[-1e10, iso_level_auto],
                colors=["#313695"], alpha=1.0)
    # 在 g>=iso_level 的区域盖淡橘色（皮肤色），边界因底层渐变自然平滑
    ax.contourf(RR, TT, G_pinn_grid, levels=[iso_level_auto, 1e10],
                colors=["#FDAE6B"], alpha=1.0)

    # FEM 和 PINN 边界线
    ax.contour(RR, TT, G_fem_grid,  levels=[iso_level_auto],
               colors=["red"],  linewidths=2.0, linestyles="-")
    ax.contour(RR, TT, G_pinn_grid, levels=[iso_level_auto],
               colors=["blue"], linewidths=2.0, linestyles="--")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color="red",  lw=2.0, ls="-",  label="FEM"),
        Line2D([0],[0], color="blue", lw=2.0, ls="--", label="PINN"),
    ], fontsize=FS_T, loc='upper right', bbox_to_anchor=(1.0, 0.92))
    ax.set_title("Cavitation region", fontsize=FS)
    ax.set_xlabel(r"$R$", fontsize=FS)
    ax.set_ylabel(r"$\theta$", fontsize=FS, rotation=0, labelpad=12)
    ax.tick_params(labelsize=FS_T)

    # ── 右图：每个 R 行的 θ 位置偏差 ─────────────────────────────────────────
    ax = axes[1]

    def find_crossings(G_grid, level, R_uniq, T_uniq):
        """对每个R行，用线性插值找g=level的入口和出口θ位置"""
        R_valid, t_in_list, t_out_list = [], [], []
        for i in range(len(R_uniq)):
            g_row = G_grid[i, :]
            above = g_row >= level
            cross = np.where(np.diff(above.astype(int)) != 0)[0]
            if len(cross) < 2:
                continue
            def interp_theta(idx):
                t0, t1 = T_uniq[idx], T_uniq[idx + 1]
                g0, g1 = g_row[idx], g_row[idx + 1]
                if abs(g1 - g0) < 1e-12:
                    return (t0 + t1) / 2
                return t0 + (level - g0) / (g1 - g0) * (t1 - t0)
            R_valid.append(R_uniq[i])
            t_in_list.append(interp_theta(cross[0]))
            t_out_list.append(interp_theta(cross[-1]))
        return np.array(R_valid), np.array(t_in_list), np.array(t_out_list)

    R_f, t_f_in, t_f_out = find_crossings(G_fem_grid,  iso_level_auto, R_uniq, T_uniq)
    R_p, t_p_in, t_p_out = find_crossings(G_pinn_grid, iso_level_auto, R_uniq, T_uniq)

    # 找共有的 R 行
    R_f_round = np.round(R_f, 8)
    R_p_round = np.round(R_p, 8)
    R_common  = np.intersect1d(R_f_round, R_p_round)

    if len(R_common) == 0:
        ax.text(0.5, 0.5,
                f"g={iso_level_auto:.2f} 等值线在FEM或PINN中不存在\n请调整 iso_level",
                transform=ax.transAxes, ha='center', va='center', fontsize=FS)
        print(f"\n  警告：g={iso_level_auto:.2f} 在FEM或PINN中找不到等值线，"
              f"请确认该值在有效范围内（FEM g max={G_fem_grid.max():.3f}，"
              f"PINN g max={G_pinn_grid.max():.3f}）")
    else:
        idx_f = [np.argmin(np.abs(R_f_round - r)) for r in R_common]
        idx_p = [np.argmin(np.abs(R_p_round - r)) for r in R_common]

        diff_in  = (t_p_in[idx_p]  - t_f_in[idx_f])  * 1e3   # ×10⁻³ rad
        diff_out = (t_p_out[idx_p] - t_f_out[idx_f]) * 1e3

        ax.plot(diff_in,  R_common, "b-",  lw=1.8, label="入口侧偏差")
        ax.plot(diff_out, R_common, "r--", lw=1.8, label="出口侧偏差")
        ax.axvline(0, color="k", lw=0.8, ls=":", label="零偏差线")
        ax.fill_betweenx(R_common, diff_in,  0, alpha=0.15, color="blue")
        ax.fill_betweenx(R_common, diff_out, 0, alpha=0.15, color="red")

        ax.set_xlabel(r"$\Delta\theta \times 10^{-3}$ rad  (PINN $-$ FEM)",
                      fontsize=FS)
        ax.set_ylabel(r"$R$", fontsize=FS, rotation=0, labelpad=12)
        ax.set_title(f"Isoline position error  (g={iso_level_auto:.2f})", fontsize=FS)
        ax.legend(fontsize=FS_T)
        ax.tick_params(labelsize=FS_T)
        ax.grid(True, alpha=0.3)

        # 数值统计
        print(f"\n  g={iso_level_auto:.2f} 等值线位置偏差（PINN − FEM）：")
        print(f"    入口侧  mean={diff_in.mean():+.3f}e-3 rad  "
              f"std={diff_in.std():.3f}e-3  "
              f"max|err|={np.abs(diff_in).max():.3f}e-3 rad")
        print(f"    出口侧  mean={diff_out.mean():+.3f}e-3 rad  "
              f"std={diff_out.std():.3f}e-3  "
              f"max|err|={np.abs(diff_out).max():.3f}e-3 rad")

    fig1.tight_layout()
    fig2.tight_layout()
    _savefig(fig1, os.path.join(out_dir, f"fig9_isoline_left_g{int(iso_level_auto*100):02d}.png"), dpi)
    _savefig(fig2, os.path.join(out_dir, f"fig9_isoline_right_g{int(iso_level_auto*100):02d}.png"), dpi)


# =============================================================================
# 5. 保存数值指标到文本
# =============================================================================
def save_metrics(metrics: dict, out_dir: str):
    path = os.path.join(out_dir, "metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("FEM vs PINN — 误差指标\n")
        f.write("=" * 52 + "\n")
        for k, v in metrics.items():
            f.write(f"  {k:<40s} = {v:.6e}\n")
    print(f"  → 指标保存: {os.path.basename(path)}")


# =============================================================================
# 6. 主函数
# =============================================================================
def main():
    args = parse_args()
    cfg = get_config(args.config_id)
    print(f"[Config] 加载配置 C{args.config_id}: Act={cfg.Act}, residual={cfg.use_residual}, "
          f"output={cfg.output_dir}")

    # ── 根据 config 推导路径 ──────────────────────────────────────────────────
    model_path = os.path.join(_HERE, cfg.output_dir, "checkpoints", "epochs_best_model")
    out_dir    = os.path.join(_HERE, cfg.output_dir, "comparison_results")
    fem_p      = _p("p_FBNS.txt")
    fem_g      = _p("g_FBNS.txt")
    dpi        = args.dpi

    os.makedirs(out_dir, exist_ok=True)

    params = compute_physical_params(cfg)

    # ── 读取 FEM ──────────────────────────────────────────────────────────────
    print("\n[1/4] 读取 FEM 数据...")
    (R_pts, T_pts, P_fem, G_fem,
     P_fem_grid, G_fem_grid,
     R_uniq, T_uniq) = load_fem(fem_p, fem_g)

    # ── PINN 推理 ─────────────────────────────────────────────────────────────
    print("\n[2/4] 加载 PINN 并推理...")
    coords = np.stack([R_pts, T_pts], axis=1).astype(np.float32)
    P_pinn, G_pinn = load_pinn_and_predict(
        model_path, coords, cfg, params
    )


    # 整形为网格（与 FEM 一致：行=R，列=theta）
    nR, nT       = len(R_uniq), len(T_uniq)
    P_pinn_grid  = P_pinn.reshape(nR, nT)
    G_pinn_grid  = G_pinn.reshape(nR, nT)

    # ── 误差指标 ──────────────────────────────────────────────────────────────
    print("\n[3/4] 计算误差指标...")
    metrics = compute_metrics(P_fem, G_fem, P_pinn, G_pinn)
    print_metrics(metrics)
    save_metrics(metrics, out_dir)

    # ── 可视化 ────────────────────────────────────────────────────────────────
    print("\n[4/4] 生成对比图...")

    plot_field_comparison(R_uniq, T_uniq,
                          P_fem_grid, G_fem_grid,
                          P_pinn_grid, G_pinn_grid,
                          out_dir, dpi)

    plot_error_maps(R_uniq, T_uniq,
                    P_fem_grid, G_fem_grid,
                    P_pinn_grid, G_pinn_grid,
                    out_dir, dpi)

    plot_cavitation_boundary(R_uniq, T_uniq,
                              G_fem_grid, G_pinn_grid,
                              out_dir, dpi)

    plot_pressure_contour_overlay(R_uniq, T_uniq,
                                   P_fem_grid, P_pinn_grid,
                                   out_dir, dpi)

    plot_profiles(R_uniq, T_uniq,
                  P_fem_grid, G_fem_grid,
                  P_pinn_grid, G_pinn_grid,
                  out_dir, dpi)

    plot_periodic_check(R_uniq, T_uniq,
                         P_fem_grid, P_pinn_grid,
                         G_fem_grid, G_pinn_grid,
                         out_dir, dpi)

    plot_scatter_correlation(P_fem, G_fem, P_pinn, G_pinn, out_dir, dpi)

    plot_jfo_complement(P_pinn, G_pinn, P_fem, G_fem, out_dir, dpi)

    plot_cavitation_isoline(R_uniq, T_uniq,
                             G_fem_grid, G_pinn_grid,
                             out_dir, dpi,
                             iso_level=0.5)

    print(f"\n完成！所有结果保存至: {os.path.abspath(out_dir)}\n")


if __name__ == "__main__":
    main()
