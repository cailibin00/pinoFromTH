"""
FEM vs PINN 对比分析脚本
================================================
直接运行即可，无需命令行参数。
所有路径自动根据本脚本所在目录确定。

依赖：reynolds_pinn.py 与本脚本放在同一目录。
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import tensorflow as tf

# ── 从主 PINN 文件导入配置与辅助函数 ──────────────────────────────────────────
from reynold_pinn import (
    Config,
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

class Args:
    model_path = _p("epochs_best_model")
    fem_p      = _p("p_FBNS.txt")
    fem_g      = _p("g_FBNS.txt")
    out_dir    = _p("comparison_results")
    n_grid     = 201
    dpi        = 300

def parse_args():
    return Args()


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
def load_pinn_and_predict(model_path: str, coords: np.ndarray, cfg: Config, params: dict):
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
    model.compile(
        cfg.layer_sizes, [f_model_FBNS], Domain, [lower_bc, upper_bc],
        u_model_switch=8, two_output=True, none_zero=False, adapt_True=False,
        isAdaptive=False, MTL_adapt=False, PCGrad_true=True, Boundary_true=False,
        R_range=params["R_lim"], theta_range=params["theta_lim"]
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
    mask_cav_pinn = G_pinn > 1e-6
    intersection  = (mask_cav & mask_cav_pinn).sum()
    union         = (mask_cav | mask_cav_pinn).sum()
    sum_both      = mask_cav.sum() + mask_cav_pinn.sum()
    metrics["cavitation_IoU"]  = intersection / (union + eps)
    metrics["cavitation_Dice"] = 2 * intersection / (sum_both + eps)

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
    图1：FEM vs PINN 压力与空化率，每张单独输出
    """
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
    """
    图2：逐点绝对误差，每张单独输出
    压力误差使用 CMAP，空化率误差使用 hot_r
    """
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
    os.makedirs(args.out_dir, exist_ok=True)

    cfg    = Config()
    params = compute_physical_params(cfg)

    # ── 读取 FEM ──────────────────────────────────────────────────────────────
    print("\n[1/4] 读取 FEM 数据...")
    (R_pts, T_pts, P_fem, G_fem,
     P_fem_grid, G_fem_grid,
     R_uniq, T_uniq) = load_fem(args.fem_p, args.fem_g)

    # ── PINN 推理 ─────────────────────────────────────────────────────────────
    print("\n[2/4] 加载 PINN 并推理...")
    coords = np.stack([R_pts, T_pts], axis=1).astype(np.float32)
    P_pinn, G_pinn = load_pinn_and_predict(
        args.model_path, coords, cfg, params
    )

    
    # 整形为网格（与 FEM 一致：行=R，列=theta）
    nR, nT       = len(R_uniq), len(T_uniq)
    P_pinn_grid  = P_pinn.reshape(nR, nT)
    G_pinn_grid  = G_pinn.reshape(nR, nT)

    # ── 误差指标 ──────────────────────────────────────────────────────────────
    print("\n[3/4] 计算误差指标...")
    metrics = compute_metrics(P_fem, G_fem, P_pinn, G_pinn)
    print_metrics(metrics)
    save_metrics(metrics, args.out_dir)

    # ── 可视化 ────────────────────────────────────────────────────────────────
    print("\n[4/4] 生成对比图...")
    dpi = args.dpi

    plot_field_comparison(R_uniq, T_uniq,
                          P_fem_grid, G_fem_grid,
                          P_pinn_grid, G_pinn_grid,
                          args.out_dir, dpi)

    plot_error_maps(R_uniq, T_uniq,
                    P_fem_grid, G_fem_grid,
                    P_pinn_grid, G_pinn_grid,
                    args.out_dir, dpi)

    plot_cavitation_boundary(R_uniq, T_uniq,
                              G_fem_grid, G_pinn_grid,
                              args.out_dir, dpi)

    plot_pressure_contour_overlay(R_uniq, T_uniq,
                                   P_fem_grid, P_pinn_grid,
                                   args.out_dir, dpi)

    plot_profiles(R_uniq, T_uniq,
                  P_fem_grid, G_fem_grid,
                  P_pinn_grid, G_pinn_grid,
                  args.out_dir, dpi)

    plot_periodic_check(R_uniq, T_uniq,
                         P_fem_grid, P_pinn_grid,
                         G_fem_grid, G_pinn_grid,
                         args.out_dir, dpi)

    plot_scatter_correlation(P_fem, G_fem, P_pinn, G_pinn, args.out_dir, dpi)

    plot_jfo_complement(P_pinn, G_pinn, P_fem, G_fem, args.out_dir, dpi)

    print(f"\n完成！所有结果保存至: {os.path.abspath(args.out_dir)}\n")


if __name__ == "__main__":
    main()
