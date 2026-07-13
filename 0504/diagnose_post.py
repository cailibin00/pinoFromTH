"""
训练后诊断分析脚本 — 独立运行，加载已训练的模型进行全面分析。

用法:
    cd E:\a_lab\lab_Pro\pinoFromTH\0504
    python diagnose_post.py

配置: 修改本文件顶部的 AnalysisConfig 类。
输出: {output_dir}/diagnostics/post_training/
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from reynold_pinn import (
    Config, compute_physical_params, create_H_func,
    create_pde_models, generate_groove_points,
)
from tensordiffeq.models import CollocationSolverND
from tensordiffeq.boundaries import dirichletBC
from tensordiffeq.domains import DomainND
from tensordiffeq.utils import get_tf_model
from diagnose import (
    post_training_analysis,
    gradient_impact_detailed_analysis,
    ReynoldsTermAnalyzer,
    LayerStatsCollector,
    _stats, _hist_counts, _to_numpy,
    ensure_dir,
)


class AnalysisConfig:
    """分析配置 — 修改此处的路径指向你的训练输出。"""

    # 要分析的输出目录 (对应训练时的 Config.output_dir)
    output_dir = os.path.join(SCRIPT_DIR, "output_tf")

    # 模型权重路径 (checkpoint)
    model_path = os.path.join(output_dir, "checkpoints", "epochs_best_model")

    # FEM 参考数据
    fem_p = os.path.join(SCRIPT_DIR, "p_FBNS.txt")
    fem_g = os.path.join(SCRIPT_DIR, "g_FBNS.txt")

    # 分析用网格密度
    n_grid = 201

    # 梯度影响力分析采样次数
    grad_n_samples = 5


def build_model_from_checkpoint(acfg):
    """
    从 checkpoint 重建模型。

    步骤:
      1. 读取训练 Config + params
      2. 重建 Domain + BCs
      3. 创建 CollocationSolverND 并 compile
      4. 加载权重
    """
    cfg = Config()
    params = compute_physical_params(cfg)

    # 重建计算图
    H_func, theta_sym = create_H_func(params, cfg)
    f_model_FBNS, f_model_FB = create_pde_models(H_func, params)

    Domain = DomainND(["R", "theta"])
    Domain.add("R", params['R_lim'], cfg.domain_fidelity)
    Domain.add("theta", params['theta_lim'], cfg.domain_fidelity)
    Domain.X_f = Domain.generate_collocation_points(cfg.N_f, 1)

    add_R, add_theta = generate_groove_points(theta_sym, params, cfg)
    Domain.X_f = np.concatenate(
        [Domain.X_f, np.concatenate([add_R, add_theta], 1)], 0
    )
    print(f"  Total collocation points: {len(Domain.X_f)}")

    lower_bc = dirichletBC(Domain, val=params['P_i'], var='R', target="lower")
    upper_bc = dirichletBC(Domain, val=params['P_o'], var='R', target="upper")

    if cfg.core == "pikan":
        u_model_switch = 13
    else:
        u_model_switch = 8

    model = CollocationSolverND()
    model.compile(
        cfg.layer_sizes, [f_model_FBNS], Domain, [lower_bc, upper_bc],
        u_model_switch=u_model_switch, two_output=True, none_zero=False,
        adapt_True=False, isAdaptive=False, MTL_adapt=False,
        PCGrad_true=True, Boundary_true=False,
        R_range=params['R_lim'], theta_range=params['theta_lim'],
        Act=cfg.Act, use_residual=cfg.use_residual,
        output_head_dim=cfg.output_head_dim, batch_size=cfg.batch_size,
        coslayer_mode=cfg.coslayer_mode,
        kan_grid_size=cfg.kan_grid_size, kan_spline_order=cfg.kan_spline_order,
        pikan_layer_sizes=cfg.pikan_layer_sizes,
    )

    # 加载权重
    if os.path.exists(acfg.model_path + ".index"):
        model.u_model.load_weights(acfg.model_path)
        print(f"  Weights loaded from: {acfg.model_path}")
    else:
        print(f"  WARNING: Checkpoint not found at {acfg.model_path}")
        print(f"  Using randomly initialized weights!")

    model.f_model_FB = tf.function(f_model_FB)
    model.f_model_list = [get_tf_model(f_model_FBNS)]

    return model, cfg, params, H_func


def main():
    acfg = AnalysisConfig()
    output_dir = acfg.output_dir
    ensure_dir(output_dir)

    print("=" * 60)
    print("  PINN Post-Training Diagnostic Analysis")
    print("=" * 60)
    print(f"  Output dir: {output_dir}")
    print(f"  Model path: {acfg.model_path}")

    # 1. 构建模型 + 加载权重
    print("\n[1/5] Building model from checkpoint ...")
    model, cfg, params, H_func = build_model_from_checkpoint(acfg)

    # 2. 全量 PDE 逐项分析
    print("\n[2/5] Full PDE term decomposition ...")
    pde_analyzer = ReynoldsTermAnalyzer(H_func, params, cfg)
    X_f = model.domain.X_f
    pde_stats = pde_analyzer.analyze(model.u_model, X_f)

    pde_path = os.path.join(output_dir, 'diagnostics', 'post_training', 'pde_terms_full.json')
    ensure_dir(os.path.dirname(pde_path))
    with open(pde_path, 'w') as f:
        json.dump(pde_stats, f, indent=2, default=str)

    print("\n  PDE Term Magnitudes (full data):")
    print(f"  {'Term':<12s} {'mean':>14s} {'std':>14s} {'max':>14s} {'p99':>14s} {'L1':>14s}")
    print("  " + "-" * 82)
    for term_name in ["part_1", "part_2", "part_3_1", "part_3_2", "stab", "f_p"]:
        t = pde_stats.get(term_name, {})
        print(f"  {term_name:<12s} {t.get('mean',0):>14.4e} {t.get('std',0):>14.4e} "
              f"{t.get('max',0):>14.4e} {t.get('p99',0):>14.4e} {t.get('L1',0):>14.4e}")

    # 3. 逐层权重 + 激活分布
    print("\n[3/5] Layer-wise weight & activation analysis ...")
    collector = LayerStatsCollector(model.u_model)
    n_sample = min(len(X_f), 5000)
    idx = np.random.choice(len(X_f), n_sample, replace=False)
    layer_stats = collector.full_analysis(X_f[idx])

    layer_path = os.path.join(output_dir, 'diagnostics', 'post_training', 'layer_stats.json')
    with open(layer_path, 'w') as f:
        json.dump(layer_stats, f, indent=2, default=str)

    # 打印关键层权重
    print("\n  Key Weight Distributions:")
    w_stats = layer_stats.get("weights", {})
    for name, info in sorted(w_stats.items()):
        if not name.endswith('_hist') and ('dense' in name.lower() or 'cos' in name.lower() or 'fc' in name.lower()):
            print(f"    {name:<55s}  mean={info.get('mean',0):>12.4e}  "
                  f"std={info.get('std',0):>12.4e}  max={info.get('max',0):>12.4e}")

    # 4. 梯度影响力
    print("\n[4/5] Gradient impact per-component ...")
    gradient_impact_detailed_analysis(model, params, cfg, output_dir,
                                      n_samples=acfg.grad_n_samples)

    # 5. 输出分布 + JFO 互补
    print("\n[5/5] Output distribution & JFO complementarity ...")
    R_tf = tf.constant(X_f[:, 0:1], dtype=tf.float32)
    theta_tf = tf.constant(X_f[:, 1:2], dtype=tf.float32)
    u_preds = model.u_model(tf.concat([R_tf, theta_tf], 1))
    p_val = _to_numpy(u_preds[0])
    g_val = _to_numpy(u_preds[1])
    fb_val = p_val + g_val - np.sqrt(p_val ** 2 + g_val ** 2)
    pg_val = p_val * g_val

    out_stats = {
        "P": _stats(p_val, name="P"),
        "gamma": _stats(g_val, name="gamma"),
        "FB_complement": _stats(fb_val, name="FB(p,g)"),
        "P_times_gamma": _stats(pg_val, name="P·γ"),
    }
    out_path = os.path.join(output_dir, 'diagnostics', 'post_training', 'output_stats.json')
    with open(out_path, 'w') as f:
        json.dump(out_stats, f, indent=2, default=str)

    print(f"\n  {'Output':<20s} {'mean':>14s} {'std':>14s} {'max':>14s}")
    print("  " + "-" * 62)
    for name, s in out_stats.items():
        print(f"  {name:<20s} {s['mean']:>14.4e} {s['std']:>14.4e} {s['max']:>14.4e}")

    # 加载 FEM 对比（如果有的话）
    if os.path.exists(acfg.fem_p) and os.path.exists(acfg.fem_g):
        print("\n  --- FEM Comparison ---")
        raw_p = np.loadtxt(acfg.fem_p)
        p_fem = raw_p[:, 2]
        print(f"  FEM P:  mean={p_fem.mean():.4e}  max={p_fem.max():.4e}  min={p_fem.min():.4e}")
        print(f"  PINN P: mean={p_val.mean():.4e}  max={p_val.max():.4e}  min={p_val.min():.4e}")
        p_rel = np.linalg.norm(p_fem - p_val.flatten()[:len(p_fem)]) / (np.linalg.norm(p_fem) + 1e-16)
        print(f"  P_rel_L2 (approximate): {p_rel:.6e}")

    print(f"\n  All results saved → {os.path.join(output_dir, 'diagnostics', 'post_training')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
