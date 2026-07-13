"""
诊断模块 — 用于定位 PyTorch/PINN 训练不收敛的根因。

功能:
  1. Reynolds 方程逐项分解统计 (PDE 各项的量级)
  2. 逐层输入输出分布 + 权重分布统计
  3. 逐层/逐组件梯度影响力分析
  4. 训练中阶段性快照 + 训练后全量分析

所有统计严格保存为 .npz / .json / .txt，便于后续对比分析。

用法:
  from diagnose import TrainingDiagnostics, post_training_analysis

  训练中:
      diag = TrainingDiagnostics(model, params, cfg, output_dir)
      ...
      diag.snapshot(epoch)      # 每个阶段拍快照
      diag.finalize()            # 训练结束汇总

  训练后:
      post_training_analysis(model, params, cfg, output_dir)
"""

import os
import json
import numpy as np
import tensorflow as tf
from collections import defaultdict

# =============================================================================
# 辅助函数
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _to_numpy(x):
    """安全转 numpy，处理 Tensor / float / list 等。"""
    if hasattr(x, 'numpy'):
        return x.numpy()
    if isinstance(x, (list, tuple)):
        return np.array([_to_numpy(v) for v in x])
    return x


def _stats(arr, name="", precision=4):
    """计算数组的基础统计量。"""
    arr = np.asarray(arr).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"name": name, "mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "p99": 0}
    fmt = lambda v: round(float(v), precision)
    return {
        "name": name,
        "mean":   fmt(np.mean(arr)),
        "std":    fmt(np.std(arr)),
        "min":    fmt(np.min(arr)),
        "max":    fmt(np.max(arr)),
        "median": fmt(np.median(arr)),
        "p99":    fmt(np.percentile(np.abs(arr), 99)),
        "L1":     fmt(np.sum(np.abs(arr))),
        "L2":     fmt(np.sqrt(np.sum(arr ** 2))),
    }


def _hist_counts(arr, bins=20):
    """直方图计数 (用于保存可读的分布)。"""
    arr = np.asarray(arr).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"bin_edges": [], "counts": []}
    counts, edges = np.histogram(arr, bins=bins)
    return {"bin_edges": edges.tolist(), "counts": counts.tolist()}


def _grad_norm(grad):
    """计算梯度的 L2 范数 (处理 None / 0-tensor)。"""
    if grad is None:
        return 0.0
    return float(tf.sqrt(tf.reduce_sum(grad ** 2) + 1e-16).numpy())


# =============================================================================
# 1. Reynolds 方程逐项统计分析
# =============================================================================

class ReynoldsTermAnalyzer:
    """
    将 f_model_FBNS 的残差分解为五项，分别统计每项的量级。

    f_p = part_1 + part_2 + part_3_1 + part_3_2 + stab

    其中:
      part_1   = ∇_R(R · H³ · ∇_R p) / R        — 径向压力扩散
      part_2   = ∇_θ(H³ · ∇_θ p) / R²            — 周向压力扩散
      part_3_1 = -Λ · ∇_θ(H)                      — 楔形效应 (膜厚梯度)
      part_3_2 = -Λ · ∇_θ(-γ · H)                 — 空化剪切项
      stab     = ∇²_θ(γ) · τ · τ₂                 — 稳定项
    """

    def __init__(self, H_func, params, cfg):
        self.H_func = H_func
        self.Lambda = float(params['Lambda'].numpy()) if hasattr(params['Lambda'], 'numpy') else float(params['Lambda'])
        self.epsilon = 0.1

    def decompose(self, u_model, R, theta):
        """将残差分解为五项，返回 dict。"""
        return self._decompose_graph(u_model, R, theta)

    @tf.function
    def _decompose_graph(self, u_model, R, theta):
        """tf.function 包裹的分解，支持 tf.gradients。"""
        p_vector = u_model(tf.concat([R, theta], 1))
        p, gamma = p_vector[0], p_vector[1]
        H = self.H_func(R, theta)

        # 梯度
        p_R     = tf.gradients(p, R)[0]
        p_theta = tf.gradients(p, theta)[0]

        # 各项分别计算
        part_1   = tf.gradients(R * H ** 3 * p_R, R)[0] / (R + 1e-16)
        part_2   = tf.gradients(H ** 3 * p_theta, theta)[0] / (R ** 2 + 1e-16)
        part_3_1 = -self.Lambda * tf.gradients(H, theta)[0]
        part_3_2 = -self.Lambda * tf.gradients(-gamma * H, theta)[0]

        # 稳定项
        div_gamma   = tf.gradients(gamma, theta)[0]
        div_2_gamma = tf.gradients(div_gamma, theta)[0]
        div_p       = tf.gradients(p, theta)[0]
        tau   = tf.stop_gradient((tf.math.abs(div_gamma) - div_gamma) * self.epsilon)
        tau_2 = tf.stop_gradient((div_p - tf.math.abs(div_p)) * self.epsilon)
        stab  = div_2_gamma * tau * tau_2

        f_p   = part_1 + part_2 + part_3_1 + part_3_2 + stab

        return {
            "part_1":   part_1,
            "part_2":   part_2,
            "part_3_1": part_3_1,
            "part_3_2": part_3_2,
            "stab":     stab,
            "f_p":      f_p,
            "p":        p,
            "gamma":    gamma,
            "H":        H,
        }

    def analyze(self, u_model, X_f):
        """在全部配点上分解并统计。"""
        R_tf     = tf.constant(X_f[:, 0:1], dtype=tf.float32)
        theta_tf = tf.constant(X_f[:, 1:2], dtype=tf.float32)
        terms = self.decompose(u_model, R_tf, theta_tf)

        stats = {}
        for key, val in terms.items():
            stats[key] = _stats(val, name=key)
            stats[key + "_hist"] = _hist_counts(val)
        return stats


# =============================================================================
# 2. 逐层输入输出 + 权重分布统计
# =============================================================================

class LayerStatsCollector:
    """
    通过钩子捕获模型各层的输入/输出/权重分布。

    TF Keras 模型: 逐层遍历 model.layers，用 tf.GradientTape 或直接调用
    子模型来捕获中间激活值。
    """

    def __init__(self, u_model):
        self.u_model = u_model
        self.layer_names = [l.name for l in u_model.layers if len(l.weights) > 0]

    def _get_layer_weights_stats(self):
        """遍历所有层，收集权重的统计量。"""
        results = {}
        for layer in self.u_model.layers:
            for w in layer.weights:
                name = f"{layer.name}/{w.name.replace(':', '_')}"
                val  = _to_numpy(w)
                results[name] = _stats(val, name=name)
                results[name + "_hist"] = _hist_counts(val)
        return results

    def _build_intermediate_model(self):
        """构建一个输出所有有参数层的中间激活的模型。"""
        outputs = {}
        for layer in self.u_model.layers:
            if len(layer.weights) > 0 and not isinstance(layer, tf.keras.layers.InputLayer):
                outputs[layer.name] = layer.output

        # 找到实际可用的输出
        available_outputs = {}
        for name, out in outputs.items():
            try:
                available_outputs[name] = out
            except Exception:
                pass

        if available_outputs:
            try:
                return tf.keras.Model(
                    inputs=self.u_model.input,
                    outputs=list(available_outputs.values())
                ), list(available_outputs.keys())
            except Exception:
                return None, []
        return None, []

    def analyze_activations(self, X_sample):
        """
        在前向传播中捕获各层的输出激活值。

        由于模型有多个输出 (P, γ)，我们直接使用 functional API
        构建中间模型捕获隐藏层。

        对 TF2 的复杂模型，使用更稳健的方式：分别创建子模型。
        """
        X_tf = tf.constant(X_sample, dtype=tf.float32)
        results = {}

        # 方式 1: 尝试构建中间模型
        inter_model, inter_names = self._build_intermediate_model()
        if inter_model is not None:
            try:
                inter_outputs = inter_model(X_tf, training=False)
                if not isinstance(inter_outputs, (list, tuple)):
                    inter_outputs = [inter_outputs]
                for name, out in zip(inter_names, inter_outputs):
                    val = _to_numpy(out)
                    results[f"act_{name}"] = _stats(val, name=f"act_{name}")
                    results[f"act_{name}_hist"] = _hist_counts(val)
                return results
            except Exception:
                pass

        # 方式 2: 对每个 Dense/Conv 层单独构建模型
        for layer in self.u_model.layers:
            if hasattr(layer, 'output') and len(layer.weights) > 0:
                try:
                    sub_model = tf.keras.Model(
                        inputs=self.u_model.input,
                        outputs=layer.output
                    )
                    out = sub_model(X_tf, training=False)
                    val = _to_numpy(out)
                    results[f"act_{layer.name}"] = _stats(val, name=f"act_{layer.name}")
                    results[f"act_{layer.name}_hist"] = _hist_counts(val)
                except Exception:
                    pass

        return results

    def full_analysis(self, X_sample):
        """同时分析权重和激活值。"""
        # Analyzing weight distributions...
        w_stats = self._get_layer_weights_stats()
        # Analyzing activation distributions...
        a_stats = self.analyze_activations(X_sample)
        return {"weights": w_stats, "activations": a_stats}


# =============================================================================
# 3. 逐层梯度影响力分析 (增强版: 分阶段 + 稳定性检测)
# =============================================================================

class GradientAnalyzer:
    """
    分阶段梯度分析器，重点检测训练不稳定。

    每次快照对多个随机 minibatch 分别计算梯度，统计:
      - 各参数梯度 L2 范数的 均值/标准差 (反映梯度噪声)
      - 层间梯度范数比 (max/min, 过大 → 不稳定)
      - 梯度变异系数 CV = std/mean (CV > 1.0 → 高度噪声)
      - 梯度消失/爆炸检测 (||grad|| < 1e-8 或 > 1e8)
      - 与上一快照的参数变化量

    所有数据按阶段分组保存，便于对比不同阶段的梯度行为。
    """

    def __init__(self, model):
        self.model = model
        self.prev_weights = None
        self.prev_weights_stage = {}  # stage → weights snapshot
        # 累积记录
        self.gradient_history = []     # [{epoch, stage, round, grad_stats, ...}, ...]
        self.stage_summaries = []      # 每个阶段的汇总

    def _to_var_name(self, var):
        return var.name.replace(':', '_').replace('/', '_')

    def _compute_per_layer_grads_multi_sample(self, loss_fn, n_samples=5, n_points=1024):
        """
        通过执行单步训练并测量 weight delta 来估计各参数的有效梯度。

        原理：不直接使用 GradientTape（避免 eager/graph 模式混合导致 NaN），
        而是保存权重 → 跑一步完整训练（fit内部用 tf.function 正确计算梯度）
        → 测量权重变化 → 恢复权重。
        delta_W ≈ -lr × grad，所以 ||delta_W|| ∝ ||grad||。
        """
        original_X_f = self.model.domain.X_f
        original_X_f_in = self.model.X_f_in
        X_f_all = original_X_f
        n_total = len(X_f_all)

        # 收集每次采样的结果
        all_sample_deltas = []

        for si in range(n_samples):
            n_use = min(n_points, n_total)
            idx = np.random.choice(n_total, n_use, replace=False)
            X_sample = X_f_all[idx]

            # 保存权重
            saved_weights = [v.numpy().copy()
                             for v in self.model.u_model.trainable_variables]

            # 临时替换配点
            self.model.domain.X_f = X_sample
            self.model.X_f_in = [
                tf.cast(np.reshape(vec, (-1, 1)), tf.float32)
                for i, vec in enumerate(X_sample.T)
            ]

            # 跑一步训练（内部用 tf.function 正确计算并应用梯度）
            self.model.fit(tf_iter=1, newton_iter=0, batch_sz=n_use)

            # 测量 weight delta
            sample_result = {}
            for var, saved in zip(self.model.u_model.trainable_variables,
                                  saved_weights):
                name = self._to_var_name(var)
                current = var.numpy()
                delta_L2 = float(np.sqrt(np.sum((current - saved) ** 2)))
                param_L2 = float(np.sqrt(np.sum(saved ** 2)))
                sample_result[name] = {
                    "delta_L2": delta_L2,     # ||W_new - W_old|| ∝ ||grad||
                    "param_L2": param_L2,
                }

            all_sample_deltas.append(sample_result)

            # 恢复权重
            for var, saved in zip(self.model.u_model.trainable_variables,
                                  saved_weights):
                var.assign(saved)

        # 恢复原始配点
        self.model.domain.X_f = original_X_f
        self.model.X_f_in = original_X_f_in

        # 汇总统计：用 delta_L2 作为 grad_L2 的代理
        var_names = list(all_sample_deltas[0].keys())
        results = {}
        for name in var_names:
            deltas = [s[name]["delta_L2"] for s in all_sample_deltas]
            p_l2   = all_sample_deltas[0][name]["param_L2"]
            d_mean = float(np.mean(deltas))
            d_std  = float(np.std(deltas))
            cv     = d_std / (d_mean + 1e-20)
            results[name] = {
                "grad_L2_mean":   d_mean,      # 实际是 ||ΔW||_2，与 ||grad|| 成正比
                "grad_L2_std":    d_std,
                "grad_L2_min":    float(np.min(deltas)),
                "grad_L2_max":    float(np.max(deltas)),
                "grad_L2_samples": [float(v) for v in deltas],
                "CV":             float(cv),
                "is_noisy":       bool(cv > 1.0),
                "is_vanishing":   bool(d_mean < 1e-12),
                "is_exploding":   bool(d_mean > 1e3),
                "param_L2":       float(p_l2),
                "eff_update_mean": float(d_mean / (p_l2 + 1e-16)),
                "param_count":    self._get_param_count(name),
            }

        return results

    def _get_param_count(self, name):
        """根据名称查找参数的元素数量。"""
        for v in self.model.u_model.trainable_variables:
            if self._to_var_name(v) == name:
                return int(tf.reduce_sum(v * 0 + 1).numpy())
        return 0

    def _compute_layer_grad_ratio(self, grad_stats):
        """
        计算层间梯度范数的不平衡度。
        max/min 比值过大 (>1000) 说明不同层梯度严重不匹配 → 不稳定。
        """
        g_norms = [info["grad_L2_mean"] for info in grad_stats.values()
                   if info["grad_L2_mean"] > 1e-20]
        if len(g_norms) < 2:
            return {"max_min_ratio": 0, "max_layer": "", "min_layer": "", "is_unstable": False}

        max_val = max(g_norms)
        min_val = min(g_norms)
        ratio = max_val / (min_val + 1e-20)

        max_name = max(grad_stats.items(), key=lambda x: x[1]["grad_L2_mean"])[0]
        min_name = min(grad_stats.items(),
                       key=lambda x: x[1]["grad_L2_mean"]
                       if x[1]["grad_L2_mean"] > 1e-20 else float('inf'))[0]

        return {
            "max_min_ratio": float(ratio),
            "max_layer": max_name,
            "max_grad": float(max_val),
            "min_layer": min_name,
            "min_grad": float(min_val),
            "is_unstable": bool(ratio > 1000),  # 层间梯度比 > 1000 → 不稳定
        }

    def _compute_param_change(self):
        """计算与上次快照的参数变化量（含相对变化）。"""
        current = {}
        for var in self.model.u_model.trainable_variables:
            current[var.name] = _to_numpy(var)

        if self.prev_weights is None:
            self.prev_weights = current
            return {}

        changes = {}
        for var in self.model.u_model.trainable_variables:
            name = var.name
            if name in self.prev_weights:
                prev_val = self.prev_weights[name]
                cur_val  = current[name]
                delta = cur_val - prev_val
                delta_L2 = float(np.sqrt(np.sum(delta ** 2)))
                prev_L2  = float(np.sqrt(np.sum(prev_val ** 2)))
                rel_change = delta_L2 / (prev_L2 + 1e-16)
                changes[self._to_var_name(var)] = {
                    "abs_delta_L2":  delta_L2,
                    "rel_delta":     float(rel_change),
                    "prev_L2":       prev_L2,
                    "current_L2":    float(np.sqrt(np.sum(cur_val ** 2))),
                }

        self.prev_weights = current
        return changes

    def snapshot(self, n_samples=5, stage_label="", include_change=True):
        """
        拍一张详细的梯度快照。

        Args:
            n_samples: 采样次数 (越多越准确)
            stage_label: 阶段标签 e.g. "Stage1_Round2"
            include_change: 是否计算参数变化

        Returns:
            dict: 包含多采样权重变化统计 + 稳定性指标 + 参数变化
        """
        # 多采样梯度（通过 weight delta 估计）
        grad_stats = self._compute_per_layer_grads_multi_sample(
            None, n_samples=n_samples
        )

        # 层间梯度比
        layer_ratio = self._compute_layer_grad_ratio(grad_stats)

        # 汇总统计
        cv_values = [info["CV"] for info in grad_stats.values()]
        noisy_count = sum(1 for info in grad_stats.values() if info["is_noisy"])
        vanish_count = sum(1 for info in grad_stats.values() if info["is_vanishing"])
        explode_count = sum(1 for info in grad_stats.values() if info["is_exploding"])

        result = {
            "stage_label": stage_label,
            "gradients": grad_stats,
            "layer_grad_balance": layer_ratio,
            "summary": {
                "n_params_total": len(grad_stats),
                "n_noisy": noisy_count,
                "n_vanishing": vanish_count,
                "n_exploding": explode_count,
                "mean_CV": float(np.mean(cv_values)),
                "max_CV": float(np.max(cv_values)),
                "grad_instability_score": float(
                    (noisy_count / max(len(grad_stats), 1)) * 0.4 +
                    (1.0 if layer_ratio["is_unstable"] else 0.0) * 0.3 +
                    (1.0 if explode_count > 0 else 0.0) * 0.2 +
                    (1.0 if vanish_count > 0 else 0.0) * 0.1
                ),  # 0~1, >0.5 表示训练可能不稳定
            },
        }

        if include_change:
            result["param_change"] = self._compute_param_change()

        # 存档
        self.gradient_history.append(result)
        return result

    def stage_boundary_snapshot(self, stage_idx, round_idx, global_epoch):
        """
        在阶段/轮次边界拍快照，记录阶段信息。
        """
        label = f"S{stage_idx+1}_R{round_idx+1}_E{global_epoch}"
        result = self.snapshot(n_samples=1, stage_label=label)
        result["stage_idx"] = stage_idx
        result["round_idx"] = round_idx
        result["global_epoch"] = global_epoch

        # 保存阶段初始权重
        current = {}
        for var in self.model.u_model.trainable_variables:
            current[var.name] = _to_numpy(var)
        self.prev_weights_stage[label] = current

        return result

    def finalize(self):
        """
        汇总所有梯度历史，生成:
          - 各阶段梯度不稳定性得分变化
          - 梯度消失/爆炸出现时间线
          - 层间梯度比变化趋势
        """
        if not self.gradient_history:
            return {}

        summary = {
            "n_snapshots": len(self.gradient_history),
            "instability_timeline": [],
            "grad_ratio_timeline": [],
            "cv_timeline": [],
            "vanishing_events": [],
            "exploding_events": [],
            "noisy_params_timeline": [],
        }

        for i, snap in enumerate(self.gradient_history):
            s = snap.get("summary", {})
            label = snap.get("stage_label", f"snap_{i}")
            summary["instability_timeline"].append({
                "label": label,
                "score": s.get("grad_instability_score", 0),
            })
            ratio = snap.get("layer_grad_balance", {})
            summary["grad_ratio_timeline"].append({
                "label": label,
                "max_min_ratio": ratio.get("max_min_ratio", 0),
                "is_unstable": ratio.get("is_unstable", False),
            })
            summary["cv_timeline"].append({
                "label": label,
                "mean_CV": s.get("mean_CV", 0),
                "max_CV": s.get("max_CV", 0),
            })
            summary["noisy_params_timeline"].append({
                "label": label,
                "n_noisy": s.get("n_noisy", 0),
                "n_total": s.get("n_params_total", 0),
            })

            if s.get("n_exploding", 0) > 0:
                summary["exploding_events"].append({
                    "label": label,
                    "count": s["n_exploding"],
                })
            if s.get("n_vanishing", 0) > 0:
                summary["vanishing_events"].append({
                    "label": label,
                    "count": s["n_vanishing"],
                })

        # 检测不稳定趋势
        instability_scores = [e["score"] for e in summary["instability_timeline"]]
        summary["overall_verdict"] = {
            "mean_instability": float(np.mean(instability_scores)) if instability_scores else 0,
            "max_instability": float(np.max(instability_scores)) if instability_scores else 0,
            "is_training_unstable": bool(
                np.mean(instability_scores) > 0.3 if instability_scores else False
            ),
            "has_explosion": len(summary["exploding_events"]) > 0,
            "has_vanishing": len(summary["vanishing_events"]) > 0,
            "recommendation": (
                "训练不稳定! 梯度有爆炸/消失" if (len(summary["exploding_events"]) > 0 or len(summary["vanishing_events"]) > 0)
                else "梯度噪声较大, 考虑减小 LR 或增大 batch_size" if (np.mean(instability_scores) > 0.5 if instability_scores else False)
                else "训练相对稳定" if (np.mean(instability_scores) < 0.2 if instability_scores else True)
                else "训练中等稳定"
            ),
        }

        return summary


# =============================================================================
# 4. 训练中诊断编排器
# =============================================================================

class TrainingDiagnostics:
    """
    训练过程中定期拍快照，收集:
      - PDE 逐项统计
      - 输出头 (P/γ) 分布
      - **分阶段梯度稳定性分析** (多采样, CV, 层间比, 消失/爆炸检测)
      - 参数变化量
      - 损失分量值

    Usage:
        diag = TrainingDiagnostics(model, params, cfg, output_dir)
        # 训练循环中:
        for epoch in range(N):
            if epoch % interval == 0:
                diag.snapshot(epoch)
        diag.stage_boundary_snapshot(stage, round, epoch)  # 阶段边界
        diag.finalize()  # 汇总生成报告
    """

    def __init__(self, model, params, cfg, output_dir):
        self.model  = model
        self.params = params
        self.cfg    = cfg
        out_dir = os.path.join(output_dir, 'diagnostics')
        ensure_dir(out_dir)
        self.out_dir = out_dir

        # 子分析器
        from reynold_pinn import create_H_func
        H_func, _ = create_H_func(self.params, self.cfg)
        self.pde_analyzer    = ReynoldsTermAnalyzer(H_func, self.params, self.cfg)
        self.layer_collector = LayerStatsCollector(model.u_model)
        self.grad_analyzer   = GradientAnalyzer(model)

        self.snapshots = []
        self.epochs    = []
        self.stage_boundaries = []  # 阶段边界的详细梯度快照

    def snapshot(self, epoch):
        """训练中间快照：仅写文件，不输出到控制台。"""
        snap = {"epoch": epoch}
        X_f = self.model.domain.X_f

        snap["pde_terms"] = self.pde_analyzer.analyze(self.model.u_model, X_f)

        R_tf     = tf.constant(X_f[:, 0:1], dtype=tf.float32)
        theta_tf = tf.constant(X_f[:, 1:2], dtype=tf.float32)
        u_preds  = self.model.u_model(tf.concat([R_tf, theta_tf], 1))
        p_val    = _to_numpy(u_preds[0])
        g_val    = _to_numpy(u_preds[1])

        snap["output_p"]  = _stats(p_val, name="P_output")
        snap["output_g"]  = _stats(g_val, name="gamma_output")
        snap["output_p_hist"] = _hist_counts(p_val)
        snap["output_g_hist"] = _hist_counts(g_val)

        fb_val = p_val + g_val - np.sqrt(p_val**2 + g_val**2)
        snap["fb_complement"] = _stats(fb_val, name="FB(p,g)")
        snap["fb_complement_hist"] = _hist_counts(fb_val)

        snap["grad_analysis"] = {"note": "gradient analysis skipped during training, run diagnose_post.py for offline analysis"}

        loss_all = self.model.update_loss_seperate()
        loss_names = ['L_Reynolds', 'L_FB', 'L_BC_gamma']
        for i, loss_val in enumerate(loss_all):
            name = loss_names[i] if i < len(loss_names) else f"loss_{i}"
            snap[f"loss_{name}"] = float(_to_numpy(loss_val))

        self.snapshots.append(snap)
        self.epochs.append(epoch)

        snap_path = os.path.join(self.out_dir, f"snapshot_epoch_{epoch:06d}.json")
        self._save_snapshot_json(snap, snap_path)

    def _save_snapshot_json(self, snap, path):
        """保存快照为 JSON (hist 和大型数组写入单独的 .npz)。"""
        json_snap = {}
        npz_data = {}
        for key, val in snap.items():
            if isinstance(val, dict) and any(k.endswith('_hist') for k in val.keys()):
                json_snap[key] = {}
                for k, v in val.items():
                    if k.endswith('_hist') or k.endswith('_raw'):
                        npz_data[f"{key}/{k}"] = np.array(v) if not isinstance(v, np.ndarray) else v
                    elif isinstance(v, (list, dict)):
                        json_snap[key][k] = v
                    else:
                        json_snap[key][k] = v
            elif isinstance(val, dict):
                json_snap[key] = {}
                for k, v in val.items():
                    if isinstance(v, dict):
                        json_snap[key][k] = {}
                        for kk, vv in v.items():
                            if isinstance(vv, np.ndarray):
                                npz_data[f"{key}/{k}/{kk}"] = vv
                            else:
                                json_snap[key][k][kk] = vv
                    elif isinstance(v, np.ndarray):
                        npz_data[f"{key}/{k}"] = v
                    else:
                        json_snap[key][k] = v
            elif isinstance(val, np.ndarray):
                npz_data[key] = val
            else:
                json_snap[key] = val

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(json_snap, f, indent=2, default=str)

        if npz_data:
            npz_path = path.replace('.json', '.npz')
            np.savez_compressed(npz_path, **npz_data)

    def stage_boundary_snapshot(self, stage_idx, round_idx, global_epoch):
        """阶段/轮次边界的轻量快照，仅写文件不输出控制台。"""
        label = f"S{stage_idx+1}_R{round_idx+1}_E{global_epoch}"
        snap = {
            "stage_label": label,
            "stage_idx": stage_idx,
            "round_idx": round_idx,
            "global_epoch": global_epoch,
        }

        X_f = self.model.domain.X_f

        snap["pde_terms"] = self.pde_analyzer.analyze(self.model.u_model, X_f)

        R_tf = tf.constant(X_f[:, 0:1], dtype=tf.float32)
        theta_tf = tf.constant(X_f[:, 1:2], dtype=tf.float32)
        u_preds = self.model.u_model(tf.concat([R_tf, theta_tf], 1))
        p_val = _to_numpy(u_preds[0])
        g_val = _to_numpy(u_preds[1])

        snap["output_p"] = _stats(p_val, name="P_output")
        snap["output_g"] = _stats(g_val, name="gamma_output")
        fb_val = p_val + g_val - np.sqrt(p_val**2 + g_val**2)
        snap["fb_complement"] = _stats(fb_val, name="FB")

        loss_all = self.model.update_loss_seperate()
        loss_names = ['L_Reynolds', 'L_FB', 'L_BC_gamma']
        for i, loss_val in enumerate(loss_all):
            name = loss_names[i] if i < len(loss_names) else f"loss_{i}"
            snap[f"loss_{name}"] = float(_to_numpy(loss_val))

        self.stage_boundaries.append(snap)

        snap_path = os.path.join(self.out_dir, f"stage_boundary_{label}.json")
        self._save_snapshot_json(snap, snap_path)

    def finalize(self):
        """训练结束，汇总所有快照 + 梯度稳定性分析，仅写文件。"""
        summary = {
            "epochs": self.epochs,
            "loss_Reynolds": [],
            "loss_FB": [],
            "output_p_mean": [],
            "output_g_mean": [],
            "fb_violation_mean": [],
            "pde_part_1_mean": [],
            "pde_part_2_mean": [],
            "pde_part_3_1_mean": [],
            "pde_part_3_2_mean": [],
            "pde_stab_mean": [],
            "pde_f_p_mean": [],
        }

        for snap in self.snapshots:
            summary["loss_Reynolds"].append(snap.get("loss_L_Reynolds", 0))
            summary["loss_FB"].append(snap.get("loss_L_FB", 0))
            summary["output_p_mean"].append(snap.get("output_p", {}).get("mean", 0))
            summary["output_g_mean"].append(snap.get("output_g", {}).get("mean", 0))
            summary["fb_violation_mean"].append(snap.get("fb_complement", {}).get("mean", 0))
            pde = snap.get("pde_terms", {})
            summary["pde_part_1_mean"].append(pde.get("part_1", {}).get("mean", 0))
            summary["pde_part_2_mean"].append(pde.get("part_2", {}).get("mean", 0))
            summary["pde_part_3_1_mean"].append(pde.get("part_3_1", {}).get("mean", 0))
            summary["pde_part_3_2_mean"].append(pde.get("part_3_2", {}).get("mean", 0))
            summary["pde_stab_mean"].append(pde.get("stab", {}).get("mean", 0))
            summary["pde_f_p_mean"].append(pde.get("f_p", {}).get("mean", 0))

        # 汇总梯度稳定性
        grad_summary = self.grad_analyzer.finalize()
        summary["gradient_stability"] = grad_summary

        summary_path = os.path.join(self.out_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # 生成可读文本报告
        report_path = os.path.join(self.out_dir, "diagnostic_report.txt")
        self._write_report(report_path, grad_summary)

        return summary

    def _write_report(self, path, grad_summary=None):
        """生成人类可读的文本报告（含梯度稳定性部分）。"""
        lines = []
        lines.append("=" * 70)
        lines.append("  PINN Training Diagnostic Report")
        lines.append("=" * 70)

        if not self.snapshots:
            lines.append("\n  (No snapshots recorded)\n")
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return

        first = self.snapshots[0]
        last  = self.snapshots[-1]

        # --- 1. Reynolds PDE 逐项对比 ---
        lines.append("\n" + "-" * 70)
        lines.append("  1. Reynolds PDE Term Magnitudes (first vs last snapshot)")
        lines.append("-" * 70)
        lines.append(f"  {'Term':<12s} {'Mean(init)':>14s} {'Mean(final)':>14s} "
                     f"{'Ratio':>10s} {'P99(init)':>14s} {'P99(final)':>14s}")
        lines.append("  " + "-" * 66)

        for term_name in ["part_1", "part_2", "part_3_1", "part_3_2", "stab", "f_p"]:
            first_t = first.get("pde_terms", {}).get(term_name, {})
            last_t  = last.get("pde_terms", {}).get(term_name, {})
            mean_i = first_t.get("mean", 0)
            mean_f = last_t.get("mean", 0)
            ratio  = abs(mean_f / (mean_i + 1e-16))
            p99_i  = first_t.get("p99", 0)
            p99_f  = last_t.get("p99", 0)
            lines.append(f"  {term_name:<12s} {mean_i:>14.4e} {mean_f:>14.4e} "
                         f"{ratio:>10.2f} {p99_i:>14.4e} {p99_f:>14.4e}")

        # --- 2. 损失对比 ---
        lines.append("\n" + "-" * 70)
        lines.append("  2. Loss Evolution")
        lines.append("-" * 70)
        for key in ["loss_L_Reynolds", "loss_L_FB"]:
            first_val = first.get(key, 0)
            last_val  = last.get(key, 0)
            ratio = abs(last_val / (first_val + 1e-16))
            lines.append(f"  {key:<25s}: {first_val:>12.4e} → {last_val:>12.4e}  "
                         f"(ratio: {ratio:.4f})")

        # --- 3. 输出分布对比 ---
        lines.append("\n" + "-" * 70)
        lines.append("  3. Output Distribution (first vs last)")
        lines.append("-" * 70)
        for out_name in ["output_p", "output_g", "fb_complement"]:
            first_o = first.get(out_name, {})
            last_o  = last.get(out_name, {})
            lines.append(f"  [{out_name}]")
            lines.append(f"    mean:  {first_o.get('mean',0):>12.4e} → {last_o.get('mean',0):>12.4e}")
            lines.append(f"    std:   {first_o.get('std',0):>12.4e} → {last_o.get('std',0):>12.4e}")
            lines.append(f"    max:   {first_o.get('max',0):>12.4e} → {last_o.get('max',0):>12.4e}")

        # --- 4. 梯度稳定性分析 ★ ---
        lines.append("\n" + "=" * 70)
        lines.append("  4. GRADIENT STABILITY ANALYSIS (Per-Stage)")
        lines.append("=" * 70)

        if grad_summary:
            verdict = grad_summary.get("overall_verdict", {})
            lines.append(f"\n  Overall Verdict: {verdict.get('recommendation', 'N/A')}")
            lines.append(f"  Mean instability score: {verdict.get('mean_instability', 0):.4f}")
            lines.append(f"  Max instability score:  {verdict.get('max_instability', 0):.4f}")
            lines.append(f"  Gradient explosion detected: {verdict.get('has_explosion', False)}")
            lines.append(f"  Gradient vanishing detected: {verdict.get('has_vanishing', False)}")

            # 不稳定性时间线
            timeline = grad_summary.get("instability_timeline", [])
            if timeline:
                lines.append(f"\n  Instability Score Timeline:")
                lines.append(f"  {'Stage/Round':<25s} {'Score':>8s} {'Verdict'}")
                lines.append("  " + "-" * 50)
                for entry in timeline:
                    score = entry["score"]
                    verdict_str = ("!! UNSTABLE" if score > 0.7 else
                                   "! NOISY" if score > 0.4 else
                                   "stable" if score < 0.15 else "moderate")
                    lines.append(f"  {entry['label']:<25s} {score:>8.4f}  {verdict_str}")

            # 层间梯度比时间线
            ratio_timeline = grad_summary.get("grad_ratio_timeline", [])
            if ratio_timeline:
                lines.append(f"\n  Layer Gradient Ratio Timeline (max/min):")
                lines.append(f"  {'Stage/Round':<25s} {'Ratio':>10s} {'Stable?'}")
                lines.append("  " + "-" * 50)
                for entry in ratio_timeline:
                    stable_str = "NO" if entry.get("is_unstable") else "yes"
                    lines.append(f"  {entry['label']:<25s} {entry['max_min_ratio']:>10.1f}  "
                                 f"{stable_str}")

            # 梯度噪声时间线
            cv_timeline = grad_summary.get("cv_timeline", [])
            if cv_timeline:
                lines.append(f"\n  Gradient Noise (CV) Timeline:")
                lines.append(f"  {'Stage/Round':<25s} {'mean_CV':>10s} {'max_CV':>10s}")
                lines.append("  " + "-" * 50)
                for entry in cv_timeline:
                    lines.append(f"  {entry['label']:<25s} {entry['mean_CV']:>10.4f} "
                                 f"{entry['max_CV']:>10.4f}")

            # 爆炸/消失事件
            if grad_summary.get("exploding_events"):
                lines.append(f"\n  !! GRADIENT EXPLOSION EVENTS:")
                for e in grad_summary["exploding_events"]:
                    lines.append(f"    {e['label']}: {e['count']} params exploded")

            if grad_summary.get("vanishing_events"):
                lines.append(f"\n  !! GRADIENT VANISHING EVENTS:")
                for e in grad_summary["vanishing_events"]:
                    lines.append(f"    {e['label']}: {e['count']} params vanished")

        # --- 5. 逐参数梯度详情（最后一个阶段边界快照） ---
        if self.stage_boundaries:
            lines.append("\n" + "=" * 70)
            lines.append("  5. Per-Parameter Gradient Details (last stage boundary)")
            lines.append("=" * 70)
            last_sb = self.stage_boundaries[-1]
            grad_data = last_sb.get("gradients", {})
            sorted_grads = sorted(grad_data.items(),
                                  key=lambda x: x[1].get("grad_L2_mean", 0),
                                  reverse=True)
            lines.append(f"\n  {'Param':<50s} {'Grad μ':>10s} {'Grad σ':>10s} "
                         f"{'CV':>8s} {'Eff.Upd':>10s} {'Noisy?':>7s}")
            lines.append("  " + "-" * 97)
            for name, info in sorted_grads[:30]:
                cv_flag = "!!" if info.get("is_noisy") else ""
                vanish_flag = " VANISH" if info.get("is_vanishing") else ""
                explode_flag = " EXPLODE" if info.get("is_exploding") else ""
                lines.append(f"  {name:<50s} {info.get('grad_L2_mean',0):>10.4e} "
                             f"{info.get('grad_L2_std',0):>10.4e} "
                             f"{info.get('CV',0):>8.4f} "
                             f"{info.get('eff_update_mean',0):>10.4e} "
                             f"{cv_flag}{vanish_flag}{explode_flag}")

        # --- 6. 参数变化 ---
        if self.stage_boundaries:
            lines.append("\n" + "-" * 70)
            lines.append("  6. Parameter Changes (last stage boundary)")
            lines.append("-" * 70)
            chg_data = last_sb.get("param_change", {})
            sorted_chg = sorted(chg_data.items(),
                                key=lambda x: x[1].get("rel_delta", 0),
                                reverse=True)
            lines.append(f"  {'Param':<50s} {'Abs Δ L2':>12s} {'Rel Δ':>10s}")
            lines.append("  " + "-" * 74)
            for name, info in sorted_chg[:20]:
                a_d = info.get("abs_delta_L2", 0)
                r_d = info.get("rel_delta", 0)
                lines.append(f"  {name:<50s} {a_d:>12.4e} {r_d:>10.6f}")

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


# =============================================================================
# 5. 训练后全量分析
# =============================================================================

def post_training_analysis(model, params, cfg, output_dir):
    """
    训练完成后进行全量分析:
      - 逐层权重分布
      - 逐层激活值分布 (在全部配点上)
      - 完整 PDE 逐项统计
      - 参数变化统计
    """
    diag_dir = os.path.join(output_dir, 'diagnostics')
    post_dir = os.path.join(diag_dir, 'post_training')
    ensure_dir(post_dir)

    # PDE 项完整统计
    from reynold_pinn import create_H_func
    H_func, _ = create_H_func(params, cfg)
    pde_analyzer = ReynoldsTermAnalyzer(H_func, params, cfg)
    pde_stats = pde_analyzer.analyze(model.u_model, model.domain.X_f)

    with open(os.path.join(post_dir, 'pde_terms_full.json'), 'w') as f:
        json.dump(pde_stats, f, indent=2, default=str)

    # 逐层统计
    layer_collector = LayerStatsCollector(model.u_model)
    # 抽样所有配点（最多 5000 点以节省内存）
    X_f = model.domain.X_f
    n_sample = min(len(X_f), 5000)
    idx = np.random.choice(len(X_f), n_sample, replace=False)
    X_sample = X_f[idx]
    layer_stats = layer_collector.full_analysis(X_sample)

    with open(os.path.join(post_dir, 'layer_stats.json'), 'w') as f:
        json.dump(layer_stats, f, indent=2, default=str)

    # 计算 FB 互补条件在全部数据上的表现
    R_tf     = tf.constant(X_f[:, 0:1], dtype=tf.float32)
    theta_tf = tf.constant(X_f[:, 1:2], dtype=tf.float32)
    u_preds  = model.u_model(tf.concat([R_tf, theta_tf], 1))
    p_val    = _to_numpy(u_preds[0])
    g_val    = _to_numpy(u_preds[1])
    fb_val   = p_val + g_val - np.sqrt(p_val**2 + g_val**2)
    p_mul_g  = p_val * g_val

    fb_stats = _stats(fb_val, name="FB_total")
    pg_stats = _stats(p_mul_g, name="P_times_gamma")
    with open(os.path.join(post_dir, 'jfo_full.json'), 'w') as f:
        json.dump({"FB_complement": fb_stats, "P_times_gamma": pg_stats}, f, indent=2)

    return pde_stats, layer_stats, fb_stats


# =============================================================================
# 6. 梯度影响力逐组件详细分析 (Point 4)
# =============================================================================

def gradient_impact_detailed_analysis(model, params, cfg, output_dir, n_samples=3):
    """
    对模型每个可训练组件，多次独立计算梯度并统计，
    得到稳健的"影响力"估计。

    对每个组件 (Dense kernel, bias 等)：
      - 梯度 L2 范数 (多次平均)
      - 参数 L2 范数
      - 有效学习率 = lr × ||grad|| / ||param||   (近似)
      - 参数变化量 ||ΔW|| (需要前后权重)

    Args:
        n_samples: 对配点随机采样 n_samples 次取平均梯度
    """
    diag_dir = os.path.join(output_dir, 'diagnostics')
    impact_dir = os.path.join(diag_dir, 'gradient_impact')
    ensure_dir(impact_dir)

    X_f = model.domain.X_f

    all_results = []
    for sample_i in range(n_samples):
        # 随机采样
        n_use = min(len(X_f), 4096)
        idx = np.random.choice(len(X_f), n_use, replace=False)
        X_sample = X_f[idx]

        # 临时替换 domain.X_f 以计算梯度
        original_X_f = model.domain.X_f
        original_X_f_in = model.X_f_in
        model.domain.X_f = X_sample
        model.X_f_in = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32)
                        for i, vec in enumerate(X_sample.T)]

        with tf.GradientTape() as tape:
            loss_all = model.update_loss_seperate()
            loss_total = sum(loss_all)

        trainable_vars = model.u_model.trainable_variables
        grads = tape.gradient(loss_total, trainable_vars)

        # 恢复
        model.domain.X_f = original_X_f
        model.X_f_in = original_X_f_in

        result = {}
        for var, grad in zip(trainable_vars, grads):
            name = var.name.replace(':', '_')
            g_l2 = _grad_norm(grad)
            p_l2 = float(tf.sqrt(tf.reduce_sum(var ** 2)).numpy())
            # 有效更新尺度 (忽略 lr 因子)
            effective_update = g_l2 / (p_l2 + 1e-16) if p_l2 > 0 else g_l2
            result[name] = {
                "grad_L2": g_l2,
                "param_L2": p_l2,
                "eff_update_scale": float(effective_update),
                "param_count": int(tf.reduce_sum(var * 0 + 1).numpy()),
            }
        all_results.append(result)

    # 汇总多采样结果 (取平均)
    merged = {}
    for name in all_results[0].keys():
        g_l2s = [r[name]["grad_L2"] for r in all_results]
        p_l2s = [r[name]["param_L2"] for r in all_results]
        eus   = [r[name]["eff_update_scale"] for r in all_results]
        merged[name] = {
            "grad_L2_mean": float(np.mean(g_l2s)),
            "grad_L2_std":  float(np.std(g_l2s)),
            "param_L2":     float(np.mean(p_l2s)),
            "eff_update_mean": float(np.mean(eus)),
            "param_count":  all_results[0][name]["param_count"],
        }

    # 按影响力排序保存
    sorted_by_impact = sorted(merged.items(),
                              key=lambda x: x[1]["eff_update_mean"],
                              reverse=True)

    impact_path = os.path.join(impact_dir, "gradient_impact.json")
    with open(impact_path, 'w') as f:
        json.dump({"per_component": merged, "sorted_by_impact": [
            {"name": n, **info} for n, info in sorted_by_impact
        ]}, f, indent=2)

    # 全量结果已保存到文件

    return merged


# =============================================================================
# 7. 综合入口: 训练后一键分析
# =============================================================================

def run_full_diagnosis(model, params, cfg, output_dir):
    """训练后运行全部诊断。"""
    post_training_analysis(model, params, cfg, output_dir)
    gradient_impact_detailed_analysis(model, params, cfg, output_dir)
