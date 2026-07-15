"""
Reynolds方程PINN求解器 - 螺旋槽空化问题
基于TensorDiffEq框架，采用JFO空化模型

作者: [叶萌]
日期: 2026
"""
import os
import sys
import json

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 禁用 XLA JIT（RTX 4090 segfault 兼容性修复）
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

import numpy as np
import tensorflow as tf
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

# TensorDiffEq imports
from tensordiffeq.boundaries import dirichletBC
from tensordiffeq.domains import DomainND
from tensordiffeq.models import CollocationSolverND
from diagnose import TrainingDiagnostics, post_training_analysis, gradient_impact_detailed_analysis


# =============================================================================
# 0. 辅助工具
# =============================================================================
def ensure_dir(path):
    """创建目录（如不存在）。"""
    os.makedirs(path, exist_ok=True)


class Tee:
    """同时输出到 stdout 和日志文件。"""
    def __init__(self, log_path):
        self.stdout = sys.stdout
        self.log = open(log_path, 'w', encoding='utf-8', buffering=1)

    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)

    def flush(self):
        self.stdout.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# =============================================================================
# 1. 配置参数
# =============================================================================
class Config:
    """集中管理所有配置参数"""

    # 几何参数 (单位: m)
    r_i = 47.0e-3          # 内径
    r_o = 52.0e-3          # 外径
    h_i = 3.0e-6           # 平衡膜厚
    K = 6.0                # 周期数(360度分6份)

    # 螺旋槽几何
    R_d_1_ratio = 1.043    # 槽起始位置比例
    R_d_2_ratio = 1.106 * 2  # 槽结束位置比例(乘了2倍，完全贯通)
    alpha_deg = 3.0        # 螺旋角 (度)
    h_texture_ratio = 3.0  # 槽深/平衡膜厚比
    groove_ratio = 0.5     # 槽宽比

    # 工况参数
    p_i = 0.1e6            # 内径压力 (Pa)
    p_o_ratio = 1.5        # 外径/内径压力比
    eta = 8.00e-4          # 动力粘度 (Pa·s)
    omega_rpm = 6000       # 转速 (rpm)

    # 数值参数
    N_f = 8000             # 配点数
    N_groove_b = 100        # 槽边界采样点数
    N_groove_r = 50        # 槽径向边界采样点数
    domain_fidelity = 100   # 域网格密度

    # ========== 模型架构参数（可配置） ==========
    Act = "tanh"            # 激活函数: "tanh" 或 "silu"
    core = "MLP"            # 网络类型: "mlp" 或 "pikan" (KAN架构)
    use_residual = False    # 是否使用残差连接
    output_head_dim = 64    # 输出头隐藏层维度 (用于深度输出头)
    coslayer_mode = "mlp"  # 输入编码层: "simple" (原版线性混合) 或 "mlp" (R/θ各自MLP通路)

    # PIKAN 参数 (仅 core="pikan" 时生效)
    kan_grid_size = 5       # B-spline 网格区间数
    kan_spline_order = 3    # B-spline 多项式阶数
    pikan_layer_sizes = [2, 64, 64, 64, 64, 2]  # PIKAN 层大小

    # 训练参数
    layer_sizes = [2, 128, 128, 256 ,128, 128, 2]
    total_epochs = 30000      # 总训练 epoch 数
    warmup_epochs = 1000      # warmup epoch 数 (前 N 步 lr 从 0 线性增长)
    peak_lr = 1e-3            # warmup 达到的最高学习率
    min_lr = 1e-6             # 余弦衰减最低学习率
    batch_size = 2048         # minibatch大小; None=全批量, int=随机minibatch

    # ========== 学习率调度器 (可配置) ==========
    # 选项: "warmup_cosine" | "cosine" | "cyclic" | "one_cycle"
    #   warmup_cosine: warmup → cosine decay (推荐: 稳定的baseline)
    #   cosine:        纯 cosine decay, 无 warmup
    #   cyclic:        Cosine Annealing with Warm Restarts (推荐: 崎岖loss景观)
    #   one_cycle:     One-Cycle Policy, 先升后降 (推荐: 快速收敛)
    lr_schedule = "warmup_cosine"
    # Cyclic / One-Cycle 参数:
    lr_cycle_period = 5000       # cyclic: 每周期epoch数; one_cycle: 上升阶段epoch数
    lr_cycle_decay = 0.7         # cyclic: 每周期peak_lr衰减因子 (<1 则逐周期递减)
    lr_cycle_min_factor = 0.01   # cyclic: min_lr 相对于 peak_lr 的比例

    # ========== Loss 平衡 (可配置) ==========
    # 选项: "none" | "fixed" | "auto"
    #   none:  所有 loss 项等权 (默认, 当前行为)
    #   fixed: 对 FB 互补项施加固定权重 (推荐: 对 JFO 问题效果好)
    #   auto:  基于梯度量级的自适应权重 (GradNorm风格)
    loss_balance_mode = "none"
    fb_loss_weight = 1.0          # fixed 模式下的 FB 项权重 (推荐: 100~10000)
    loss_balance_alpha = 0.2      # auto 模式下的 EMA 平滑系数

    # ========== L-BFGS 精调 (可配置) ==========
    fine_tune_enabled = False     # Adam 训练结束后是否运行 L-BFGS 精调
    fine_tune_epochs = 1000       # L-BFGS 迭代步数 (推荐: 500~2000)
    fine_tune_eager = False       # True=eager模式 (有get_weights修复), False=graph模式 (推荐, 更稳健)

    # 诊断参数
    diag_enabled = True         # 是否启用训练诊断
    diag_interval = 1000         # 诊断快照间隔 (epoch 数); 0=仅阶段边界

    # 硬件 / 输出
    output_dir = "output_resF_tanh" #"output_resF_tanh_large"  # 输出目录
    device = "5"              # GPU设备ID, 例如 "0", "1", "5"

    # 绘图参数
    dpi_save = 600
    dpi_watch = 150
    text_size = 18
    font_size = 20


# =============================================================================
# 2. 物理参数计算
# =============================================================================
def compute_physical_params(cfg: Config):
    """计算无量纲化参数"""
    r_base = cfg.r_o
    p_c = 0.0                 #空化压力
    p_o = cfg.p_o_ratio * cfg.p_i + p_c
    p_base = 10 * p_o
    omega = cfg.omega_rpm * 2 * np.pi / 60
    
    # 无量纲参数
    Lambda = (6 * cfg.eta * omega * r_base**2) / (cfg.h_i**2 * p_base)
    P_i = cfg.p_i / p_base
    P_o = p_o / p_base
    R_lim = [cfg.r_i / r_base, cfg.r_o / r_base]
    theta_lim = [0.0, 2 * np.pi / cfg.K]
    #theta_lim = [np.pi/6, np.pi/6 + 2 * np.pi / cfg.K]
    
    # 螺旋槽参数
    R_d_1 = cfg.R_d_1_ratio * cfg.r_i / r_base
    R_d_2 = cfg.R_d_2_ratio * cfg.r_i / r_base
    alpha = cfg.alpha_deg / 180 * np.pi
    h_texture = cfg.h_texture_ratio * cfg.h_i
    
    return {
        'Lambda': tf.constant(Lambda, dtype=tf.float32),
        'P_i': P_i, 'P_o': P_o,
        'R_lim': R_lim, 'theta_lim': theta_lim,
        'R_d_1': tf.constant(R_d_1, dtype=tf.float32),
        'R_d_2': tf.constant(R_d_2, dtype=tf.float32),
        'r_g': tf.constant(R_d_1, dtype=tf.float32),
        'alpha': tf.constant(alpha, dtype=tf.float32),
        'h_texture': tf.constant(h_texture, dtype=tf.float32),
        'h_i': cfg.h_i,
        'K': tf.constant(cfg.K, dtype=tf.float32),
        'groove_ratio': cfg.groove_ratio,
    }


# =============================================================================
# 3. 膜厚函数
# =============================================================================
def create_H_func(params, cfg: Config):
    """创建膜厚函数 H(R, theta)"""
    R_d_1 = params['R_d_1']
    R_d_2 = params['R_d_2']
    r_g = params['r_g']
    alpha = params['alpha']
    h_texture = params['h_texture']
    h_i = params['h_i']
    groove_ratio = params['groove_ratio']
    #K = params['K']
    K_val = float(params['K'].numpy())
    
    # 平滑参数
    N_xi = 50.0
    R_xi = 100.0
    xi_R = (params['R_lim'][1] - params['R_lim'][0]) / R_xi
    xi_theta = (params['theta_lim'][1] - params['theta_lim'][0]) / N_xi
    theta_offset = np.pi / 6  # 30度相位偏移

    def theta_sym(R):
        """螺旋线方程"""
        return tf.math.log(R / r_g) / tf.math.tan(alpha)+theta_offset

    def H_func(R, theta):
        """膜厚分布函数，使用sigmoid平滑"""
        # 螺旋线方程（内联以避免 tf.function autograph 闭包穿透问题）
        theta_spiral = tf.math.log(R / r_g) / tf.math.tan(alpha) + theta_offset

        # 螺旋槽区域判断（使用sigmoid平滑）
        periodic_offsets = [0, -2*np.pi/K_val, -4*np.pi/K_val, 2*np.pi/K_val, 4*np.pi/K_val]
        periodic_terms = []
        for offset in periodic_offsets:
            term = (tf.math.sigmoid((theta - theta_spiral + offset) / xi_theta) *
                   tf.math.sigmoid((theta_spiral - theta + 2*np.pi/K_val*groove_ratio - offset) / xi_theta))
            periodic_terms.append(term)
        
        is_texture = (tf.math.sigmoid((R - R_d_1) / xi_R) * 
                     tf.math.sigmoid((R_d_2 - R) / xi_R) * 
                     sum(periodic_terms))
        
        H = 1.0 * (1 - is_texture) + (1.0 + h_texture / h_i) * is_texture
        return H
    
    return H_func, theta_sym


# =============================================================================
# 4. PDE残差模型
# =============================================================================
def create_pde_models(H_func, params):
    """创建PDE残差函数"""
    Lambda = params['Lambda']
    
    def f_model_FBNS(u_model, R, theta):
        """Reynolds方程残差 + JFO稳定项"""
        p_vector = u_model(tf.concat([R, theta], 1))
        p, gamma = p_vector[0], p_vector[1]         #p是压力,gamma是空化率
        H = H_func(R, theta)
        
        # 压力梯度
        p_R = tf.gradients(p, R)[0]
        p_theta = tf.gradients(p, theta)[0]
        
        # Reynolds方程各项
        part_1 = tf.gradients(R * H**3 * p_R, R)[0] / R
        part_2 = tf.gradients(H**3 * p_theta, theta)[0] / R**2
        part_3_1 = -Lambda * tf.gradients(H, theta)[0]
        part_3_2 = -Lambda * tf.gradients(-gamma * H, theta)[0]
        
        # 稳定项（处理空化边界）
        div_gamma = tf.gradients(gamma, theta)[0]
        div_2_gamma = tf.gradients(div_gamma, theta)[0]
        div_p = tf.gradients(p, theta)[0]
        
        epsilon = 0.1
        tau = tf.stop_gradient((tf.math.abs(div_gamma) - div_gamma) * epsilon)
        tau_2 = tf.stop_gradient((div_p - tf.math.abs(div_p)) * epsilon)

        f_p = part_1 + part_2 + part_3_1 + part_3_2 + div_2_gamma * tau * tau_2
        #f_p = part_1 + part_2 + part_3_1 + part_3_2
        return f_p
    
    def f_model_FB(u_model, R, theta):
        """Fischer-Burmeister互补条件"""
        p_vector = u_model(tf.concat([R, theta], 1))
        p, gamma = p_vector[0], p_vector[1]
        return p + gamma - tf.math.sqrt(p**2 + gamma**2)
    
    return f_model_FBNS, f_model_FB


# =============================================================================
# 5. 螺旋槽边界配点生成
# =============================================================================
def generate_groove_points(theta_sym, params, cfg: Config):
    """在螺旋槽边界生成额外配点"""
    R_lim = params['R_lim']
    theta_lim = params['theta_lim']
    R_d_1 = float(params['R_d_1'].numpy())
    K = float(params['K'].numpy())
    groove_ratio = params['groove_ratio']
    
    # 槽边界线配点
    R_list = np.linspace(R_d_1, R_lim[1], cfg.N_groove_b)
    theta_1 = np.array(theta_sym(R_list))
    theta_2 = theta_1 + 2 * np.pi / K * groove_ratio
    
    R_all = np.concatenate([R_list, R_list])
    theta_all = np.concatenate([theta_1, theta_2])
    
    # 周期性扩展
    R_final, theta_final = [], []
    for offset in [0, -2*np.pi/K, -4*np.pi/K, 2*np.pi/K, 4*np.pi/K]:
        theta_shifted = theta_all + offset
        mask = (theta_shifted > theta_lim[0]) & (theta_shifted < theta_lim[1])
        theta_final.extend(theta_shifted[mask])
        R_final.extend(R_all[mask])
    
    # 径向边界配点
    theta_radial = np.linspace(float(theta_sym(R_d_1)), 
                               float(theta_sym(R_d_1)) + 2*np.pi/K*groove_ratio, 
                               cfg.N_groove_r)
    for offset in [0, -2*np.pi/K, -4*np.pi/K, 2*np.pi/K, 4*np.pi/K]:
        theta_shifted = theta_radial + offset
        mask = (theta_shifted > theta_lim[0]) & (theta_shifted < theta_lim[1])
        filtered_theta = theta_shifted[mask]
        theta_final.extend(filtered_theta)
        R_final.extend([R_d_1] * len(filtered_theta))        
    
    return np.array(R_final).reshape(-1, 1), np.array(theta_final).reshape(-1, 1)


# =============================================================================
# 6. 学习率调度器
# =============================================================================
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Warmup + 余弦衰减学习率调度器。

    lr:  0 ──线性增长──→ peak_lr ──余弦衰减──→ min_lr
         |← warmup_epochs →|←───  decay zone  ───→|
    """

    def __init__(self, peak_lr, warmup_epochs, total_epochs, min_lr=1e-6):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        w = tf.cast(self.warmup_epochs, tf.float32)
        t = tf.cast(self.total_epochs, tf.float32)

        warmup_lr = self.peak_lr * (step / tf.maximum(w, 1.0))

        decay_steps = t - w
        cosine = 0.5 * (1.0 + tf.cos(
            tf.constant(np.pi, dtype=tf.float32) * (step - w) / tf.maximum(decay_steps, 1.0)
        ))
        decay_lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine

        return tf.where(step < w, warmup_lr, decay_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
            "min_lr": self.min_lr,
        }


# ── 额外的 LR 调度器 ──────────────────────────────────────────────────────────
class CosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """纯 Cosine 衰减 (无 warmup): peak_lr → min_lr"""

    def __init__(self, peak_lr, total_epochs, min_lr=1e-6):
        super().__init__()
        self.peak_lr = peak_lr
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        t = tf.cast(self.total_epochs, tf.float32)
        cosine = 0.5 * (1.0 + tf.cos(
            tf.constant(np.pi, dtype=tf.float32) * step / tf.maximum(t, 1.0)
        ))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def get_config(self):
        return {"peak_lr": self.peak_lr, "total_epochs": self.total_epochs,
                "min_lr": self.min_lr}


class CyclicCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine Annealing with Warm Restarts.
    每个周期: peak_lr → min_lr (cosine 衰减), 然后 restart 到新的 peak_lr。
    每周期 peak_lr 乘以 cycle_decay 逐渐降低。
    """

    def __init__(self, peak_lr, cycle_period, min_lr_factor=0.01,
                 cycle_decay=0.7):
        super().__init__()
        self.peak_lr = peak_lr
        self.cycle_period = cycle_period
        self.min_lr_factor = min_lr_factor
        self.cycle_decay = cycle_decay

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        period = tf.cast(self.cycle_period, tf.float32)
        # 当前在第几个周期
        cycle = tf.math.floordiv(step, period)
        step_in_cycle = step - cycle * period
        # 当前周期的 peak_lr
        current_peak = self.peak_lr * (self.cycle_decay ** tf.cast(cycle, tf.float32))
        min_lr = current_peak * self.min_lr_factor
        cosine = 0.5 * (1.0 + tf.cos(
            tf.constant(np.pi, dtype=tf.float32) * step_in_cycle / tf.maximum(period, 1.0)
        ))
        return min_lr + (current_peak - min_lr) * cosine

    def get_config(self):
        return {"peak_lr": self.peak_lr, "cycle_period": self.cycle_period,
                "min_lr_factor": self.min_lr_factor, "cycle_decay": self.cycle_decay}


class OneCycleDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """One-Cycle Policy: warmup 阶段 LR 线性升到 peak_lr, 然后 cosine 降到 min_lr。
    warmup_prop: 上升阶段占总步数的比例 (典型值 0.3)。
    """

    def __init__(self, peak_lr, total_epochs, min_lr=1e-6, warmup_prop=0.3):
        super().__init__()
        self.peak_lr = peak_lr
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_prop = warmup_prop

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        t = tf.cast(self.total_epochs, tf.float32)
        warmup_steps = t * self.warmup_prop
        warmup_lr = self.min_lr + (self.peak_lr - self.min_lr) * (step / tf.maximum(warmup_steps, 1.0))
        decay_steps = t - warmup_steps
        cosine = 0.5 * (1.0 + tf.cos(
            tf.constant(np.pi, dtype=tf.float32) * (step - warmup_steps) / tf.maximum(decay_steps, 1.0)
        ))
        decay_lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "total_epochs": self.total_epochs,
                "min_lr": self.min_lr, "warmup_prop": self.warmup_prop}


def _create_lr_schedule(cfg):
    """根据 Config 创建学习率调度器。"""
    name = cfg.lr_schedule
    if name == "warmup_cosine":
        return WarmupCosineDecay(
            peak_lr=cfg.peak_lr, warmup_epochs=cfg.warmup_epochs,
            total_epochs=cfg.total_epochs, min_lr=cfg.min_lr,
        )
    elif name == "cosine":
        return CosineDecay(
            peak_lr=cfg.peak_lr, total_epochs=cfg.total_epochs,
            min_lr=cfg.min_lr,
        )
    elif name == "cyclic":
        return CyclicCosineDecay(
            peak_lr=cfg.peak_lr, cycle_period=cfg.lr_cycle_period,
            min_lr_factor=cfg.lr_cycle_min_factor,
            cycle_decay=cfg.lr_cycle_decay,
        )
    elif name == "one_cycle":
        return OneCycleDecay(
            peak_lr=cfg.peak_lr, total_epochs=cfg.total_epochs,
            min_lr=cfg.min_lr,
        )
    else:
        raise ValueError(f"未知的 lr_schedule: '{name}'。"
                         f"可选: warmup_cosine, cosine, cyclic, one_cycle")


# =============================================================================
# 7. 训练流程 (支持 loss 平衡 / LR 调度器选择 / L-BFGS 精调)
# =============================================================================
def train_model(model, cfg: Config, params=None, diag=None, skip_adam=False):
    """单阶段 Adam 训练 + 可选 L-BFGS 精调。
    skip_adam=True 时跳过 Adam，直接进入 L-BFGS（用于断点续训）。
    """

    if not skip_adam:
        # ── 1. 学习率调度器 ────────────────────────────────────────────────
        lr_schedule = _create_lr_schedule(cfg)
        model.tf_optimizer = tf.keras.optimizers.legacy.Adam(lr_schedule, beta_1=0.99)

        print(f"\n[Train] LR schedule: {cfg.lr_schedule}, peak={cfg.peak_lr}, "
              f"total={cfg.total_epochs}, min={cfg.min_lr}")
        if cfg.lr_schedule == "cyclic":
            print(f"[Train]   cycle_period={cfg.lr_cycle_period}, "
                  f"decay={cfg.lr_cycle_decay}, min_factor={cfg.lr_cycle_min_factor}")

        # ── 2. Loss 平衡 ────────────────────────────────────────────────────
        if cfg.loss_balance_mode == "fixed":
            n_terms = model.adaptive_constant_func.adaptive_constant.shape[1]
            new_weights = np.ones((1, n_terms), dtype=np.float32)
            if n_terms >= 2:
                new_weights[0, 1] = cfg.fb_loss_weight
            model.adaptive_constant_func.adaptive_constant.assign(
                tf.constant(new_weights)
            )
            print(f"[Train] Loss balance: fixed, FB weight={cfg.fb_loss_weight:.1e}")
        elif cfg.loss_balance_mode == "auto":
            print(f"[Train] Loss balance: auto (GradNorm), alpha={cfg.loss_balance_alpha}")
        else:
            print(f"[Train] Loss balance: none (equal weights)")

        print(f"[Train] Architecture: Act={cfg.Act}, residual={cfg.use_residual}, "
              f"layers={cfg.layer_sizes}")

        # ── 3. Adam 训练阶段 ────────────────────────────────────────────────
        if diag is not None and cfg.diag_enabled:
            diag.snapshot(0)

        remaining = cfg.total_epochs
        global_epoch = 0

        while remaining > 0:
            chunk = min(cfg.diag_interval, remaining) if cfg.diag_enabled else remaining
            model.fit(tf_iter=chunk, newton_iter=0, batch_sz=cfg.batch_size)
            global_epoch += chunk
            remaining -= chunk
            print(f"  epoch {global_epoch}/{cfg.total_epochs}, loss={model.loss_history[-1]:.2f}")

            if diag is not None and cfg.diag_enabled and remaining > 0:
                diag.snapshot(global_epoch)

        if diag is not None and cfg.diag_enabled:
            diag.finalize()
    else:
        print("\n[Resume] 跳过 Adam 训练，从已保存权重恢复...")

    # ── 4. L-BFGS 精调阶段 ──────────────────────────────────────────────────
    if cfg.fine_tune_enabled:
        # 确保模型状态正确 (resume 后 optimizer 可能未初始化)
        if skip_adam:
            # 为 L-BFGS 准备一个 dummy optimizer（仅用于兼容性）
            dummy_lr = _create_lr_schedule(cfg)
            model.tf_optimizer = tf.keras.optimizers.legacy.Adam(dummy_lr, beta_1=0.99)
        print(f"\n[Fine-Tune] Starting L-BFGS fine-tuning "
              f"({cfg.fine_tune_epochs} steps, eager={cfg.fine_tune_eager})...")
        model.fit(tf_iter=0, newton_iter=cfg.fine_tune_epochs,
                  newton_eager=cfg.fine_tune_eager, batch_sz=cfg.batch_size)
        print(f"[Fine-Tune] Done. Final loss={model.loss_history[-1]:.2f}")

    return model


# =============================================================================
# 8. 可视化
# =============================================================================
def plot_results(model, params, cfg: Config, save_prefix='result'):
    """
    绘制结果图
    修改说明：参考Reynold.py，使用401x401高分辨率网格预测 + pcolormesh绘制均匀云图
    """
    # 样式配置 (参考 Reynold.py)
    Text_size = 18
    plt_rcParams_font_size = 20
    dpi_save = 600
    dpi_watch = 150
    cmap_choice = cm.RdYlBu_r
    
    # 全局字体设置
    plt.rcParams['font.size'] = plt_rcParams_font_size

    # 1. 生成高分辨率规则网格 (401 x 401)
    n_x, n_y = (401, 401) 
    
    # 获取根据物理参数定义的范围
    R_lim = params['R_lim']
    theta_lim = params['theta_lim']
    
    x_point = np.linspace(R_lim[0], R_lim[1], n_x)
    y_point = np.linspace(theta_lim[0], theta_lim[1], n_y)
    
    X, Y = np.meshgrid(x_point, y_point)
    # 展平以输入模型 (N, 2)
    X_Y_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    
    # 2. 模型预测 (直接在密集网格上预测，而非插值)
    # 注意：这里假设 model.u_model 接受 (N, 2) 的输入并返回列表 [p, gamma]
    u_pred_tensor = model.u_model(X_Y_star)
    
    # 提取压力 P 和 空化率 Gamma，并重塑回网格形状 (ny, nx)
    # u_pred_tensor[0] 是 P, u_pred_tensor[1] 是 Gamma
    p_pred = u_pred_tensor[0].numpy().reshape(n_y, n_x)
    gamma_pred = u_pred_tensor[1].numpy().reshape(n_y, n_x)
    
    # 3. 绘图 - 压力分布 P
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi_watch)
    # 使用 pcolormesh 替代 scatter，shading='auto' 处理网格
    sc1 = plt.pcolormesh(X, Y, p_pred, shading='auto', cmap=cmap_choice)
    
    # Colorbar 设置
    cbar = fig.colorbar(sc1)
    # 如果需要像Reynold.py那样设置刻度格式，可以取消下面注释
    # fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    # fmt.set_powerlimits((0, 0))
    # cbar = plt.colorbar(sc1, format=fmt)
    
    plt.xlabel(r'$R$', fontsize=Text_size)
    plt.ylabel(r'$\theta$', rotation=0, fontsize=Text_size) # theta通常横着放比较易读
    plt.title(r'Predicted $P(R, \theta)$')
    
    # 刻度字体
    plt.xticks(fontsize=Text_size)
    plt.yticks(fontsize=Text_size)
    
    # 保存
    plt.savefig(f'{save_prefix}_pressure_contour.png', bbox_inches='tight', dpi=dpi_save, pad_inches=0.1)
    plt.close()
    
    # 4. 绘图 - 空化函数 Gamma
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi_watch)
    sc2 = plt.pcolormesh(X, Y, gamma_pred, shading='auto', cmap=cmap_choice)
    cbar = fig.colorbar(sc2)
    
    plt.xlabel(r'$R$', fontsize=Text_size)
    plt.ylabel(r'$\theta$', rotation=0, fontsize=Text_size)
    plt.title(r'Predicted $\gamma(R, \theta)$')
    plt.xticks(fontsize=Text_size)
    plt.yticks(fontsize=Text_size)
    
    plt.savefig(f'{save_prefix}_cavitation_contour.png', bbox_inches='tight', dpi=dpi_save, pad_inches=0.1)
    plt.close()
    
    # 5. 绘图 - 膜厚 H (H_only)
    # 为了计算H，需要重新获取或传入H_func。
    H_func_plot, _ = create_H_func(params, cfg)
    
    # 将 numpy 的 meshgrid 转换为 tensor 用于计算
    X_tf = tf.constant(X, dtype=tf.float32)
    Y_tf = tf.constant(Y, dtype=tf.float32)
    
    # 计算 H 值并转回 numpy
    H_val = H_func_plot(X_tf, Y_tf).numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi_watch)
    sc3 = plt.pcolormesh(X, Y, H_val, shading='auto', cmap=cmap_choice)
    cbar = fig.colorbar(sc3)
    plt.xlabel(r'$R$', fontsize=Text_size)
    plt.ylabel(r'$\theta$', rotation=0, fontsize=Text_size)
    plt.title(r'Film Thickness $H(R, \theta)$')
    plt.xticks(fontsize=Text_size)
    plt.yticks(fontsize=Text_size)
    plt.savefig(f'{save_prefix}_H_only.png', bbox_inches='tight', dpi=dpi_save, pad_inches=0.1)
    plt.close()

    # 6. 绘图 - 训练 Loss (对数坐标)
    fig = plt.figure(figsize=(10, 6), dpi=dpi_watch)
    plt.axes(yscale='log') # 对应 Reynold.py 的 plt.axes(yscale='log')
    
    # 绘制 Total Loss
    plt.plot(np.array(model.epoch_history[1:]), np.array(model.loss_history), label='Total Loss')
    
    plt.xlabel('epoch', fontsize=Text_size)
    plt.ylabel('loss', fontsize=Text_size)
    plt.title('Training History')
    plt.xticks(fontsize=Text_size)
    plt.yticks(fontsize=Text_size)
    
    plt.savefig(f'{save_prefix}_loss_log.png', bbox_inches='tight', dpi=dpi_save, pad_inches=0.1)
    plt.close()

    print(f"figures saved to: {os.path.dirname(save_prefix)}")


# =============================================================================
# 8.5. 学习率曲线记录 & 绘图 (训练结束后自动调用)
# =============================================================================
def _compute_lr_curve(peak_lr, warmup_epochs, total_epochs, min_lr=1e-6,
                      record_every=10):
    """复现 WarmupCosineDecay 的学习率曲线，返回 (epochs, lrs)。"""
    epochs_list = []
    lrs_list = []
    for step in range(1, total_epochs + 1):
        if step < warmup_epochs:
            lr = peak_lr * (step / max(warmup_epochs, 1))
        else:
            decay_steps = total_epochs - warmup_epochs
            cosine = 0.5 * (1.0 + np.cos(
                np.pi * (step - warmup_epochs) / max(decay_steps, 1)
            ))
            lr = min_lr + (peak_lr - min_lr) * cosine
        if step % record_every == 0 or step == 1:
            epochs_list.append(step)
            lrs_list.append(float(lr))
    return epochs_list, lrs_list


def _plot_lr_curve(epochs, lrs, figures_dir, cfg, dpi=300):
    """绘制学习率曲线 (线性 + 对数坐标) 并保存到 figures/ 目录。"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=150)
    fig.suptitle(
        f"Learning Rate Schedule  (C{getattr(cfg, 'config_id', '?')})\n"
        f"peak={cfg.peak_lr:.0e}, min={cfg.min_lr:.0e}, "
        f"warmup={cfg.warmup_epochs}, total={cfg.total_epochs}",
        fontsize=16,
    )

    for ax, yscale, title in [
        (axes[0], 'linear', 'Linear Scale'),
        (axes[1], 'log',    'Log Scale'),
    ]:
        ax.plot(epochs, lrs, 'b-', lw=0.8)
        ax.axvline(cfg.warmup_epochs, color='gray', ls='--', lw=1, alpha=0.7,
                   label=f'warmup end ({cfg.warmup_epochs})')
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Learning Rate', fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        if yscale == 'log':
            ax.set_yscale('log')

    fig.tight_layout()
    save_path = os.path.join(figures_dir, 'learning_rate_schedule.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"[LR] 学习率曲线图保存至: {save_path}")


# =============================================================================
# 9. 主程序
# =============================================================================
def main(config_id=1, resume=False, resume_path=None):
    # 初始化配置 — 从 config/ 目录按序号加载
    from config import get_config
    cfg = get_config(config_id)
    print(f"[Config] 加载配置 C{config_id}: Act={cfg.Act}, residual={cfg.use_residual}, "
          f"layers={cfg.layer_sizes}, output={cfg.output_dir}")

    # ---- 设备设置 ----
    gpu_id = int(cfg.device)
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError(f"[Device] 没有检测到 GPU！无法使用 GPU:{gpu_id}")
    if gpu_id >= len(gpus):
        print(f"[Device] 警告: 请求 GPU:{gpu_id} 但只有 {len(gpus)} 张 GPU，改用 GPU:0")
        gpu_id = 0
    # 两步法：先隐藏全部 GPU，再只显示目标 GPU
    tf.config.set_visible_devices([], 'GPU')
    tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
    print(f"[Device] GPU {gpu_id} (physical: {gpus[gpu_id].name})")

    # ---- 输出目录结构 (匹配 PyTorch 版) ----
    output_dir = os.path.join(SCRIPT_DIR, cfg.output_dir)
    log_dir = os.path.join(output_dir, 'log')
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    models_dir = os.path.join(output_dir, 'models')
    figures_dir = os.path.join(output_dir, 'figures')
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(models_dir)
    ensure_dir(figures_dir)

    # 重定向 stdout 到日志文件
    log_path = os.path.join(log_dir, 'train.txt')
    tee = Tee(log_path)
    sys.stdout = tee

    # ---- 打印配置摘要 ----
    print("=" * 60)
    print("Reynolds Equation PINN Solver (TensorFlow)")
    print("=" * 60)
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Device: GPU (visible: {len(gpus)})" if gpus else "Device: CPU")
    print(f"Batch size: {cfg.batch_size if cfg.batch_size else 'full-batch'}")
    print(f"Core: {cfg.core}")
    print(f"Coslayer mode: {cfg.coslayer_mode}")
    print(f"Activation: {cfg.Act}")
    print(f"Residual: {cfg.use_residual}")
    print(f"Layer sizes: {cfg.layer_sizes}")
    if cfg.core == "pikan":
        print(f"PIKAN: grid={cfg.kan_grid_size}, order={cfg.kan_spline_order}, "
              f"layers={cfg.pikan_layer_sizes}")
    print(f"Output dir: {output_dir}")

    params = compute_physical_params(cfg)
    print(f"Lambda = {float(params['Lambda']):.4f}")
    print(f"P_i = {params['P_i']:.6f}, P_o = {params['P_o']:.6f}")
    print(f"R_lim = {params['R_lim']}")
    print(f"theta_lim = {params['theta_lim']}")

    # 创建膜厚函数和PDE模型
    H_func, theta_sym = create_H_func(params, cfg)
    H_func_tf = tf.function(H_func)  # 包装为 tf.function，避免 autograph 闭包穿透导致 NameError
    f_model_FBNS, f_model_FB = create_pde_models(H_func_tf, params)  # f_model_FB 保留备用

    # 创建计算域
    Domain = DomainND(["R", "theta"])
    Domain.add("R", params['R_lim'], cfg.domain_fidelity)
    Domain.add("theta", params['theta_lim'], cfg.domain_fidelity)
    Domain.X_f = Domain.generate_collocation_points(cfg.N_f, 1)

    # 添加螺旋槽边界配点
    add_R, add_theta = generate_groove_points(theta_sym, params, cfg)
    Domain.X_f = np.concatenate([Domain.X_f, np.concatenate([add_R, add_theta], 1)], 0)
    N_f_true = len(Domain.X_f)
    print(f"Total collocation points: {N_f_true}")

    # 设置边界条件
    lower_bc = dirichletBC(Domain, val=params['P_i'], var='R', target="lower")
    upper_bc = dirichletBC(Domain, val=params['P_o'], var='R', target="upper")
    BCs = [lower_bc, upper_bc]

    # 确定 u_model_switch: 8=mlp-two-output, 13=pikan-two-output
    if cfg.core == "pikan":
        u_model_switch = 13
    else:
        u_model_switch = 8

    # 创建并编译模型
    model = CollocationSolverND()
    model.compile(
        cfg.layer_sizes, [f_model_FBNS], Domain, BCs,
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

    # 最佳模型保存到 checkpoints/
    model.best_weights_path = os.path.join(ckpt_dir, 'epochs_best_model')

    # ---- 断点续训: 加载已保存权重 ----
    skip_adam = False
    if resume:
        load_path = resume_path if resume_path else model.best_weights_path
        if not os.path.exists(load_path + ".index") and not os.path.exists(load_path + ".data-00000-of-00001"):
            # 尝试 TF checkpoint 格式
            if os.path.exists(os.path.dirname(load_path)):
                ckpt_files = os.listdir(os.path.dirname(load_path))
                if ckpt_files:
                    print(f"[Resume] checkpoint 目录内容: {ckpt_files}")
            raise FileNotFoundError(
                f"[Resume] 找不到模型权重: {load_path}\n"
                f"  请确认权重文件存在于: {os.path.dirname(load_path)}"
            )
        model.u_model.load_weights(load_path)
        print(f"[Resume] 成功加载已训练权重: {load_path}")
        skip_adam = True

        # 如果用户没开 fine_tune，提示
        if not cfg.fine_tune_enabled:
            print("[Resume] 警告: fine_tune_enabled=False，将不会进行 L-BFGS 精调。")
            print("[Resume] 仅完成权重加载。如需精调，请在 config 中设置 fine_tune_enabled=True。")
    else:
        # 正常训练：初始化保存初始权重
        model.u_model.save_weights(model.best_weights_path)

    # ---- 初始化诊断 ----
    diag = None
    if cfg.diag_enabled:
        diag = TrainingDiagnostics(model, params, cfg, output_dir)

    # 训练
    print("Starting training..." if not skip_adam else "Starting fine-tuning...")
    model = train_model(model, cfg, params=params, diag=diag, skip_adam=skip_adam)

    # ---- 训练后诊断 ----
    if cfg.diag_enabled:
        print("\n[Post-Training] Running full diagnosis...")
        post_training_analysis(model, params, cfg, output_dir)
        gradient_impact_detailed_analysis(model, params, cfg, output_dir)

    # 保存最终模型到 models/
    model_name = f'reynolds_pinn_N{cfg.N_f}_epoch{cfg.total_epochs}'
    model_path = os.path.join(models_dir, model_name)
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # 保存 loss history 为 JSON
    loss_json_path = os.path.join(log_dir, 'loss_history.json')
    loss_data = {
        'loss_history': [float(v) if hasattr(v, 'numpy') else v
                         for v in model.loss_history],
        'epoch_history': model.epoch_history,
        'loss_all_history': [[float(vv) if hasattr(vv, 'numpy') else vv
                              for vv in v] for v in model.loss_all_history],
    }
    # ── 记录学习率曲线 ──────────────────────────────────────────────────────
    lr_epochs, lr_values = _compute_lr_curve(
        peak_lr=cfg.peak_lr,
        warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.total_epochs,
        min_lr=cfg.min_lr,
        record_every=10,
    )
    loss_data['lr_history'] = lr_values
    with open(loss_json_path, 'w') as f:
        json.dump(loss_data, f, indent=2)
    print(f"Loss history saved to: {loss_json_path}")

    # ── 绘制学习率曲线 ──────────────────────────────────────────────────────
    _plot_lr_curve(lr_epochs, lr_values, figures_dir, cfg, dpi=cfg.dpi_save)

    # 可视化结果保存到 figures/
    fig_prefix = os.path.join(figures_dir, model_name)
    plot_results(model, params, cfg, save_prefix=fig_prefix)

    # 恢复 stdout
    sys.stdout = tee.stdout
    tee.close()

    print(f"\nTraining complete! Output saved to: {output_dir}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reynolds PINN 训练")
    parser.add_argument("config_id", nargs="?", type=int, default=1,
                        help="配置序号 (对应 config/cN.py)，默认 1")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="从 checkpoints/epochs_best_model 恢复权重，"
                             "跳过 Adam 训练，直接进入 L-BFGS 精调")
    parser.add_argument("--resume-path", type=str, default=None,
                        help="指定权重恢复路径 (覆盖默认的 checkpoints/epochs_best_model)")
    args = parser.parse_args()
    model = main(config_id=args.config_id, resume=args.resume,
                 resume_path=args.resume_path)
