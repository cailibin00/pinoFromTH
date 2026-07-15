"""
配置 C1 — 基准对照 (tanh + 原版)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reynold_pinn import Config as BaseConfig


class Config(BaseConfig):
    # ========== 几何参数 (单位: m) ==========
    r_i = 47.0e-3
    r_o = 52.0e-3
    h_i = 3.0e-6
    K = 6.0

    # ========== 螺旋槽几何 ==========
    R_d_1_ratio = 1.043
    R_d_2_ratio = 1.106 * 2
    alpha_deg = 3.0
    h_texture_ratio = 3.0
    groove_ratio = 0.5

    # ========== 工况参数 ==========
    p_i = 0.1e6
    p_o_ratio = 1.5
    eta = 8.00e-4
    omega_rpm = 6000

    # ========== 数值参数 ==========
    N_f = 8000
    N_groove_b = 100
    N_groove_r = 50
    domain_fidelity = 100

    # ========== 模型架构 ==========
    Act = "tanh"
    core = "MLP"
    use_residual = False
    output_head_dim = 64
    coslayer_mode = "mlp"

    # ========== 训练参数 ==========
    layer_sizes = [2, 128, 128, 128, 256, 256, 256, 128, 128, 128, 2]
    total_epochs = 30000
    warmup_epochs = 1000
    peak_lr = 1e-3
    min_lr = 1e-6
    batch_size = 2048

    # ========== 学习率调度器 ==========
    lr_schedule = "warmup_cosine"
    lr_cycle_period = 5000
    lr_cycle_decay = 0.7
    lr_cycle_min_factor = 0.01

    # ========== Loss 平衡 ==========
    loss_balance_mode = "none"
    fb_loss_weight = 1.0
    loss_balance_alpha = 0.2

    # ========== L-BFGS 精调 ==========
    fine_tune_enabled = False
    fine_tune_epochs = 1000
    fine_tune_eager = True

    # ========== 诊断 ==========
    diag_enabled = True
    diag_interval = 1000

    # ========== 输出 ==========
    output_dir = "output_output_resF_tanh_largec1_tanh"
    device = "5"

    # ========== 绘图 ==========
    dpi_save = 600
    dpi_watch = 150
    text_size = 18
    font_size = 20
