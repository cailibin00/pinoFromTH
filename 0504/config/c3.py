"""
配置 C3 — SiLU + 残差 + 宽网络
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
    Act = "silu"
    core = "mlp"
    use_residual = True
    output_head_dim = 64
    coslayer_mode = "mlp"

    # ========== PIKAN 参数 ==========
    kan_grid_size = 5
    kan_spline_order = 3
    pikan_layer_sizes = [2, 64, 64, 64, 64, 2]

    # ========== 训练参数 ==========
    layer_sizes = [2, 256, 256, 256, 128, 128, 2]   # ← 更宽
    total_epochs = 30000
    warmup_epochs = 1000
    peak_lr = 1e-3
    min_lr = 1e-6
    batch_size = 2048

    # ========== 诊断 ==========
    diag_enabled = True
    diag_interval = 1000

    # ========== 输出 ==========
    output_dir = "output_c3_silu_wide"
    device = "5"

    # ========== 绘图 ==========
    dpi_save = 600
    dpi_watch = 150
    text_size = 18
    font_size = 20
