"""
配置 C2 — SiLU 激活 + 残差连接
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
    core = "MLP"
    use_residual = False       
    output_head_dim = 128
    gamma_output_transform = "sigmoid"
    coslayer_mode = "mlp"

    # ========== 训练参数 ==========
    layer_sizes = [2, 64 ,128, 128,256,256, 256,128, 128, 64 ,2]
    total_epochs = 25000
    warmup_epochs = 1000
    peak_lr = 1e-3
    min_lr = 1e-6
    batch_size = 2048

    # ========== 诊断 ==========
    diag_enabled = True
    diag_interval = 5000

    # ========== 输出 ==========
    output_dir = "output_resF_silu_small_epoch25000"
    device = "5"

    # ========== 绘图 ==========
    dpi_save = 600
    dpi_watch = 150
    text_size = 18
    font_size = 20

    # ========== 学习率调度器 ==========
    # 推荐: "cyclic"（对此问题loss景观崎岖效果好）| "warmup_cosine"（稳定baseline）
    lr_schedule = "cyclic"  # warmup_cosine | cyclic | cosine | one_cycle
    lr_cycle_period = 5000  # 每 5000 epochs restart
    lr_cycle_decay = 0.7    # 每次 restart peak_lr 降为 70%
    lr_cycle_min_factor = 0.01

    # ========== Loss 平衡 ==========
    # 推荐: "fixed" + fb_loss_weight=1000（JFO 问题 FB 项需要强加权）
    loss_balance_mode = "fixed"
    fb_loss_weight = 1 # fb权重
    p_gamma_loss_weight = 1e3 # P*gamma 互补乘积权重
    loss_balance_alpha = 0.2 # auto 模式下 EMA 平滑系数

    # ========== Reynolds 损失缩放 (缩小 PDE 残差项) ==========
    # schedule 上限: log-space 1e-8 → 1e-6 → 1e-4 → max，光滑渐进升温
    reynolds_weight_mode = "fixed"  # "fixed" | "adaptive"
    reynolds_loss_weight = 1e-4        # fixed 模式: 直接缩放因子
    reynolds_weight_target = 0.01      # adaptive: loss-driven 目标贡献值
    reynolds_weight_min = 1e-9         # adaptive: 起始权重 (epoch 0)
    reynolds_weight_max = 1e-3         # adaptive: 最终权重 (epoch → total)

    # ========== L-BFGS 精调 ==========
    # 推荐: True + fine_tune_epochs=1000（Adam落地后精调边界）
    fine_tune_enabled = False
    fine_tune_epochs = 1000
    fine_tune_eager = False
