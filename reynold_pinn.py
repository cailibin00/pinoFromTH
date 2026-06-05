"""
Reynolds方程PINN求解器 - 螺旋槽空化问题
基于TensorDiffEq框架，采用JFO空化模型

作者: [叶萌]
日期: 2026
"""
import os

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 输出目录 —— 所有训练结果集中存放
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

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
from tensordiffeq.utils import get_tf_model


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
    N_f = 4900             # 配点数
    N_groove_b = 50        # 槽边界采样点数
    N_groove_r = 20        # 槽径向边界采样点数
    domain_fidelity = 50   # 域网格密度
    
    # 训练参数
    layer_sizes = [2, 128, 128, 128, 128,128 ,2]
    N_train = 5000         # 每阶段训练迭代数
    NL_train = 4           # RAD细化轮数
    ratio_RAD_list = [0.03, 0.01]  # RAD采样比例

    # ── 模型架构开关 ─────────────────────────────────────────────────
    u_model_switch = 13    # 8=旧极坐标硬BC, 13=新Fourier解耦架构
    bc_switch = 1          # 1=改进硬BC(MLP_g+plateau σ), 2=纯软约束
    num_fourier_freq = 4   # 每坐标Fourier特征频率数 (2^0, 2^1, ..., 2^{L-1})·π
    embed_dim = 64         # R/θ MLP编码输出维度

    # ── 膜厚过渡类型 ─────────────────────────────────────────────────
    h_step_type = 'hermite'  # 'sigmoid' | 'hermite' | 'relu'
                             # hermite: C¹分段三次, 精确0/1在过渡区外
                             # sigmoid: C∞但过渡模糊 (旧行为)
                             # relu:    C⁰锐利, 导数只有一点不连续

    # ── 楔形项课程学习 (w_wedge 从 0→1 阶梯上升) ───────────────────
    w_wedge_init = 1e-2
    w_wedge_final = 1.0
    use_stop_gradient_H = True   # 截断楔形项中H的梯度 (方法A)
    use_point_weight = True      # 按|∂H/∂θ|逐点降权 (方法C)
    
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
    """创建膜厚函数 H(R, theta)

    支持三种过渡类型 (cfg.h_step_type):
        'hermite' — C¹分段三次多项式, 在过渡区外精确为0/1 (推荐)
        'sigmoid' — C∞光滑, 但过渡区模糊 (旧行为)
        'relu'    — C⁰锐利裁剪, 梯度更集中
    """
    R_d_1 = params['R_d_1']
    R_d_2 = params['R_d_2']
    r_g = params['r_g']
    alpha = params['alpha']
    h_texture = params['h_texture']
    h_i = params['h_i']
    groove_ratio = params['groove_ratio']
    K_val = float(params['K'].numpy())

    # 平滑参数
    N_xi = 50.0
    R_xi = 100.0
    xi_R = (params['R_lim'][1] - params['R_lim'][0]) / R_xi
    xi_theta = (params['theta_lim'][1] - params['theta_lim'][0]) / N_xi
    theta_offset = np.pi / 6  # 30度相位偏移

    step_type = getattr(cfg, 'h_step_type', 'sigmoid')

    def _step_fn(x):
        """过渡函数, 根据 step_type 选择"""
        if step_type == 'hermite':
            # C¹ 分段三次 Hermite: h(t)=3t²-2t³ for t∈[0,1]
            t = tf.clip_by_value(x, 0.0, 1.0)
            return 3 * t**2 - 2 * t**3
        elif step_type == 'relu':
            # C⁰ 锐利裁剪
            return tf.clip_by_value(x, 0.0, 1.0)
        else:
            # sigmoid (旧行为)
            return tf.math.sigmoid(x)

    def theta_sym(R):
        """螺旋线方程"""
        return tf.math.log(R / r_g) / tf.math.tan(alpha)+theta_offset

    def H_func(R, theta):
        """膜厚分布函数"""
        # 螺旋槽区域判断
        periodic_offsets = [0, -2*np.pi/K_val, -4*np.pi/K_val, 2*np.pi/K_val, 4*np.pi/K_val]
        periodic_terms = []
        for offset in periodic_offsets:
            term = (_step_fn((theta - theta_sym(R) + offset) / xi_theta) *
                   _step_fn((theta_sym(R) - theta + 2*np.pi/K_val*groove_ratio - offset) / xi_theta))
            periodic_terms.append(term)

        is_texture = (_step_fn((R - R_d_1) / xi_R) *
                     _step_fn((R_d_2 - R) / xi_R) *
                     sum(periodic_terms))

        H = 1.0 * (1 - is_texture) + (1.0 + h_texture / h_i) * is_texture
        return H

    return H_func, theta_sym


# =============================================================================
# 4. PDE残差模型
# =============================================================================
def create_pde_models(H_func, params, cfg: Config = None):
    """创建PDE残差函数

    方法ABC组合控梯策略:
        A. stop_gradient(H) 用于楔形项, 截断 ∂H/∂θ 对网络参数的梯度
        B. w_wedge 课程权重 (初值~1e-2 → 终值1.0), 先学扩散后学楔形
        C. 逐点权重 ∝ 1/(1+|∂H/∂θ|/mean), 自动压低槽边界处的残差量级
    """
    Lambda = params['Lambda']

    # 可调控件
    use_sg = getattr(cfg, 'use_stop_gradient_H', True)
    use_pw = getattr(cfg, 'use_point_weight', True)
    w_wedge_var = tf.Variable(
        getattr(cfg, 'w_wedge_init', 1e-2),
        trainable=False, dtype=tf.float32
    )

    def f_model_FBNS(u_model, R, theta):
        """Reynolds方程残差 + JFO稳定项 (ABC组合)"""
        p_vector = u_model(tf.concat([R, theta], 1))
        p, gamma = p_vector[0], p_vector[1]
        H = H_func(R, theta)

        # 压力梯度
        p_R = tf.gradients(p, R)[0]
        p_theta = tf.gradients(p, theta)[0]

        # ── 方法A: stop_gradient 截断楔形项中 H 的参数梯度 ──────────
        # 不能直接对 H 整体 stop_gradient 后再对 theta 求导，
        # 否则 tf.gradients(H_for_wedge, theta) 会返回 None。
        H_theta = tf.gradients(H, theta)[0]
        H_mult = H
        if use_sg:
            H_theta = tf.stop_gradient(H_theta)
            H_mult = tf.stop_gradient(H)

        # Poiseuille 扩散 (始终使用完整 H, 不受 stop_gradient 影响)
        part_1 = tf.gradients(R * H**3 * p_R, R)[0] / R
        part_2 = tf.gradients(H**3 * p_theta, theta)[0] / R**2

        # ── 方法B: 课程权重 w_wedge ──────────────────────────────
        w = w_wedge_var
        gamma_theta = tf.gradients(gamma, theta)[0]

        # 楔形效应项
        part_3_1 = -Lambda * H_theta * w
        part_3_2 = -Lambda * (-(gamma_theta * H_mult + gamma * H_theta)) * w

        # 稳定项 (处理空化边界)
        div_gamma = gamma_theta
        div_2_gamma = tf.gradients(div_gamma, theta)[0]
        div_p = tf.gradients(p, theta)[0]

        epsilon = 0.1
        tau = tf.stop_gradient((tf.math.abs(div_gamma) - div_gamma) * epsilon)
        tau_2 = tf.stop_gradient((div_p - tf.math.abs(div_p)) * epsilon)

        f_p_raw = part_1 + part_2 + part_3_1 + part_3_2 + div_2_gamma * tau * tau_2

        # ── 方法C: 逐点权重, 压低槽边突变区的loss贡献 ───────────
        if use_pw:
            # 使用原始 H 计算梯度 (不被 stop_gradient 影响)
            grad_H_theta = tf.gradients(H, theta)[0]
            mean_grad = tf.stop_gradient(
                tf.reduce_mean(tf.abs(grad_H_theta)) + 1e-8
            )
            point_weight = tf.stop_gradient(
                1.0 / (1.0 + tf.abs(grad_H_theta) / mean_grad)
            )
            f_p = point_weight * f_p_raw
        else:
            f_p = f_p_raw

        return f_p

    def f_model_FB(u_model, R, theta):
        """Fischer-Burmeister互补条件"""
        p_vector = u_model(tf.concat([R, theta], 1))
        p, gamma = p_vector[0], p_vector[1]
        return p + gamma - tf.math.sqrt(p**2 + gamma**2)

    def set_w_wedge(new_val):
        """更新楔形项课程权重"""
        w_wedge_var.assign(new_val)

    return f_model_FBNS, f_model_FB, set_w_wedge


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
# 6. 训练流程
# =============================================================================
def train_model(model, cfg: Config, N_f_true):
    """多阶段训练流程, 含 w_wedge 课程学习"""
    # 学习率配置
    lr_schedules = [
        {'boundaries': [20000, 40000], 'values': [1e-3, 1e-4, 1e-5]},
        {'boundaries': [20000, 40000], 'values': [1e-4, 1e-4, 1e-5]},
        {'boundaries': [20000, 40000], 'values': [1e-5, 1e-4, 1e-5]},
        {'boundaries': [20000, 40000], 'values': [1e-5, 1e-5, 1e-6]},
    ]

    # w_wedge 课程: 从 ~0.01 阶梯上升到 1.0
    # 阶段1: 几乎纯扩散 (poiseuille), 阶段4: 满楔形效应
    w_wedge_init = getattr(cfg, 'w_wedge_init', 1e-2)
    w_wedge_final = getattr(cfg, 'w_wedge_final', 1.0)
    n_stages = len(lr_schedules)
    w_wedge_values = [
        w_wedge_init + (w_wedge_final - w_wedge_init) * i / (n_stages - 1)
        for i in range(n_stages)
    ]

    for stage_idx, schedule in enumerate(lr_schedules):
        # 更新课程权重
        if hasattr(model, 'set_w_wedge'):
            w_val = w_wedge_values[stage_idx]
            model.set_w_wedge(w_val)
            print(f"[Stage {stage_idx+1}/{n_stages}] w_wedge = {w_val:.3e}")

        lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=schedule['boundaries'], values=schedule['values'])

        model.tf_optimizer = tf.keras.optimizers.Adam(lr_decay, beta_1=0.99)

        for _ in range(cfg.NL_train):
            model.fit(tf_iter=cfg.N_train, newton_iter=0)
            # 残差自适应细化
            model.RAD_FB(
                model.f_model_list + [model.f_model_FB],
                N_f_true,
                num_add_points_test=round(10 * N_f_true),
                num_add_points=[round(r * N_f_true) for r in cfg.ratio_RAD_list],
                k=1, c=1e-16
            )

    return model


# =============================================================================
# 7. 可视化
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
# 8. 主程序
# =============================================================================
def main():
    # 初始化配置
    cfg = Config()
    params = compute_physical_params(cfg)
    
    # 创建膜厚函数和PDE模型
    H_func, theta_sym = create_H_func(params, cfg)
    f_model_FBNS, f_model_FB, set_w_wedge = create_pde_models(H_func, params, cfg)

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

    # 创建并编译模型
    model = CollocationSolverND()
    model.compile(
        cfg.layer_sizes, [f_model_FBNS], Domain, BCs,
        u_model_switch=cfg.u_model_switch, two_output=True,
        none_zero=False, adapt_True=False,
        isAdaptive=False, MTL_adapt=False, PCGrad_true=True, Boundary_true=False,
        R_range=params['R_lim'], theta_range=params['theta_lim'],
        bc_switch=cfg.bc_switch, num_freq=cfg.num_fourier_freq,
        embed_dim=cfg.embed_dim
    )

    # 创建输出目录结构
    for d in [OUTPUT_DIR, CHECKPOINT_DIR, MODEL_DIR, FIGURE_DIR]:
        os.makedirs(d, exist_ok=True)

    model.best_weights_path = os.path.join(CHECKPOINT_DIR, 'epochs_best_model.weights.h5')
    model.u_model.save_weights(model.best_weights_path)  # 重新初始化到正确位置

    # 设置额外模型
    model.f_model_FB = tf.function(f_model_FB)
    model.f_model_list = [get_tf_model(f_model_FBNS)]
    model.set_w_wedge = set_w_wedge  # w_wedge 课程调度器

    # 训练
    print("Starting training...")
    model = train_model(model, cfg, N_f_true)

    # 保存模型到 output/models/
    model_name = f'reynolds_pinn_N{cfg.N_f}_iter{cfg.N_train*cfg.NL_train*4}'
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Model saved as: {model_path}")

    # 保存可视化到 output/figures/{model_name}/
    fig_dir = os.path.join(FIGURE_DIR, model_name)
    os.makedirs(fig_dir, exist_ok=True)
    fig_prefix = os.path.join(fig_dir, model_name)
    plot_results(model, params, cfg, save_prefix=fig_prefix)
    
    return model


if __name__ == "__main__":
    model = main()
