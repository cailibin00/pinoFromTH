"""
Reynolds方程PINN求解器 (PyTorch版) - 螺旋槽空化问题
基于torch_pinn框架，采用JFO空化模型

将原 TensorFlow 版本完整移植到 PyTorch。
运行方式: python reynold_pinn_torch.py
"""

import os
import numpy as np
import torch
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

# ── PyTorch PINN imports ──
from torch_pinn import (
    FourierDecoupledPINN,
    TorchCollocationSolver,
    DomainND,
    dirichletBC,
    train_model_torch,
    piecewise_lr,
)

# =============================================================================
# 路径配置
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_torch")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# =============================================================================
# 1. 配置参数
# =============================================================================
class Config:
    """集中管理所有配置参数 (与原TF版完全一致)"""

    # 几何参数 (单位: m)
    r_i = 47.0e-3          # 内径
    r_o = 52.0e-3          # 外径
    h_i = 3.0e-6           # 平衡膜厚
    K = 6.0                # 周期数(360度分6份)

    # 螺旋槽几何
    R_d_1_ratio = 1.043    # 槽起始位置比例
    R_d_2_ratio = 1.106 * 2  # 槽结束位置比例
    alpha_deg = 3.0        # 螺旋角 (度)
    h_texture_ratio = 3.0  # 槽深/平衡膜厚比
    groove_ratio = 0.5     # 槽宽比

    # 工况参数
    p_i = 0.1e6            # 内径压力 (Pa)
    p_o_ratio = 1.5        # 外径/内径压力比
    eta = 8.00e-4          # 动力粘度 (Pa·s)
    omega_rpm = 6000       # 转速 (rpm)

    # 数值参数
    N_f = 10000            # 配点数 (100×100网格, 提升空化边界分辨率)
    N_groove_b = 80        # 槽边界采样点数 (沿螺旋边界加密)
    N_groove_r = 32        # 槽径向边界采样点数 (槽起点径向线加密)
    domain_fidelity = 80   # 域网格密度 (BC边界网格分辨率)

    # 训练参数
    layer_sizes = [2, 128, 128, 128, 128, 128, 2]
    N_train = 4000         # 每阶段训练迭代数 (配合更多配点的适度调整)
    NL_train = 5           # RAD细化轮数 (更多轮次渐进优化边界)
    ratio_RAD_list = [0.05, 0.02]  # RAD采样比例 (每轮在高残差区增加5%+2%配点)
    batch_size = 4096      # Mini-batch大小 (None或0=全批量; 推荐2048~8192)

    # ── 模型架构开关 ──
    u_model_switch = 8     # 8=SimpleMLPPINN (vanilla MLP), 13=FourierDecoupledPINN
    bc_switch = 1          # 1=硬约束BC (g_net+hermite σ, 原始方案), 2=纯软约束
    num_fourier_freq = 4   # (仅switch=13使用) Fourier特征频率数
    embed_dim = 64         # (仅switch=13使用) R/θ MLP编码输出维度

    # ── 膜厚过渡类型 ──
    h_step_type = 'sigmoid'  # 'sigmoid' | 'hermite' | 'relu'

    # 绘图参数
    dpi_save = 600
    dpi_watch = 150
    text_size = 18
    font_size = 20


# =============================================================================
# 2. 物理参数计算
# =============================================================================
def compute_physical_params(cfg: Config):
    """计算无量纲化参数 (PyTorch版)"""
    r_base = cfg.r_o
    p_c = 0.0
    p_o = cfg.p_o_ratio * cfg.p_i + p_c
    p_base = 10 * p_o
    omega = cfg.omega_rpm * 2 * np.pi / 60

    # 无量纲参数
    Lambda = (6 * cfg.eta * omega * r_base**2) / (cfg.h_i**2 * p_base)
    P_i = cfg.p_i / p_base
    P_o = p_o / p_base
    R_lim = [cfg.r_i / r_base, cfg.r_o / r_base]
    theta_lim = [0.0, 2 * np.pi / cfg.K]

    # 螺旋槽参数
    R_d_1 = cfg.R_d_1_ratio * cfg.r_i / r_base
    R_d_2 = cfg.R_d_2_ratio * cfg.r_i / r_base
    alpha = cfg.alpha_deg / 180 * np.pi
    h_texture = cfg.h_texture_ratio * cfg.h_i

    return {
        'Lambda': torch.tensor(Lambda, dtype=torch.float32),
        'P_i': P_i, 'P_o': P_o,
        'R_lim': R_lim, 'theta_lim': theta_lim,
        'R_d_1': torch.tensor(R_d_1, dtype=torch.float32),
        'R_d_2': torch.tensor(R_d_2, dtype=torch.float32),
        'r_g': torch.tensor(R_d_1, dtype=torch.float32),
        'alpha': torch.tensor(alpha, dtype=torch.float32),
        'h_texture': torch.tensor(h_texture, dtype=torch.float32),
        'h_i': cfg.h_i,
        'K': torch.tensor(cfg.K, dtype=torch.float32),
        'groove_ratio': cfg.groove_ratio,
    }


# =============================================================================
# 3. 膜厚函数
# =============================================================================
def create_H_func(params, cfg: Config):
    """创建膜厚函数 H(R, theta) — 简单sigmoid过渡, 固定ξ

    不使用hermite过渡, 不使用可训练ξ, 不使用ABC控梯策略。
    回归最简单的sigmoid平滑膜厚。
    """
    R_d_1 = params['R_d_1']
    R_d_2 = params['R_d_2']
    r_g = params['r_g']
    alpha = params['alpha']
    h_texture = params['h_texture']
    h_i = params['h_i']
    groove_ratio = params['groove_ratio']
    K_val = float(params['K'].item())

    # 固定过渡宽度
    xi_R = (params['R_lim'][1] - params['R_lim'][0]) / 100.0
    xi_theta = (params['theta_lim'][1] - params['theta_lim'][0]) / 50.0
    theta_offset = np.pi / 6

    def theta_sym(R):
        """螺旋线方程"""
        return torch.log(R / r_g) / torch.tan(alpha) + theta_offset

    def H_func(R, theta):
        """膜厚分布函数 — 纯sigmoid过渡"""
        periodic_offsets = [0, -2 * np.pi / K_val, -4 * np.pi / K_val,
                           2 * np.pi / K_val, 4 * np.pi / K_val]
        periodic_terms = []
        for offset in periodic_offsets:
            term = (torch.sigmoid((theta - theta_sym(R) + offset) / xi_theta) *
                    torch.sigmoid((theta_sym(R) - theta + 2 * np.pi / K_val * groove_ratio - offset) / xi_theta))
            periodic_terms.append(term)

        is_texture = (torch.sigmoid((R - R_d_1) / xi_R) *
                      torch.sigmoid((R_d_2 - R) / xi_R) *
                      sum(periodic_terms))

        H = 1.0 * (1 - is_texture) + (1.0 + h_texture / h_i) * is_texture
        return H

    return H_func, theta_sym


# =============================================================================
# 4. PDE残差模型
# =============================================================================
def create_pde_models(H_func, params, cfg: Config = None):
    """创建PDE残差函数 — 标准Reynolds方程, 无控梯策略

    不包含:
      - stop_gradient (detach H)
      - w_wedge 课程学习
      - point_weight 逐点降权
      - 稳定项 τ*τ_2
    回归最简单的PINN PDE残差形式。
    """
    Lambda = params['Lambda']

    def f_model_FBNS(u_model, R, theta):
        """标准 Reynolds 方程残差 (简单版)"""
        p, gamma = u_model(torch.cat([R, theta], dim=1))
        H = H_func(R, theta)

        # 压力梯度
        p_R = torch.autograd.grad(
            p, R, grad_outputs=torch.ones_like(p),
            create_graph=True, retain_graph=True
        )[0]
        p_theta = torch.autograd.grad(
            p, theta, grad_outputs=torch.ones_like(p),
            create_graph=True, retain_graph=True
        )[0]

        # ∂H/∂θ (楔形项)
        H_theta = torch.autograd.grad(
            H, theta, grad_outputs=torch.ones_like(H),
            create_graph=True, retain_graph=True
        )[0]

        # γ 的 θ 导数
        gamma_theta = torch.autograd.grad(
            gamma, theta, grad_outputs=torch.ones_like(gamma),
            create_graph=True, retain_graph=True
        )[0]

        # Poiseuille 扩散项
        term1 = R * H**3 * p_R
        part_1 = torch.autograd.grad(
            term1, R, grad_outputs=torch.ones_like(term1),
            create_graph=True, retain_graph=True
        )[0] / R

        term2 = H**3 * p_theta
        part_2 = torch.autograd.grad(
            term2, theta, grad_outputs=torch.ones_like(term2),
            create_graph=True, retain_graph=True
        )[0] / R**2

        # 楔形效应项 (直接使用, 无课程权重)
        part_3_1 = -Lambda * H_theta
        part_3_2 = -Lambda * (-(gamma_theta * H + gamma * H_theta))

        f_p = part_1 + part_2 + part_3_1 + part_3_2
        return f_p

    def f_model_FB(u_model, R, theta):
        """Fischer-Burmeister互补条件"""
        p, gamma = u_model(torch.cat([R, theta], dim=1))
        return p + gamma - torch.sqrt(p**2 + gamma**2)

    # 兼容旧接口: set_w_wedge 为 no-op
    def set_w_wedge(new_val):
        pass

    return f_model_FBNS, f_model_FB, set_w_wedge


# =============================================================================
# 5. 螺旋槽边界配点生成
# =============================================================================
def generate_groove_points(theta_sym, params, cfg: Config):
    """在螺旋槽边界生成额外配点 (纯numpy, 无需改动)"""
    R_lim = params['R_lim']
    theta_lim = params['theta_lim']
    R_d_1 = float(params['R_d_1'].item())
    K = float(params['K'].item())
    groove_ratio = params['groove_ratio']

    # 槽边界线配点
    R_list = np.linspace(R_d_1, R_lim[1], cfg.N_groove_b)
    R_tensor = torch.tensor(R_list, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        theta_1 = theta_sym(R_tensor).numpy().flatten()
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
    theta_radial = np.linspace(float(theta_sym(R_tensor)[0].item()),
                               float(theta_sym(R_tensor)[0].item()) + 2*np.pi/K*groove_ratio,
                               cfg.N_groove_r)
    for offset in [0, -2*np.pi/K, -4*np.pi/K, 2*np.pi/K, 4*np.pi/K]:
        theta_shifted = theta_radial + offset
        mask = (theta_shifted > theta_lim[0]) & (theta_shifted < theta_lim[1])
        filtered_theta = theta_shifted[mask]
        theta_final.extend(filtered_theta)
        R_final.extend([R_d_1] * len(filtered_theta))

    return np.array(R_final).reshape(-1, 1), np.array(theta_final).reshape(-1, 1)


# =============================================================================
# 6. 可视化
# =============================================================================
def plot_results(model, params, cfg: Config, H_func, save_prefix='result'):
    """绘制结果图 (PyTorch版)"""
    Text_size = 18
    plt_rcParams_font_size = 20
    dpi_save = 600
    dpi_watch = 150
    cmap_choice = cm.RdYlBu_r

    plt.rcParams['font.size'] = plt_rcParams_font_size

    # 高分辨率规则网格
    n_x, n_y = (401, 401)
    R_lim = params['R_lim']
    theta_lim = params['theta_lim']

    x_point = np.linspace(R_lim[0], R_lim[1], n_x)
    y_point = np.linspace(theta_lim[0], theta_lim[1], n_y)
    X, Y = np.meshgrid(x_point, y_point)
    X_Y_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # 模型预测
    p_pred, gamma_pred = model.predict(X_Y_star)
    p_pred = p_pred.reshape(n_y, n_x)
    gamma_pred = gamma_pred.reshape(n_y, n_x)

    # 压力分布
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi_watch)
    sc1 = plt.pcolormesh(X, Y, p_pred, shading='auto', cmap=cmap_choice)
    cbar = fig.colorbar(sc1)
    plt.xlabel(r'$R$', fontsize=Text_size)
    plt.ylabel(r'$\theta$', rotation=0, fontsize=Text_size)
    plt.title(r'Predicted $P(R, \theta)$')
    plt.xticks(fontsize=Text_size)
    plt.yticks(fontsize=Text_size)
    plt.savefig(f'{save_prefix}_pressure_contour.png', bbox_inches='tight', dpi=dpi_save, pad_inches=0.1)
    plt.close()

    # 空化函数
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

    # 膜厚 H
    R_t = torch.tensor(X, dtype=torch.float32)
    theta_t = torch.tensor(Y, dtype=torch.float32)
    with torch.no_grad():
        H_val = H_func(R_t, theta_t).numpy()

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

    # Loss 曲线
    fig = plt.figure(figsize=(10, 6), dpi=dpi_watch)
    plt.axes(yscale='log')
    valid_len = min(len(model.epoch_history) - 1, len(model.loss_history))
    if valid_len > 0:
        epochs = np.array(model.epoch_history[1:valid_len + 1])
        losses = np.array(model.loss_history[:valid_len])
        plt.plot(epochs, losses, label='Total Loss')
    plt.xlabel('epoch', fontsize=Text_size)
    plt.ylabel('loss', fontsize=Text_size)
    plt.title('Training History')
    plt.xticks(fontsize=Text_size)
    plt.yticks(fontsize=Text_size)
    plt.legend()
    plt.savefig(f'{save_prefix}_loss_log.png', bbox_inches='tight', dpi=dpi_save, pad_inches=0.1)
    plt.close()

    print(f"figures saved to: {os.path.dirname(save_prefix)}")


# =============================================================================
# 7. 主程序
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
    model = TorchCollocationSolver()
    model.compile(
        cfg.layer_sizes, [f_model_FBNS], Domain, BCs,
        u_model_switch=cfg.u_model_switch, two_output=True,
        none_zero=False, adapt_True=False,
        isAdaptive=False, MTL_adapt=False, PCGrad_true=True, Boundary_true=False,
        R_range=params['R_lim'], theta_range=params['theta_lim'],
        bc_switch=cfg.bc_switch, batch_size=cfg.batch_size
    )

    # 创建输出目录
    for d in [OUTPUT_DIR, CHECKPOINT_DIR, MODEL_DIR, FIGURE_DIR]:
        os.makedirs(d, exist_ok=True)

    model.best_weights_path = os.path.join(CHECKPOINT_DIR, 'epochs_best_model.pt')
    model.save_weights(model.best_weights_path)

    # 设置额外模型引用
    model.f_model_FB = f_model_FB
    model.f_model_list = [f_model_FBNS]
    model.set_w_wedge = set_w_wedge

    # 训练
    print("Starting training (PyTorch)...")
    model = train_model_torch(model, cfg, N_f_true)

    # 保存模型
    model_name = f'reynolds_pinn_N{cfg.N_f}_iter{cfg.N_train*cfg.NL_train*4}'
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Model saved as: {model_path}")

    # 保存可视化
    fig_dir = os.path.join(FIGURE_DIR, model_name)
    os.makedirs(fig_dir, exist_ok=True)
    fig_prefix = os.path.join(fig_dir, model_name)
    plot_results(model, params, cfg, H_func, save_prefix=fig_prefix)

    return model


if __name__ == "__main__":
    model = main()
