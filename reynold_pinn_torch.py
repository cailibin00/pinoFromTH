"""
Reynolds Equation PINN Solver - Spiral Groove Cavitation Problem.
Faithful 1:1 port from 0504/reynold_pinn.py (TensorFlow -> PyTorch).

Solves the Reynolds equation for spiral-groove thrust bearings with
JFO (Jakobsson-Floberg-Olsson) cavitation model using PINN.
"""

import os
import sys
import json

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

# Torch PINN imports
from torch_pinn.boundaries import dirichletBC
from torch_pinn.domains import DomainND
from torch_pinn.models import CollocationSolverND
from torch_pinn.networks import new_neural_period_polar_exactBC_two_output, count_model_params
from torch_pinn.utils import Tee, ensure_dir


# =============================================================================
# 1. Configuration
# =============================================================================
class Config:
    """Centralized configuration matching 0504/reynold_pinn.py exactly."""

    # Geometry (units: m)
    r_i = 47.0e-3           # inner radius
    r_o = 52.0e-3           # outer radius
    h_i = 3.0e-6            # equilibrium film thickness
    K = 6.0                 # number of periods (360 deg / 6)

    # Spiral groove geometry
    R_d_1_ratio = 1.043     # groove start position ratio
    R_d_2_ratio = 1.106 * 2 # groove end position ratio (times 2 = full penetration)
    alpha_deg = 3.0         # spiral angle (degrees)
    h_texture_ratio = 3.0   # groove depth / equilibrium thickness ratio
    groove_ratio = 0.5      # groove width ratio

    # Operating conditions
    p_i = 0.1e6             # inner pressure (Pa)
    p_o_ratio = 1.5         # outer/inner pressure ratio
    eta = 8.00e-4           # dynamic viscosity (Pa·s)
    omega_rpm = 6000        # rotational speed (rpm)

    # Numerical parameters
    N_f = 4900              # number of collocation points
    N_groove_b = 50         # groove boundary sample points
    N_groove_r = 20         # groove radial boundary sample points
    domain_fidelity = 50    # domain mesh density

    # Training parameters
    N_train = 5000          # training iterations per stage
    NL_train = 4            # RAD refinement rounds
    ratio_RAD_list = [0.03, 0.01]  # RAD sampling ratios

    # Hardware / batch
    device = "cuda"         # "cuda", "cpu", or "auto" (auto-select CUDA if available)
    batch_size = None       # minibatch size; None = full-batch, int = stochastic minibatch

    output_dir = "output_torch1"

    # Model architecture
    core = "mlp"            # "mlp" (standard MLP) or "pikan" (KAN-based architecture)
    layer_sizes = [2, 128, 128, 256, 256, 256, 128, 2]  # MLP layer sizes
    output_head_dim = 64    # hidden dim inside deep output heads (P: 128→64→1, γ: 128→64→cat→64→1)
    # PIKAN params (only used when core == "pikan")
    kan_grid_size = 5       # B‑spline grid intervals
    kan_spline_order = 3    # B‑spline polynomial order
    pikan_layer_sizes = [2, 64, 64, 64, 64, 2]  # None = auto‑compute to match MLP param count; or specify manually

    # Plotting
    dpi_save = 600
    dpi_watch = 150
    text_size = 18
    font_size = 20


# =============================================================================
# 2. Physical Parameter Computation
# =============================================================================
def compute_physical_params(cfg):
    """Compute dimensionless parameters matching TF version exactly."""
    r_base = cfg.r_o
    p_c = 0.0               # cavitation pressure
    p_o = cfg.p_o_ratio * cfg.p_i + p_c
    p_base = 10 * p_o
    omega = cfg.omega_rpm * 2 * np.pi / 60

    # Dimensionless parameters
    Lambda = (6 * cfg.eta * omega * r_base**2) / (cfg.h_i**2 * p_base)
    P_i = cfg.p_i / p_base
    P_o = p_o / p_base
    R_lim = [cfg.r_i / r_base, cfg.r_o / r_base]
    theta_lim = [0.0, 2 * np.pi / cfg.K]

    # Spiral groove parameters
    R_d_1 = cfg.R_d_1_ratio * cfg.r_i / r_base
    R_d_2 = cfg.R_d_2_ratio * cfg.r_i / r_base
    alpha = cfg.alpha_deg / 180 * np.pi
    h_texture = cfg.h_texture_ratio * cfg.h_i

    return {
        'Lambda': Lambda,
        'P_i': P_i, 'P_o': P_o,
        'R_lim': R_lim, 'theta_lim': theta_lim,
        'R_d_1': R_d_1,
        'R_d_2': R_d_2,
        'r_g': R_d_1,
        'alpha': alpha,
        'h_texture': h_texture,
        'h_i': cfg.h_i,
        'K': cfg.K,
        'groove_ratio': cfg.groove_ratio,
    }


# =============================================================================
# 3. Film Thickness Function
# =============================================================================
def create_H_func(params, cfg):
    """
    Create film thickness function H(R, theta).
    Uses sigmoid smoothing for groove transitions - matching TF exactly.
    """
    R_d_1 = params['R_d_1']
    R_d_2 = params['R_d_2']
    r_g = params['r_g']
    alpha = params['alpha']
    h_texture = params['h_texture']
    h_i = params['h_i']
    groove_ratio = params['groove_ratio']
    K_val = cfg.K

    # Smoothing parameters (matching TF)
    N_xi = 50.0
    R_xi = 100.0
    xi_R = (params['R_lim'][1] - params['R_lim'][0]) / R_xi
    xi_theta = (params['theta_lim'][1] - params['theta_lim'][0]) / N_xi
    theta_offset = np.pi / 6  # 30 degree phase offset

    # Convert scalar params to float for use in torch functions
    r_g_val = float(r_g) if not isinstance(r_g, float) else r_g
    alpha_val = float(alpha) if not isinstance(alpha, float) else alpha
    R_d_1_val = float(R_d_1) if not isinstance(R_d_1, float) else R_d_1
    R_d_2_val = float(R_d_2) if not isinstance(R_d_2, float) else R_d_2
    h_texture_val = float(h_texture) if not isinstance(h_texture, float) else h_texture
    import math
    tan_alpha = math.tan(alpha_val)

    def theta_sym(R):
        """Spiral line equation."""
        return torch.log(R / r_g_val) / tan_alpha + theta_offset

    def H_func(R, theta):
        """Film thickness distribution with sigmoid smoothing."""
        periodic_offsets = [0, -2*np.pi/K_val, -4*np.pi/K_val, 2*np.pi/K_val, 4*np.pi/K_val]
        periodic_terms = []
        for offset in periodic_offsets:
            term = (torch.sigmoid((theta - theta_sym(R) + offset) / xi_theta) *
                    torch.sigmoid((theta_sym(R) - theta + 2*np.pi/K_val*groove_ratio - offset) / xi_theta))
            periodic_terms.append(term)

        is_texture = (torch.sigmoid((R - R_d_1_val) / xi_R) *
                      torch.sigmoid((R_d_2_val - R) / xi_R) *
                      sum(periodic_terms))

        H = 1.0 * (1 - is_texture) + (1.0 + h_texture_val / h_i) * is_texture
        return H

    return H_func, theta_sym


# =============================================================================
# 4. PDE Residual Models
# =============================================================================
def create_pde_models(H_func, params):
    """
    Create PDE residual functions.
    Matching TF create_pde_models exactly.

    Returns:
        f_model_FBNS: Reynolds equation with JFO stabilization
        f_model_FB: Fischer-Burmeister complementarity condition
    """
    Lambda = params['Lambda']

    def f_model_FBNS(u_model, R, theta):
        """Reynolds equation residual + JFO stabilization term."""
        p_vector = u_model(torch.cat([R, theta], dim=1))
        p, gamma = p_vector[0], p_vector[1]
        H = H_func(R, theta)

        # Pressure gradients (using torch.autograd.grad)
        p_R = torch.autograd.grad(p, R, grad_outputs=torch.ones_like(p),
                                   retain_graph=True, create_graph=True)[0]
        p_theta = torch.autograd.grad(p, theta, grad_outputs=torch.ones_like(p),
                                       retain_graph=True, create_graph=True)[0]

        # Reynolds equation terms
        term_1_inner = R * H**3 * p_R
        part_1 = torch.autograd.grad(term_1_inner, R, grad_outputs=torch.ones_like(term_1_inner),
                                      retain_graph=True, create_graph=True)[0] / R

        term_2_inner = H**3 * p_theta
        part_2 = torch.autograd.grad(term_2_inner, theta, grad_outputs=torch.ones_like(term_2_inner),
                                      retain_graph=True, create_graph=True)[0] / R**2

        H_theta = torch.autograd.grad(H, theta, grad_outputs=torch.ones_like(H),
                                       retain_graph=True, create_graph=True)[0]
        part_3_1 = -Lambda * H_theta

        neg_gamma_H_theta = torch.autograd.grad(-gamma * H, theta,
                                                 grad_outputs=torch.ones_like(gamma),
                                                 retain_graph=True, create_graph=True)[0]
        part_3_2 = -Lambda * neg_gamma_H_theta

        # Stabilization term (upwind-type for cavitation boundary)
        div_gamma = torch.autograd.grad(gamma, theta, grad_outputs=torch.ones_like(gamma),
                                         retain_graph=True, create_graph=True)[0]
        div_2_gamma = torch.autograd.grad(div_gamma, theta, grad_outputs=torch.ones_like(div_gamma),
                                           retain_graph=True, create_graph=True)[0]
        div_p = torch.autograd.grad(p, theta, grad_outputs=torch.ones_like(p),
                                     retain_graph=True, create_graph=True)[0]

        epsilon = 0.1
        tau = (torch.abs(div_gamma) - div_gamma) * epsilon  # stop_gradient in TF -> detach in torch
        tau_2 = (div_p - torch.abs(div_p)) * epsilon
        tau = tau.detach()
        tau_2 = tau_2.detach()

        f_p = part_1 + part_2 + part_3_1 + part_3_2 + div_2_gamma * tau * tau_2
        return f_p

    def f_model_FB(u_model, R, theta):
        """Fischer-Burmeister complementarity: P + gamma - sqrt(P^2 + gamma^2)."""
        p_vector = u_model(torch.cat([R, theta], dim=1))
        p, gamma = p_vector[0], p_vector[1]
        return p + gamma - torch.sqrt(p**2 + gamma**2)

    return f_model_FBNS, f_model_FB


# =============================================================================
# 5. Groove Boundary Point Generation
# =============================================================================
def generate_groove_points(theta_sym, params, cfg):
    """Generate additional collocation points along spiral groove boundaries."""
    R_lim = params['R_lim']
    theta_lim = params['theta_lim']
    R_d_1 = float(params['R_d_1'])
    K = float(params['K'])
    groove_ratio = params['groove_ratio']

    # Groove boundary line points
    R_list_np = np.linspace(R_d_1, R_lim[1], cfg.N_groove_b)
    R_t = torch.tensor(R_list_np, dtype=torch.float32)
    theta_1 = theta_sym(R_t).detach().numpy()
    theta_2 = theta_1 + 2 * np.pi / K * groove_ratio

    R_all = np.concatenate([R_list_np, R_list_np])
    theta_all = np.concatenate([theta_1, theta_2])

    # Periodic extension
    R_final, theta_final = [], []
    for offset in [0, -2*np.pi/K, -4*np.pi/K, 2*np.pi/K, 4*np.pi/K]:
        theta_shifted = theta_all + offset
        mask = (theta_shifted > theta_lim[0]) & (theta_shifted < theta_lim[1])
        theta_final.extend(theta_shifted[mask])
        R_final.extend(R_all[mask])

    # Radial boundary points
    theta_radial = np.linspace(float(theta_sym(torch.tensor(R_d_1, dtype=torch.float32)).numpy()),
                                float(theta_sym(torch.tensor(R_d_1, dtype=torch.float32)).numpy()) + 2*np.pi/K*groove_ratio,
                                cfg.N_groove_r)
    for offset in [0, -2*np.pi/K, -4*np.pi/K, 2*np.pi/K, 4*np.pi/K]:
        theta_shifted = theta_radial + offset
        mask = (theta_shifted > theta_lim[0]) & (theta_shifted < theta_lim[1])
        filtered_theta = theta_shifted[mask]
        theta_final.extend(filtered_theta)
        R_final.extend([R_d_1] * len(filtered_theta))

    return np.array(R_final).reshape(-1, 1), np.array(theta_final).reshape(-1, 1)


# =============================================================================
# 6. Training Workflow
# =============================================================================
def train_model(model, cfg, N_f_true):
    """
    Multi-stage training workflow.
    Matching TF train_model exactly: 4 stages with piecewise LR + RAD refinement.
    """
    # Learning rate schedules (matching TF)
    lr_schedules = [
        {'boundaries': [20000, 40000], 'values': [1e-3, 1e-4, 1e-5]},
        {'boundaries': [20000, 40000], 'values': [1e-4, 1e-4, 1e-5]},
        {'boundaries': [20000, 40000], 'values': [1e-5, 1e-4, 1e-5]},
        {'boundaries': [20000, 40000], 'values': [1e-5, 1e-5, 1e-6]},
    ]

    for schedule in lr_schedules:
        # Update learning rate for this stage
        current_lr = schedule['values'][0]
        for param_group in model.tf_optimizer.param_groups:
            param_group['lr'] = current_lr

        # Store boundaries for piecewise LR
        model._lr_boundaries = schedule['boundaries']
        model._lr_values = schedule['values']
        model._lr_stage_epoch = 0

        for i_round in range(cfg.NL_train):
            # Run Adam training for this round
            model.fit(tf_iter=cfg.N_train, newton_iter=0)

            # RAD refinement after each round
            if model.f_model_FB is not None:
                model.RAD_FB(
                    model.f_model_list + [model.f_model_FB],
                    N_f_true,
                    num_add_points_test=round(10 * N_f_true),
                    num_add_points=[round(r * N_f_true) for r in cfg.ratio_RAD_list],
                    k=1, c=1e-16
                )

    return model


# =============================================================================
# 7. Visualization
# =============================================================================
def plot_results(model, params, cfg, H_func, save_prefix='result'):
    """
    Plot results: pressure, cavitation, film thickness, loss history.
    Matching TF plot_results: 401x401 high-res grid + pcolormesh.
    """
    Text_size = cfg.text_size
    dpi_save = cfg.dpi_save
    dpi_watch = cfg.dpi_watch
    cmap_choice = cm.RdYlBu_r
    plt.rcParams['font.size'] = cfg.font_size

    # High-resolution grid (401 x 401)
    n_x, n_y = (401, 401)
    R_lim = params['R_lim']
    theta_lim = params['theta_lim']

    x_point = np.linspace(R_lim[0], R_lim[1], n_x)
    y_point = np.linspace(theta_lim[0], theta_lim[1], n_y)
    X, Y = np.meshgrid(x_point, y_point)
    X_Y_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # Model prediction
    u_pred = model.predict(X_Y_star)
    p_pred = u_pred[0].reshape(n_y, n_x)
    gamma_pred = u_pred[1].reshape(n_y, n_x)

    # 1. Pressure contour
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

    # 2. Cavitation contour
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

    # 3. Film thickness H
    H_func_plot, _ = create_H_func(params, cfg)
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    with torch.no_grad():
        H_val = H_func_plot(X_t, Y_t).numpy()

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

    # 4. Training loss (log scale)
    fig = plt.figure(figsize=(10, 6), dpi=dpi_watch)
    plt.axes(yscale='log')
    plt.plot(np.array(model.epoch_history[1:]), np.array(model.loss_history), label='Total Loss')
    plt.xlabel('epoch', fontsize=Text_size)
    plt.ylabel('loss', fontsize=Text_size)
    plt.title('Training History')
    plt.xticks(fontsize=Text_size)
    plt.yticks(fontsize=Text_size)
    plt.savefig(f'{save_prefix}_loss_log.png', bbox_inches='tight', dpi=dpi_save, pad_inches=0.1)
    plt.close()

    print(f"Figures saved to: {os.path.dirname(save_prefix)}")


# =============================================================================
# 8. Main
# =============================================================================
def main():
    # Setup
    cfg = Config()

    # Resolve device
    if cfg.device == 'auto':
        use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        use_device = cfg.device

    # Output directories
    output_dir = os.path.join(SCRIPT_DIR, cfg.output_dir)
    log_dir = os.path.join(output_dir, 'log')
    ensure_dir(log_dir)

    # Redirect stdout to log file
    log_path = os.path.join(log_dir, 'train.txt')
    tee = Tee(log_path)
    sys.stdout = tee

    print("=" * 60)
    print("Reynolds Equation PINN Solver (PyTorch)")
    print("Faithful port from 0504 TensorFlow version")
    print("=" * 60)
    print(f"Device: {use_device}  (CUDA available: {torch.cuda.is_available()})")
    print(f"Batch size: {cfg.batch_size if cfg.batch_size else 'full-batch'}")
    print(f"Core: {cfg.core}")
    print(f"Layer sizes: {cfg.layer_sizes}")

    # Compute physical parameters
    params = compute_physical_params(cfg)
    print(f"Lambda = {params['Lambda']:.4f}")
    print(f"P_i = {params['P_i']:.6f}, P_o = {params['P_o']:.6f}")
    print(f"R_lim = {params['R_lim']}")
    print(f"theta_lim = {params['theta_lim']}")

    # Create film thickness and PDE models
    H_func, theta_sym = create_H_func(params, cfg)
    f_model_FBNS, f_model_FB = create_pde_models(H_func, params)

    # Create computational domain
    Domain = DomainND(["R", "theta"])
    Domain.add("R", params['R_lim'], cfg.domain_fidelity)
    Domain.add("theta", params['theta_lim'], cfg.domain_fidelity)
    Domain.X_f = Domain.generate_collocation_points(cfg.N_f, 1)

    # Add groove boundary points
    add_R, add_theta = generate_groove_points(theta_sym, params, cfg)
    Domain.X_f = np.concatenate([Domain.X_f, np.concatenate([add_R, add_theta], 1)], 0)
    N_f_true = len(Domain.X_f)
    print(f"Total collocation points: {N_f_true}")

    # Boundary conditions
    lower_bc = dirichletBC(Domain, val=params['P_i'], var='R', target="lower")
    upper_bc = dirichletBC(Domain, val=params['P_o'], var='R', target="upper")
    BCs = [lower_bc, upper_bc]

    # Create and compile model
    model = CollocationSolverND(device=use_device)

    # Resolve PIKAN layer sizes: auto‑compute if None
    if cfg.core == 'pikan' and cfg.pikan_layer_sizes is not None:
        effective_layer_sizes = cfg.pikan_layer_sizes
    else:
        effective_layer_sizes = cfg.layer_sizes

    model.compile(
        effective_layer_sizes, [f_model_FBNS], Domain, BCs,
        u_model_switch=8, two_output=True, none_zero=False, adapt_True=False,
        isAdaptive=False, MTL_adapt=False, PCGrad_true=True, Boundary_true=False,
        R_range=params['R_lim'], theta_range=params['theta_lim'],
        batch_size=cfg.batch_size,
        core=cfg.core,
        kan_grid_size=cfg.kan_grid_size,
        kan_spline_order=cfg.kan_spline_order,
        output_head_dim=cfg.output_head_dim,
    )

    # Print param‑count comparison
    mlp_dummy = new_neural_period_polar_exactBC_two_output(
        cfg.layer_sizes, [params['P_i'], params['P_o']],
        params['R_lim'], params['theta_lim']
    )
    mlp_params, _ = count_model_params(mlp_dummy)
    pikan_params = sum(p.numel() for p in model.u_model.parameters() if p.requires_grad)
    print(f"Param count — MLP: {mlp_params:,}  |  PIKAN: {pikan_params:,}" +
          (f"  (delta: {pikan_params - mlp_params:+,})" if cfg.core == 'pikan' else ""))

    # Set save paths
    model.best_weights_path = os.path.join(output_dir, 'checkpoints', 'epochs_best_model.pt')
    ensure_dir(os.path.dirname(model.best_weights_path))
    model.save_weights(model.best_weights_path)

    # Set extra models (matching TF)
    model.f_model_FB = f_model_FB
    model.f_model_list = [f_model_FBNS]

    # Train
    print("Starting training...")
    model = train_model(model, cfg, N_f_true)

    # Save model
    model_name = f'reynolds_pinn_N{cfg.N_f}_iter{cfg.N_train * cfg.NL_train * 4}'
    model_path = os.path.join(output_dir, 'models', model_name)
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Save loss history as JSON
    loss_json_path = os.path.join(log_dir, 'loss_history.json')
    with open(loss_json_path, 'w') as f:
        json.dump({
            'loss_history': model.loss_history,
            'epoch_history': model.epoch_history,
            'loss_all_history': model.loss_all_history,
        }, f, indent=2)
    print(f"Loss history saved to: {loss_json_path}")

    # Plot results
    figures_dir = os.path.join(output_dir, 'figures', model_name)
    ensure_dir(figures_dir)
    plot_results(model, params, cfg, H_func, save_prefix=os.path.join(figures_dir, model_name))

    # Restore stdout
    sys.stdout = tee.stdout
    tee.close()

    print(f"\nTraining complete! Output saved to: {output_dir}")
    return model


if __name__ == "__main__":
    model = main()
