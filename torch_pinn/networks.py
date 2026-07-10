"""
Neural network architectures for PINN.
Faithful 1:1 port from tensordiffeq/networks.py (TensorFlow).

Key architectures:
- Coslayer_normalization: Fourier feature layer with trainable frequencies/phases
- Out_Imp_BC_layer: Learnable BC distance function (exponential)
- Out_Imp_BC_value_layer: 2-point cubic Hermite interpolation
- new_neural_period_polar_exactBC_two_output: Main model (switch=8)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# =============================================================================
# Coslayer_normalization - Fourier Feature Layer
# =============================================================================
class Coslayer_normalization(nn.Module):
    """
    Fourier feature encoding layer with trainable frequencies and phases.

    Architecture:
    1. Normalize R and theta to [-1, 1]
    2. Compute cos(pi * theta_norm + phi) as Fourier features
    3. Combine with R_norm via learned weights
    4. Apply bias and tanh activation

    This encodes periodic boundary conditions directly into the layer.
    """

    def __init__(self, units, r_lim, theta_lim, activation=nn.Tanh()):
        super(Coslayer_normalization, self).__init__()
        self.units = units
        self.r_lim = r_lim
        self.theta_lim = theta_lim

        # Trainable weights
        # kernel: [2, units] - combines R_norm and cos features
        self.kernel = nn.Parameter(torch.empty(2, units))
        # phy: [units] - phase shift for each Fourier feature
        self.phy = nn.Parameter(torch.empty(units))
        # K: scalar = pi (constant, not trained)
        self.register_buffer('K', torch.tensor(np.pi, dtype=torch.float32))

        self.use_bias = True
        self.bias = nn.Parameter(torch.empty(units))
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.kernel)
        init.xavier_uniform_(self.phy.unsqueeze(0))
        init.zeros_(self.bias)

    def forward(self, inputs):
        """
        Args:
            inputs: [N, 2] tensor with columns [R, theta]
        Returns:
            outputs: [N, units] Fourier features
            inputs_R: [N, 1] normalized R coordinate
        """
        # Extract R and theta columns
        inputs_r = inputs[:, 0:1]     # [N, 1]
        inputs_theta = inputs[:, 1:2]  # [N, 1]

        # Normalize to [-1, 1]
        inputs_R = 2.0 * (inputs_r - self.r_lim[0]) / (self.r_lim[1] - self.r_lim[0]) - 1.0
        inputs_Theta = 2.0 * (inputs_theta - self.theta_lim[0]) / (self.theta_lim[1] - self.theta_lim[0]) - 1.0

        # Fourier features: cos(pi * theta_norm + phi)
        # inputs_Theta: [N, 1], K: scalar, phy: [units]
        # -> [N, units]
        outputs = inputs_Theta * self.K  # [N, 1]
        outputs = outputs + self.phy.unsqueeze(0)  # [N, units]
        outputs = torch.cos(outputs)  # [N, units]

        # Combine: kernel[0,:] * R_norm + kernel[1,:] * cos_features
        # kernel: [2, units]
        outputs_2 = outputs * self.kernel[1:2, :]  # [N, units] * [1, units]

        # Broadcast R_norm to [N, units] by adding zero
        inputs_r_broadcast = inputs_R + 0.0 * self.phy.unsqueeze(0)  # [N, units]
        outputs_1 = inputs_r_broadcast * self.kernel[0:1, :]  # [N, units]

        outputs = outputs_1 + outputs_2

        if self.use_bias:
            outputs = outputs + self.bias.unsqueeze(0)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs, inputs_R


# =============================================================================
# Out_Imp_BC_layer - Learnable BC Distance Function
# =============================================================================
class Out_Imp_BC_layer(nn.Module):
    """
    Learnable exponential boundary condition distance function.

    sigma = p3 * (1 - exp(p1 * (-1 - R_norm))) * (1 - exp(p2 * (R_norm - 1)))

    This produces a smooth function that vanishes at R = -1 and R = +1
    (the normalized boundaries), with learnable shape parameters.
    """

    def __init__(self, para_exp_BC_initializer=1.0):
        super(Out_Imp_BC_layer, self).__init__()
        init_val = para_exp_BC_initializer if isinstance(para_exp_BC_initializer, float) else 1.0
        self.para_exp_BC_1 = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
        self.para_exp_BC_2 = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
        self.para_exp_BC_3 = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))

    def forward(self, inputs_R):
        """
        Args:
            inputs_R: [N, 1] normalized R coordinate in [-1, 1]
        Returns:
            sigma_func: [N, 1] distance function
        """
        sigma_func = self.para_exp_BC_3 * \
                     (1.0 - torch.exp(self.para_exp_BC_1 * (-1.0 - inputs_R))) * \
                     (1.0 - torch.exp(self.para_exp_BC_2 * (inputs_R - 1.0)))
        return sigma_func


# =============================================================================
# Out_Imp_BC_value_layer - Hermite Interpolation for BC Values
# =============================================================================
class Out_Imp_BC_value_layer(nn.Module):
    """
    Two-point cubic Hermite interpolation with learnable endpoint derivatives.

    g_func = bc[0] * (1 + 2*(R+1)/2) * ((R-1)/(-2))^2
           + bc[1] * (1 + 2*(R-1)/(-2)) * ((R+1)/2)^2
           + m0 * (R+1) * ((R-1)/(-2))^2
           + m1 * (R-1) * ((R+1)/2)^2

    where m0, m1 are learned derivative values at endpoints.
    """

    def __init__(self, bc_values, para_Hermite_BC_initializer=0.0):
        super(Out_Imp_BC_value_layer, self).__init__()
        self.bc_values = bc_values
        init_val = para_Hermite_BC_initializer if isinstance(para_Hermite_BC_initializer, float) else 0.0
        self.para_Hermite_BC_1 = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
        self.para_Hermite_BC_2 = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))

    def forward(self, inputs_R):
        """
        Args:
            inputs_R: [N, 1] normalized R coordinate in [-1, 1]
        Returns:
            g_func: [N, 1] interpolated boundary value
        """
        # Hermite basis functions
        H_00 = 1.0 + 2.0 * (inputs_R + 1.0) / 2.0
        H_10 = (inputs_R - 1.0) / (-2.0)
        H_01 = 1.0 + 2.0 * (inputs_R - 1.0) / (-2.0)
        H_11 = (inputs_R + 1.0) / 2.0

        g_func = (
            self.bc_values[0] * H_00 * (H_10 ** 2) +
            self.bc_values[1] * H_01 * (H_11 ** 2) +
            self.para_Hermite_BC_1 * (inputs_R + 1.0) * (H_10 ** 2) +
            self.para_Hermite_BC_2 * (inputs_R - 1.0) * (H_11 ** 2)
        )
        return g_func


# =============================================================================
# new_neural_period_polar_exactBC_two_output - Main Model (u_model_switch=8)
# =============================================================================
class new_neural_period_polar_exactBC_two_output(nn.Module):
    """
    Main PINN architecture for Reynolds equation with JFO cavitation.

    Architecture:
    - Coslayer_normalization: Fourier feature encoding for periodic theta
    - Dual-branch gating: tanh(gate * U + (1-gate) * V) per hidden layer
    - Out_Imp_BC_layer: Learnable BC enforcement for both outputs
    - Hermite interpolation: via x_1, x_2 branches
    - Two output heads: P (pressure) and gamma (cavitation fraction)
    - Both outputs via tanh^2 for non-negativity

    This is the u_model_switch=8 architecture from the TF code.
    """

    def __init__(self, layer_sizes, bc_values, r_lim, theta_lim):
        """
        Args:
            layer_sizes: [input_dim, hidden_0, ..., output_dim]
                         e.g. [2, 128, 128, 128, 128, 2]
            bc_values: [bc_lower, bc_upper] boundary pressure values
            r_lim: [r_min, r_max]
            theta_lim: [theta_min, theta_max]
        """
        super(new_neural_period_polar_exactBC_two_output, self).__init__()

        self.r_lim = r_lim
        self.theta_lim = theta_lim
        self.bc_values = bc_values
        self.layer_sizes = layer_sizes

        # Coslayer_normalization: Fourier feature encoding
        self.coslayer = Coslayer_normalization(
            units=layer_sizes[1],
            r_lim=r_lim,
            theta_lim=theta_lim,
            activation=nn.Tanh()
        )

        # Two feature branches (U and V) for gating
        # x_U, x_V: Dense(layer_sizes[2], tanh)
        hidden_width = layer_sizes[2]
        self.x_U = nn.Linear(layer_sizes[1], hidden_width)
        self.x_V = nn.Linear(layer_sizes[1], hidden_width)

        # Extra branches for Hermite interpolation
        # x_1, x_2: Dense(1, tanh)
        self.x_1 = nn.Linear(layer_sizes[1], 1)
        self.x_2 = nn.Linear(layer_sizes[1], 1)

        # Hidden layers with gating
        # For each width in layer_sizes[2:-1]:
        #   x_t = Dense(width, tanh)
        #   x = x_t * x_U + (1-x_t) * x_V
        self.hidden_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        self.hidden_widths = []
        for width in layer_sizes[2:-1]:
            self.gate_layers.append(nn.Linear(layer_sizes[1], width))
            self.hidden_widths.append(width)

        # Output heads
        self.p_head = nn.Linear(layer_sizes[1], 1)
        self.gamma_head = nn.Linear(layer_sizes[1], 1)

        # Out_Imp_BC layers: learnable BC distance functions
        self.bc_sigma_1 = Out_Imp_BC_layer(para_exp_BC_initializer=1.0)
        self.bc_sigma_2 = Out_Imp_BC_layer(para_exp_BC_initializer=1.0)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights matching TF's glorot_normal."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
        # Special initializations matching TF
        init.zeros_(self.p_head.bias)
        init.zeros_(self.gamma_head.bias)
        # gamma head kernel initialized to ~1e-6 (very small)
        init.constant_(self.gamma_head.weight, 1e-6)

    @property
    def num_params(self):
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _param_breakdown(self):
        """Detailed parameter count per sub‑module."""
        rows = []
        for name, child in self.named_children():
            n = sum(p.numel() for p in child.parameters() if p.requires_grad)
            rows.append((name, n))
        rows.append(('TOTAL', self.num_params))
        return rows

    def param_summary(self):
        """Pretty‑print parameter breakdown."""
        lines = [f"{'Module':<35s} {'Params':>10s}", "-" * 47]
        for name, n in self._param_breakdown():
            lines.append(f"{name:<35s} {n:>10,d}")
        return "\n".join(lines)

    def forward(self, inputs):

        # Fourier feature encoding
        x, inputs_R = self.coslayer(inputs)  # x: [N, layer_sizes[1]]

        # Two feature branches
        x_U = torch.tanh(self.x_U(x))  # [N, layer_sizes[2]]
        x_V = torch.tanh(self.x_V(x))  # [N, layer_sizes[2]]

        # Hermite interpolation branches
        x_1 = torch.tanh(self.x_1(x))  # [N, 1]
        x_2 = torch.tanh(self.x_2(x))  # [N, 1]

        # Hidden layers with gating mechanism
        # x_t = tanh(Dense(width)(x_original)) — gate computed from coslayer output
        # x = x_t * x_U + (1 - x_t) * x_V
        for i, gate_layer in enumerate(self.gate_layers):
            x_t = torch.tanh(gate_layer(x))  # gate from coslayer output
            x = x_t * x_U + (1.0 - x_t) * x_V

        # Output heads
        predictions = self.p_head(x)       # [N, 1] raw P
        prediction_g = self.gamma_head(x)  # [N, 1] raw gamma

        # BC enforcement: learnable distance functions
        sigma_func_1 = self.bc_sigma_1(inputs_R)  # [N, 1]
        sigma_func_2 = self.bc_sigma_2(inputs_R)  # [N, 1]

        # g_func_1: atanh-based interpolation of sqrt(bc_values)
        # g_func = atanh((sqrt(bc[1]) - sqrt(bc[0]))/2 * (R+1) + sqrt(bc[0]))
        bc_0 = self.bc_values[0]
        bc_1 = self.bc_values[1]
        g_func_1 = torch.atanh(
            (torch.sqrt(torch.tensor(bc_1, dtype=torch.float32, device=inputs.device)) -
             torch.sqrt(torch.tensor(bc_0, dtype=torch.float32, device=inputs.device))) / 2.0 *
            (inputs_R + 1.0) +
            torch.sqrt(torch.tensor(bc_0, dtype=torch.float32, device=inputs.device))
        )

        # g_func_2: 2-point cubic Hermite interpolation
        g_func_2 = (
            x_1 * (inputs_R + 1.0) * ((inputs_R - 1.0) / (-2.0)) ** 2 +
            x_2 * (inputs_R - 1.0) * ((inputs_R + 1.0) / 2.0) ** 2
        )

        g_func = g_func_1 + g_func_2

        # Apply BC enforcement and final activations
        predictions = g_func + sigma_func_1 * predictions
        prediction_g = sigma_func_2 * prediction_g

        # Non-negative outputs via tanh^2
        predictions = torch.tanh(predictions) ** 2
        prediction_g = torch.tanh(prediction_g) ** 2

        return [predictions, prediction_g]


# =============================================================================
# KANLinear - Kolmogorov-Arnold Network Layer
# =============================================================================
class KANLinear(nn.Module):
    """
    KAN (Kolmogorov-Arnold Network) linear layer.

    Replaces standard Linear(weight·x + bias) with learnable B-spline
    activation functions on each edge:

        y_j = Σ_i [ w_b_ij · silu(x_i) + w_s_ij · Σ_k c_ijk · B_k(x_i) ]

    where B_k are B-spline basis functions of order `spline_order` on a
    uniform grid of `grid_size` intervals over `grid_range`.

    This is based on the efficient-KAN formulation (arXiv:2403.07288).
    """

    def __init__(self, in_features, out_features, grid_size=5,
                 spline_order=3, grid_range=(-1, 1)):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Build uniform grid with K extra points on each side for boundary
        h = (grid_range[1] - grid_range[0]) / grid_size
        n_total = grid_size + 2 * spline_order + 1
        grid = torch.linspace(
            grid_range[0] - h * spline_order,
            grid_range[1] + h * spline_order,
            n_total
        )
        self.register_buffer('grid', grid)  # [G + 2K + 1]

        # Trainable parameters
        # Base weight: standard linear weight applied after silu activation
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

        # Spline scale weight: per-edge scale factor for spline contribution
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features))

        # Spline coefficients: one set of (G+K) coefficients per edge
        self.spline_coeffs = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        # Small initial spline contribution — let the network ramp it up
        init.normal_(self.spline_weight, mean=0.0, std=0.1)
        init.xavier_uniform_(
            self.spline_coeffs.view(self.out_features * self.in_features, -1)
        )

    def _b_spline_basis(self, x):
        """
        Compute B-spline basis functions via Cox-de Boor recurrence.

        Args:
            x: [batch, in_features] in grid_range
        Returns:
            basis: [batch, in_features, grid_size + spline_order]
        """
        grid = self.grid  # [G + 2K + 1]
        k = self.spline_order

        x = x.unsqueeze(-1)  # [batch, in, 1]

        # ---- Order-0 basis ----
        # N_{i,0}(x) = 1  if t_i <= x < t_{i+1}  else 0
        left = grid[:-1].view(1, 1, -1)   # [1, 1, G+2K]
        right = grid[1:].view(1, 1, -1)   # [1, 1, G+2K]

        bases = ((x >= left) & (x < right)).float()
        # Handle the rightmost point: x == grid[-1] → last basis = 1
        bases[:, :, -1] = bases[:, :, -1] + (x.squeeze(-1) == grid[-1]).float()

        # ---- Cox-de Boor recurrence for orders 1..K ----
        for order in range(1, k + 1):
            # N_{i,order}(x) =
            #   (x-t_i)      /(t_{i+order}-t_i)        * N_{i,order-1}(x)
            # + (t_{i+order+1}-x)/(t_{i+order+1}-t_{i+1}) * N_{i+1,order-1}(x)
            n_current = bases.shape[-1]  # = G + 2K - order + 1

            t_i   = grid[:n_current - 1].view(1, 1, -1)
            t_ik  = grid[order:order + n_current - 1].view(1, 1, -1)
            t_ik1 = grid[order + 1:order + n_current].view(1, 1, -1)
            t_i1  = grid[1:n_current].view(1, 1, -1)

            alpha = (x - t_i) / (t_ik - t_i + 1e-12)
            beta  = (t_ik1 - x) / (t_ik1 - t_i1 + 1e-12)

            alpha = torch.clamp(alpha, 0.0, 1.0)
            beta  = torch.clamp(beta, 0.0, 1.0)

            bases = alpha * bases[:, :, :-1] + beta * bases[:, :, 1:]

        # bases shape: [batch, in_features, grid_size + spline_order]
        return bases

    def forward(self, x):
        """
        Args:
            x: [batch, in_features]
        Returns:
            y: [batch, out_features]
        """
        # ---- Base path: silu activation + linear ----
        base = F.silu(x) @ self.base_weight.T  # [batch, out]

        # ---- Spline path: B-spline + coefficients ----
        basis = self._b_spline_basis(x)  # [batch, in, G+K]

        # Scale spline coefficients by per-edge weight
        coeffs_scaled = self.spline_coeffs * self.spline_weight.unsqueeze(-1)
        # [out, in, G+K]

        # Contraction: sum over input features and basis functions
        spline = torch.einsum('bik,oik->bo', basis, coeffs_scaled)

        return base + spline


# =============================================================================
# PIKAN — Physics-Informed Kolmogorov-Arnold Network
# =============================================================================
class PIKAN_Polar_BC_Two_Output(nn.Module):
    """
    PIKAN architecture for Reynolds equation with JFO cavitation.

    Replaces the MLP hidden layers of the standard model with KAN layers
    while preserving:
      - Coslayer_normalization  (Fourier features for periodic θ)
      - Dual-branch gating      (per‑layer U/V blending)
      - Hermite interpolation   (two‑point BC value)
      - Out_Imp_BC_layer        (learnable BC distance)
      - tanh² output activation (non‑negativity)

    The KAN layers learn B-spline activation functions on each edge instead
    of fixed activations on nodes, giving higher expressivity per parameter.
    """

    def __init__(self, layer_sizes, bc_values, r_lim, theta_lim,
                 kan_grid_size=5, kan_spline_order=3):
        """
        Args:
            layer_sizes: [input_dim, cos_units, hidden, ..., hidden, output_dim]
                         e.g. [2, 46, 36, 36, 36, 2]
            bc_values:   [bc_lower, bc_upper]
            r_lim:       [r_min, r_max]
            theta_lim:   [theta_min, theta_max]
            kan_grid_size:    G — number of B‑spline grid intervals
            kan_spline_order: K — polynomial order of B‑splines
        """
        super(PIKAN_Polar_BC_Two_Output, self).__init__()

        self.r_lim = r_lim
        self.theta_lim = theta_lim
        self.bc_values = bc_values
        self.layer_sizes = layer_sizes

        # ---- 1. Fourier feature encoding (same as MLP version) ----
        self.coslayer = Coslayer_normalization(
            units=layer_sizes[1],
            r_lim=r_lim,
            theta_lim=theta_lim,
            activation=nn.Tanh()
        )

        cos_units = layer_sizes[1]
        hid_width = layer_sizes[2]

        # ---- 2. KAN branches for U / V feature representations ----
        self.x_U = KANLinear(cos_units, hid_width,
                             grid_size=kan_grid_size,
                             spline_order=kan_spline_order)
        self.x_V = KANLinear(cos_units, hid_width,
                             grid_size=kan_grid_size,
                             spline_order=kan_spline_order)

        # ---- 3. KAN branches for Hermite interpolation ----
        self.x_1 = KANLinear(cos_units, 1,
                             grid_size=kan_grid_size,
                             spline_order=kan_spline_order)
        self.x_2 = KANLinear(cos_units, 1,
                             grid_size=kan_grid_size,
                             spline_order=kan_spline_order)

        # ---- 4. Per‑layer gate modules (KAN, then tanh) ----
        self.gate_layers = nn.ModuleList()
        self.hidden_widths = []
        for width in layer_sizes[2:-1]:
            self.gate_layers.append(
                KANLinear(cos_units, width,
                          grid_size=kan_grid_size,
                          spline_order=kan_spline_order)
            )
            self.hidden_widths.append(width)

        # ---- 5. Output heads (plain Linear — final projections) ----
        self.p_head = nn.Linear(cos_units, 1)
        self.gamma_head = nn.Linear(cos_units, 1)

        # ---- 6. Learnable BC distance functions (same as MLP) ----
        self.bc_sigma_1 = Out_Imp_BC_layer(para_exp_BC_initializer=1.0)
        self.bc_sigma_2 = Out_Imp_BC_layer(para_exp_BC_initializer=1.0)

        self._init_weights()

    def _init_weights(self):
        """Initialize output heads matching the MLP convention."""
        for m in [self.p_head, self.gamma_head]:
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        init.zeros_(self.p_head.bias)
        init.zeros_(self.gamma_head.bias)
        init.constant_(self.gamma_head.weight, 1e-6)

    @property
    def num_params(self):
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _param_breakdown(self):
        """Detailed parameter count per sub‑module (for logging)."""
        rows = []
        for name, child in self.named_children():
            n = sum(p.numel() for p in child.parameters() if p.requires_grad)
            rows.append((name, n))
        rows.append(('TOTAL', self.num_params))
        return rows

    def param_summary(self):
        """Pretty‑print parameter breakdown."""
        lines = []
        lines.append(f"{'Module':<35s} {'Params':>10s}")
        lines.append("-" * 47)
        for name, n in self._param_breakdown():
            lines.append(f"{name:<35s} {n:>10,d}")
        return "\n".join(lines)

    def forward(self, inputs):
        """
        Args:
            inputs: [N, 2] — columns [R, theta]
        Returns:
            (predictions, prediction_g): tuple of [N, 1] tensors
        """
        # ---- Fourier feature encoding ----
        x, inputs_R = self.coslayer(inputs)  # x: [N, cos_units]

        # ---- U / V feature branches (with tanh for gating range) ----
        x_U = torch.tanh(self.x_U(x))  # [N, hid_width]
        x_V = torch.tanh(self.x_V(x))  # [N, hid_width]

        # ---- Hermite interpolation branches ----
        x_1 = torch.tanh(self.x_1(x))  # [N, 1]
        x_2 = torch.tanh(self.x_2(x))  # [N, 1]

        # ---- Gated hidden layers (same logic as MLP) ----
        for gate_layer in self.gate_layers:
            x_t = torch.tanh(gate_layer(x))       # gate from coslayer output
            x = x_t * x_U + (1.0 - x_t) * x_V     # blend U / V

        # ---- Output heads ----
        predictions   = self.p_head(x)       # [N, 1] raw P
        prediction_g  = self.gamma_head(x)   # [N, 1] raw γ

        # ---- BC distance functions ----
        sigma_func_1 = self.bc_sigma_1(inputs_R)  # [N, 1]
        sigma_func_2 = self.bc_sigma_2(inputs_R)  # [N, 1]

        # ---- g_func: atanh interpolation of sqrt(bc) (same as MLP) ----
        bc0 = self.bc_values[0]
        bc1 = self.bc_values[1]
        g_func_1 = torch.atanh(
            (torch.sqrt(torch.tensor(bc1, dtype=torch.float32, device=inputs.device)) -
             torch.sqrt(torch.tensor(bc0, dtype=torch.float32, device=inputs.device))) / 2.0 *
            (inputs_R + 1.0) +
            torch.sqrt(torch.tensor(bc0, dtype=torch.float32, device=inputs.device))
        )

        # ---- g_func_2: Hermite interpolation ----
        g_func_2 = (
            x_1 * (inputs_R + 1.0) * ((inputs_R - 1.0) / (-2.0)) ** 2 +
            x_2 * (inputs_R - 1.0) * ((inputs_R + 1.0) / 2.0) ** 2
        )

        g_func = g_func_1 + g_func_2

        # ---- Apply BC enforcement ----
        predictions  = g_func + sigma_func_1 * predictions
        prediction_g = sigma_func_2 * prediction_g

        # ---- Non‑negative outputs via tanh² ----
        predictions  = torch.tanh(predictions) ** 2
        prediction_g = torch.tanh(prediction_g) ** 2

        return [predictions, prediction_g]


# =============================================================================
# Utility — parameter counting
# =============================================================================
def count_model_params(model):
    """Return (trainable, total) parameter counts for a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def auto_pikan_layer_sizes(mlp_layer_sizes, kan_grid_size, kan_spline_order,
                           target_params=None):
    """
    Auto‑compute PIKAN layer sizes that approximately match the
    parameter count of the MLP defined by `mlp_layer_sizes`.

    IMPORTANT: The gating mechanism requires cos_units == hidden_width
    (because x is overwritten each iteration). So the PIKAN format is
    [2, C, C, C, C, 2] with a single width C.

    Args:
        mlp_layer_sizes:  e.g. [2, 128, 128, 128, 128, 2]
        kan_grid_size:    B‑spline grid intervals G
        kan_spline_order: B‑spline polynomial order K
        target_params:    target param count (auto‑computed from MLP if None)

    Returns:
        pikan_layer_sizes: [2, C, C, C, C, 2]
    """
    if target_params is None:
        bc_values = [0.0, 1.0]
        r_lim = [0.0, 1.0]
        theta_lim = [0.0, 1.0]
        mlp_model = new_neural_period_polar_exactBC_two_output(
            mlp_layer_sizes, bc_values, r_lim, theta_lim
        )
        target_params, _ = count_model_params(mlp_model)

    M = kan_grid_size + kan_spline_order + 2  # param multiplier per KAN edge

    # PIKAN formula with C == H (required by gating):
    #   Total = (5*C² + 2*C)*M + 5*C + 8
    #   Solve for C.
    best_C = None
    best_diff = float('inf')
    for C in range(8, 120):
        actual = (5 * C * C + 2 * C) * M + 5 * C + 8
        diff = abs(actual - target_params)
        if diff < best_diff:
            best_diff = diff
            best_C = C

    return [2, best_C, best_C, best_C, best_C, 2]
