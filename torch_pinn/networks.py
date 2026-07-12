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
    Separate R / θ encoding with independent MLP pathways before fusion.

    Architecture (v2):
    1. Normalize R and θ to [-1, 1]
    2. θ pathway:  cos(π·θ_norm + φ)  →  MLP_θ  →  θ_features   [N, half]
    3. R pathway:  R_norm              →  MLP_R  →  R_features   [N, half]
    4. Concat(θ_features, R_features)  →  activation  →  x       [N, units]

    The old design merged R via a single scalar coefficient per channel
    (kernel[0,i] × R_norm).  Now R gets its own two‑layer MLP, giving
    both coordinates rich independent representations before fusion.
    """

    def __init__(self, units, r_lim, theta_lim, activation=nn.Tanh()):
        super(Coslayer_normalization, self).__init__()
        self.units = units
        self.r_lim = r_lim
        self.theta_lim = theta_lim

        half = units // 2

        # ---- θ pathway ----
        # Fourier encoding with trainable phases
        self.phy = nn.Parameter(torch.empty(units))
        self.register_buffer('K', torch.tensor(np.pi, dtype=torch.float32))
        # θ MLP:  fourier(units) → SiLU → hidden(units) → SiLU → θ_features(half)
        self.theta_fc1 = nn.Linear(units, units)
        self.theta_fc2 = nn.Linear(units, half)

        # ---- R pathway ----
        # R MLP:  R_norm(1) → SiLU → hidden(units) → SiLU → R_features(half)
        self.r_fc1 = nn.Linear(1, units)
        self.r_fc2 = nn.Linear(units, half)

        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        # θ pathway
        init.xavier_uniform_(self.phy.unsqueeze(0))
        init.xavier_uniform_(self.theta_fc1.weight)
        init.zeros_(self.theta_fc1.bias)
        init.xavier_uniform_(self.theta_fc2.weight)
        init.zeros_(self.theta_fc2.bias)

        # R pathway
        init.xavier_uniform_(self.r_fc1.weight)
        init.zeros_(self.r_fc1.bias)
        init.xavier_uniform_(self.r_fc2.weight)
        init.zeros_(self.r_fc2.bias)

    def forward(self, inputs):
        """
        Args:
            inputs: [N, 2] tensor with columns [R, theta]
        Returns:
            outputs:  [N, units]  fused R/θ features
            inputs_R: [N, 1]      normalized R (for BC sigma functions)
        """
        # ---- Normalize both coordinates to [-1, 1] ----
        inputs_r = inputs[:, 0:1]
        inputs_theta = inputs[:, 1:2]

        inputs_R = 2.0 * (inputs_r - self.r_lim[0]) / (self.r_lim[1] - self.r_lim[0]) - 1.0
        inputs_Theta = 2.0 * (inputs_theta - self.theta_lim[0]) / (self.theta_lim[1] - self.theta_lim[0]) - 1.0

        # ---- θ pathway: Fourier features → MLP ----
        # cos(π·θ_norm + φ)  →  FC → SiLU  →  FC → SiLU
        theta_fourier = torch.cos(inputs_Theta * self.K + self.phy.unsqueeze(0))  # [N, units]
        theta_feat = F.silu(self.theta_fc1(theta_fourier))                         # [N, units]
        theta_feat = F.silu(self.theta_fc2(theta_feat))                            # [N, half]

        # ---- R pathway: raw coordinate → MLP ----
        # R_norm  →  FC → SiLU  →  FC → SiLU
        r_feat = F.silu(self.r_fc1(inputs_R))                                      # [N, units]
        r_feat = F.silu(self.r_fc2(r_feat))                                        # [N, half]

        # ---- Merge independent pathways ----
        outputs = torch.cat([theta_feat, r_feat], dim=1)                           # [N, units]

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

    Architecture (v2 — residuals + deep cross‑talk heads):
    ┌─────────────────────────────────────────────────────┐
    │  Coslayer_normalization (Fourier features)          │
    │  ├─ U_base, V_base  (fixed reference features)      │
    │  ├─ Hermite branches: x₁, x₂ → g_func₂              │
    │  └─ Hidden layers with gating + RESIDUAL skips:     │
    │       gate = tanh(Linear(C, w)(x_cos))              │
    │       main  = gate·U_w + (1−gate)·V_w               │
    │       x_out = main + skip_proj(x_prev)   ← residual │
    ├─────────────────────────────────────────────────────┤
    │  Deep Output Heads:                                 │
    │    P:  x → FC(H)→tanh → [FC(H)→tanh + skip] → FC(1)│
    │    γ:  x → FC(H)→tanh → [FC(H)→tanh + skip] →       │
    │         concat(P_hid, γ_hid) →                       │
    │         FC(H)→tanh → [FC(H)→tanh + skip] → FC(1)    │
    │         ↑ fuses pressure information                 │
    ├─────────────────────────────────────────────────────┤
    │  BC enforcement + tanh² output (unchanged)           │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, layer_sizes, bc_values, r_lim, theta_lim,
                 output_head_dim=64, use_residual=True):
        """
        Args:
            layer_sizes: [in_dim, cos_units, hidden_0, ..., hidden_N, out_dim]
                         e.g. [2, 128, 128, 256, 256, 256, 128, 2]
            bc_values:   [bc_lower, bc_upper]
            r_lim:       [r_min, r_max]
            theta_lim:   [theta_min, theta_max]
            output_head_dim: hidden dimension inside the deep output heads
            use_residual: whether to use residual skip connections in hidden layers
        """
        super(new_neural_period_polar_exactBC_two_output, self).__init__()

        self.r_lim = r_lim
        self.theta_lim = theta_lim
        self.bc_values = bc_values
        self.layer_sizes = layer_sizes
        self.head_dim = output_head_dim

        cos_units   = layer_sizes[1]          # Fourier-feature output width
        hidden_w    = list(layer_sizes[2:-1]) # all hidden widths
        base_width  = hidden_w[0]             # U / V baseline dimension
        last_dim    = hidden_w[-1]            # last hidden → input to output heads

        self.use_residual = use_residual

        # ---- 1. Fourier feature encoding ----
        self.coslayer = Coslayer_normalization(
            units=cos_units, r_lim=r_lim, theta_lim=theta_lim,
            activation=nn.Tanh()
        )

        # ---- 2. U / V base branches (at first hidden width) ----
        self.x_U = nn.Linear(cos_units, base_width)
        self.x_V = nn.Linear(cos_units, base_width)

        # ---- 3. Hermite interpolation branches ----
        self.x_1 = nn.Linear(cos_units, 1)
        self.x_2 = nn.Linear(cos_units, 1)

        # ---- 4. Gate layers (one per hidden width) ----
        # Gate inputs: first gate from coslayer, subsequent gates from
        # the EVOLVING hidden state (matching TF: each gate reads the
        # output of the previous gated blend, not the raw coslayer).
        gate_inputs = [cos_units] + hidden_w[:-1]
        self.gate_layers = nn.ModuleList([
            nn.Linear(gate_inputs[i], hidden_w[i]) for i in range(len(hidden_w))
        ])
        self.hidden_widths = hidden_w

        # ---- 5. U / V projection layers (only when w ≠ base_width) ----
        self.U_proj = nn.ModuleDict()
        self.V_proj = nn.ModuleDict()
        for i, w in enumerate(hidden_w):
            if w != base_width:
                self.U_proj[str(i)] = nn.Linear(base_width, w)
                self.V_proj[str(i)] = nn.Linear(base_width, w)

        # ---- 6. Skip projections for residual connections ----
        self.skip_proj = nn.ModuleList()
        for i in range(len(hidden_w) - 1):
            w_prev = hidden_w[i]
            w_next = hidden_w[i + 1]
            if w_prev != w_next:
                self.skip_proj.append(nn.Linear(w_prev, w_next))
            else:
                self.skip_proj.append(nn.Identity())

        # ---- 7. Deep Output Heads ----
        H = output_head_dim

        # --- Pressure head: last_dim → H → (residual H→H) → 1 ---
        self.p_fc1      = nn.Linear(last_dim, H)
        self.p_fc2      = nn.Linear(H, H)          # residual block
        self.p_fc_out   = nn.Linear(H, 1)

        # --- Gamma head (stage 1): last_dim → H → (residual H→H) ---
        self.g_fc1      = nn.Linear(last_dim, H)
        self.g_fc2      = nn.Linear(H, H)          # residual block

        # --- Gamma head (stage 2 — fuses pressure hidden state):
        #     concat(P_hid, γ_hid) → H → (residual H→H) → 1 ---
        self.g_cat_fc1  = nn.Linear(2 * H, H)
        self.g_cat_fc2  = nn.Linear(H, H)          # residual block
        self.g_fc_out   = nn.Linear(H, 1)

        # ---- 8. BC distance functions ----
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
        init.zeros_(self.p_fc_out.bias)
        init.zeros_(self.g_fc_out.bias)
        # Gamma output kernel initialized to ~1e-6 (very small)
        init.constant_(self.g_fc_out.weight, 1e-6)

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

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, inputs):

        # ---- Fourier feature encoding ----
        x, inputs_R = self.coslayer(inputs)       # x: [N, cos_units]

        # ---- U / V base features ----
        U_base = F.silu(self.x_U(x))               # [N, base_width]
        V_base = F.silu(self.x_V(x))               # [N, base_width]

        # ---- Hermite interpolation branches ----
        x_1 = torch.tanh(self.x_1(x))              # [N, 1]  — keep tanh: Hermite bounded
        x_2 = torch.tanh(self.x_2(x))              # [N, 1]  — keep tanh: Hermite bounded

        # ---- Gated hidden layers with RESIDUAL connections ----
        hidden_w   = self.hidden_widths
        base_width = hidden_w[0]
        prev = None

        for i, (w, gate_layer) in enumerate(zip(hidden_w, self.gate_layers)):
            # Gate signal from previous layer (or coslayer for first layer)
            # Matching TF: deeper gates see hierarchical features, not raw coslayer
            gate_input = x if prev is None else prev
            gate = torch.tanh(gate_layer(gate_input))       # [N, w]

            # U / V at current width
            if w == base_width:
                U_w, V_w = U_base, V_base
            else:
                U_w = F.silu(self.U_proj[str(i)](U_base))
                V_w = F.silu(self.V_proj[str(i)](V_base))

            # Gated blend
            main = gate * U_w + (1.0 - gate) * V_w   # [N, w]

            # Residual skip from previous layer
            if prev is not None and self.use_residual:
                skip = self.skip_proj[i - 1](prev)   # identity or projection
                main = main + skip

            prev = main

        # prev is now the last hidden state  [N, last_dim]
        H = self.head_dim

        # ================================================================
        #  Deep Output Heads
        # ================================================================

        # --- Pressure head ---
        #  last_dim → H  →  (H→H residual)  →  1
        p_h1 = F.silu(self.p_fc1(prev))
        p_h2 = F.silu(self.p_fc2(p_h1))
        if self.use_residual:
            p_h2 = p_h2 + p_h1                            # residual
        p_raw = self.p_fc_out(p_h2)                        # [N, 1]

        # --- Gamma head (stage 1) ---
        #  last_dim → H  →  (H→H residual)
        g_h1 = F.silu(self.g_fc1(prev))
        g_h2 = F.silu(self.g_fc2(g_h1))
        if self.use_residual:
            g_h2 = g_h2 + g_h1                             # residual

        # --- Gamma head (stage 2 — fuse pressure hidden state) ---
        #  concat(p_h2, g_h2)  →  H  →  (H→H residual)  →  1
        g_cat  = torch.cat([p_h2, g_h2], dim=1)            # [N, 2H]
        g_cat1 = F.silu(self.g_cat_fc1(g_cat))
        g_cat2 = F.silu(self.g_cat_fc2(g_cat1))
        if self.use_residual:
            g_cat2 = g_cat2 + g_cat1                       # residual
        g_raw  = self.g_fc_out(g_cat2)                     # [N, 1]

        # ================================================================
        #  BC enforcement (unchanged from original)
        # ================================================================
        sigma_func_1 = self.bc_sigma_1(inputs_R)           # [N, 1]
        sigma_func_2 = self.bc_sigma_2(inputs_R)           # [N, 1]

        bc_0 = self.bc_values[0]
        bc_1 = self.bc_values[1]
        g_func_1 = torch.atanh(
            (torch.sqrt(torch.tensor(bc_1, dtype=torch.float32, device=inputs.device)) -
             torch.sqrt(torch.tensor(bc_0, dtype=torch.float32, device=inputs.device))) / 2.0 *
            (inputs_R + 1.0) +
            torch.sqrt(torch.tensor(bc_0, dtype=torch.float32, device=inputs.device))
        )

        g_func_2 = (
            x_1 * (inputs_R + 1.0) * ((inputs_R - 1.0) / (-2.0)) ** 2 +
            x_2 * (inputs_R - 1.0) * ((inputs_R + 1.0) / 2.0) ** 2
        )

        g_func = g_func_1 + g_func_2

        predictions   = g_func + sigma_func_1 * p_raw
        prediction_g  = sigma_func_2 * g_raw

        # Non-negative outputs via tanh²
        predictions  = torch.tanh(predictions) ** 2
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
                 kan_grid_size=5, kan_spline_order=3, output_head_dim=64,
                 use_residual=True):
        """
        Args:
            layer_sizes: [input_dim, cos_units, hidden, ..., hidden, output_dim]
                         All hidden widths MUST be equal (required by KAN gating).
                         e.g. [2, 64, 64, 64, 64, 2]
            bc_values:   [bc_lower, bc_upper]
            r_lim:       [r_min, r_max]
            theta_lim:   [theta_min, theta_max]
            kan_grid_size:    G — number of B‑spline grid intervals
            kan_spline_order: K — polynomial order of B‑splines
            output_head_dim:  hidden dim inside deep output heads
            use_residual: whether to use residual skip connections
        """
        super(PIKAN_Polar_BC_Two_Output, self).__init__()

        self.r_lim = r_lim
        self.theta_lim = theta_lim
        self.bc_values = bc_values
        self.layer_sizes = layer_sizes
        self.head_dim = output_head_dim
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.use_residual = use_residual

        cos_units = layer_sizes[1]
        hidden_w  = list(layer_sizes[2:-1])
        base_width = hidden_w[0]
        last_dim   = hidden_w[-1]

        # ---- 1. Fourier feature encoding ----
        self.coslayer = Coslayer_normalization(
            units=cos_units, r_lim=r_lim, theta_lim=theta_lim,
            activation=nn.Tanh()
        )

        # ---- 2. KAN U / V base branches (at first hidden width) ----
        self.x_U = KANLinear(cos_units, base_width,
                             grid_size=kan_grid_size,
                             spline_order=kan_spline_order)
        self.x_V = KANLinear(cos_units, base_width,
                             grid_size=kan_grid_size,
                             spline_order=kan_spline_order)

        # ---- 3. KAN Hermite interpolation branches ----
        self.x_1 = KANLinear(cos_units, 1,
                             grid_size=kan_grid_size,
                             spline_order=kan_spline_order)
        self.x_2 = KANLinear(cos_units, 1,
                             grid_size=kan_grid_size,
                             spline_order=kan_spline_order)

        # ---- 4. KAN gate layers (one per hidden width) ----
        # Gate inputs: first gate from coslayer, subsequent gates from
        # the EVOLVING hidden state (matching TF).
        gate_inputs = [cos_units] + hidden_w[:-1]
        self.gate_layers = nn.ModuleList()
        self.hidden_widths = hidden_w
        for i, w in enumerate(hidden_w):
            self.gate_layers.append(
                KANLinear(gate_inputs[i], w,
                          grid_size=kan_grid_size,
                          spline_order=kan_spline_order)
            )

        # ---- 5. U / V projection layers (only when w ≠ base_width) ----
        self.U_proj = nn.ModuleDict()
        self.V_proj = nn.ModuleDict()
        for i, w in enumerate(hidden_w):
            if w != base_width:
                self.U_proj[str(i)] = KANLinear(base_width, w,
                                                grid_size=kan_grid_size,
                                                spline_order=kan_spline_order)
                self.V_proj[str(i)] = KANLinear(base_width, w,
                                                grid_size=kan_grid_size,
                                                spline_order=kan_spline_order)

        # ---- 6. Skip projections (Linear — cheap dim change) ----
        self.skip_proj = nn.ModuleList()
        for i in range(len(hidden_w) - 1):
            w_prev = hidden_w[i]
            w_next = hidden_w[i + 1]
            if w_prev != w_next:
                self.skip_proj.append(nn.Linear(w_prev, w_next))
            else:
                self.skip_proj.append(nn.Identity())

        # ---- 7. Deep Output Heads (same as MLP v2) ----
        H = output_head_dim

        # Pressure head: last_dim → H → (residual H→H) → 1
        self.p_fc1    = nn.Linear(last_dim, H)
        self.p_fc2    = nn.Linear(H, H)
        self.p_fc_out = nn.Linear(H, 1)

        # Gamma head stage 1: last_dim → H → (residual H→H)
        self.g_fc1    = nn.Linear(last_dim, H)
        self.g_fc2    = nn.Linear(H, H)

        # Gamma head stage 2 (fuses pressure hidden): 2H → H → (residual H→H) → 1
        self.g_cat_fc1 = nn.Linear(2 * H, H)
        self.g_cat_fc2 = nn.Linear(H, H)
        self.g_fc_out  = nn.Linear(H, 1)

        # ---- 8. BC distance functions ----
        self.bc_sigma_1 = Out_Imp_BC_layer(para_exp_BC_initializer=1.0)
        self.bc_sigma_2 = Out_Imp_BC_layer(para_exp_BC_initializer=1.0)

        self._init_weights()

    def _init_weights(self):
        """Initialize output heads matching the MLP convention."""
        for m in [self.p_fc_out, self.g_fc_out]:
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        init.zeros_(self.p_fc_out.bias)
        init.zeros_(self.g_fc_out.bias)
        init.constant_(self.g_fc_out.weight, 1e-6)

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

        # ---- U / V base features ----
        U_base = F.silu(self.x_U(x))  # [N, base_width]
        V_base = F.silu(self.x_V(x))  # [N, base_width]

        # ---- Hermite interpolation branches ----
        x_1 = torch.tanh(self.x_1(x))  # [N, 1]
        x_2 = torch.tanh(self.x_2(x))  # [N, 1]

        # ---- Gated hidden layers with RESIDUAL connections ----
        hidden_w   = self.hidden_widths
        base_width = hidden_w[0]
        prev = None

        for i, (w, gate_layer) in enumerate(zip(hidden_w, self.gate_layers)):
            # Gate signal from previous layer (or coslayer for first layer)
            gate_input = x if prev is None else prev
            gate = torch.tanh(gate_layer(gate_input))       # [N, w]

            # U / V at current width
            if w == base_width:
                U_w, V_w = U_base, V_base
            else:
                U_w = F.silu(self.U_proj[str(i)](U_base))
                V_w = F.silu(self.V_proj[str(i)](V_base))

            main = gate * U_w + (1.0 - gate) * V_w   # [N, w]

            # Residual skip
            if prev is not None and self.use_residual:
                skip = self.skip_proj[i - 1](prev)
                main = main + skip

            prev = main

        # ---- Deep Output Heads (same as MLP v2) ----
        H = self.head_dim

        # Pressure head
        p_h1  = F.silu(self.p_fc1(prev))
        p_h2  = F.silu(self.p_fc2(p_h1))
        if self.use_residual:
            p_h2 = p_h2 + p_h1
        p_raw = self.p_fc_out(p_h2)

        # Gamma head stage 1
        g_h1 = F.silu(self.g_fc1(prev))
        g_h2 = F.silu(self.g_fc2(g_h1))
        if self.use_residual:
            g_h2 = g_h2 + g_h1

        # Gamma head stage 2 (fuse pressure info)
        g_cat  = torch.cat([p_h2, g_h2], dim=1)
        g_cat1 = F.silu(self.g_cat_fc1(g_cat))
        g_cat2 = F.silu(self.g_cat_fc2(g_cat1))
        if self.use_residual:
            g_cat2 = g_cat2 + g_cat1
        g_raw  = self.g_fc_out(g_cat2)

        # ---- BC distance functions ----
        sigma_func_1 = self.bc_sigma_1(inputs_R)
        sigma_func_2 = self.bc_sigma_2(inputs_R)

        # ---- g_func ----
        bc0 = self.bc_values[0]
        bc1 = self.bc_values[1]
        g_func_1 = torch.atanh(
            (torch.sqrt(torch.tensor(bc1, dtype=torch.float32, device=inputs.device)) -
             torch.sqrt(torch.tensor(bc0, dtype=torch.float32, device=inputs.device))) / 2.0 *
            (inputs_R + 1.0) +
            torch.sqrt(torch.tensor(bc0, dtype=torch.float32, device=inputs.device))
        )

        g_func_2 = (
            x_1 * (inputs_R + 1.0) * ((inputs_R - 1.0) / (-2.0)) ** 2 +
            x_2 * (inputs_R - 1.0) * ((inputs_R + 1.0) / 2.0) ** 2
        )

        g_func = g_func_1 + g_func_2

        # ---- Apply BC enforcement ----
        predictions  = g_func + sigma_func_1 * p_raw
        prediction_g = sigma_func_2 * g_raw

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
