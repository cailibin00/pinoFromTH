"""
Neural network architectures for PINN.
Faithful 1:1 port from tensordiffeq/networks.py (TensorFlow).

Key architectures:
- Coslayer_normalization: Fourier feature layer with trainable frequencies/phases
- Out_Imp_BC_layer: Learnable BC distance function (exponential)
- Out_Imp_BC_value_layer: 2-point cubic Hermite interpolation
- new_neural_period_polar_exactBC_two_output: Main model (switch=8)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


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

    def forward(self, inputs):
        """
        Args:
            inputs: [N, 2] tensor with columns [R, theta]
        Returns:
            (predictions, prediction_g): tuple of [N, 1] tensors
                predictions: P (pressure, non-negative via tanh^2)
                prediction_g: gamma (cavitation fraction, in [0, 1] via tanh^2)
        """
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
