import torch
from torch import nn


class FourierDecoupledPINN(nn.Module):
    def __init__(self, layer_sizes, bc_values, r_lim, theta_lim, bc_switch=1, num_freq=4, embed_dim=64):
        super().__init__()
        self.r_lim = r_lim
        self.theta_lim = theta_lim
        self.bc_switch = bc_switch
        self.num_freq = num_freq
        self.bc_values = bc_values

        self.r_encoder = nn.Sequential(
            nn.Linear(2 * num_freq, 32),
            nn.Tanh(),
            nn.Linear(32, embed_dim),
            nn.Tanh(),
        )
        self.theta_encoder = nn.Sequential(
            nn.Linear(2 * num_freq, 32),
            nn.Tanh(),
            nn.Linear(32, embed_dim),
            nn.Tanh(),
        )

        base_width = layer_sizes[2]
        self.u_proj = nn.Sequential(nn.Linear(2 * embed_dim, base_width), nn.Tanh())
        self.v_proj = nn.Sequential(nn.Linear(2 * embed_dim, base_width), nn.Tanh())
        self.x_proj = nn.Sequential(nn.Linear(2 * embed_dim, base_width), nn.Tanh())

        self.mix_layers = nn.ModuleList()
        self.u_adapters = nn.ModuleList()
        self.v_adapters = nn.ModuleList()
        current_width = base_width
        for width in layer_sizes[2:-1]:
            layer = nn.ModuleDict({
                "gate": nn.Sequential(nn.Linear(current_width, width), nn.Sigmoid()),
                "x": nn.Sequential(nn.Linear(current_width, width), nn.Tanh()) if current_width != width else nn.Identity(),
            })
            self.mix_layers.append(layer)
            self.u_adapters.append(nn.Sequential(nn.Linear(base_width, width), nn.Tanh()) if base_width != width else nn.Identity())
            self.v_adapters.append(nn.Sequential(nn.Linear(base_width, width), nn.Tanh()) if base_width != width else nn.Identity())
            current_width = width

        final_width = layer_sizes[-2]
        self.p_head = nn.Linear(final_width, 1)
        self.gamma_head = nn.Linear(final_width, 1)
        self.g_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def _normalize_r(self, r):
        return 2.0 * (r - self.r_lim[0]) / (self.r_lim[1] - self.r_lim[0]) - 1.0

    def _normalize_theta(self, theta):
        return 2.0 * (theta - self.theta_lim[0]) / (self.theta_lim[1] - self.theta_lim[0]) - 1.0

    def _fourier_features(self, x):
        feats = []
        for i in range(self.num_freq):
            freq = float((2 ** i) * torch.pi)
            feats.append(torch.sin(freq * x))
            feats.append(torch.cos(freq * x))
        return torch.cat(feats, dim=1)

    def forward(self, inputs):
        r = inputs[:, 0:1]
        theta = inputs[:, 1:2]
        r_norm = self._normalize_r(r)
        theta_norm = self._normalize_theta(theta)

        r_embed = self.r_encoder(self._fourier_features(r_norm))
        theta_embed = self.theta_encoder(self._fourier_features(theta_norm))
        x = torch.cat([r_embed, theta_embed], dim=1)

        x_u = self.u_proj(x)
        x_v = self.v_proj(x)
        x = self.x_proj(x)

        for layer, u_adapter, v_adapter in zip(self.mix_layers, self.u_adapters, self.v_adapters):
            x = layer["x"](x)
            gate = layer["gate"](x)
            x_u_cur = u_adapter(x_u)
            x_v_cur = v_adapter(x_v)
            x = gate * x_u_cur + (1.0 - gate) * x_v_cur

        nn_p = self.p_head(x)
        nn_gamma = self.gamma_head(x)

        if self.bc_switch == 1:
            g_func = self.g_net(r_norm)
            transition = 0.03
            t_left = torch.clamp((r_norm + 1.0) / transition, 0.0, 1.0)
            t_right = torch.clamp((1.0 - r_norm) / transition, 0.0, 1.0)
            sigma = (3.0 * t_left ** 2 - 2.0 * t_left ** 3) * (3.0 * t_right ** 2 - 2.0 * t_right ** 3)
            p_raw = g_func + sigma * nn_p
            gamma_raw = nn_gamma
        elif self.bc_switch == 2:
            p_raw = nn_p
            gamma_raw = nn_gamma
        else:
            raise ValueError(f"bc_switch must be 1 or 2, got {self.bc_switch}")

        p = torch.tanh(p_raw) ** 2
        gamma = torch.tanh(gamma_raw) ** 2
        return p, gamma
