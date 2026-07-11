"""
CollocationSolverND - Core PINN solver.
Faithful 1:1 port from tensordiffeq/models.py (TensorFlow).

This is the heart of the PINN framework:
- Manages NN model, domain, BCs, PDE residuals
- Handles loss computation (residual + BC + JFO + non-negative)
- Implements PCGrad with adaptive loss weighting
- Supports RAD (Residual-based Adaptive Distribution) for point refinement
- Two-phase training: Adam + L-BFGS
"""

import os
import numpy as np
import torch
from .networks import (
    new_neural_period_polar_exactBC_two_output,
    PIKAN_Polar_BC_Two_Output,
    count_model_params,
    auto_pikan_layer_sizes,
)
from .utils import mse, to_torch, latin_hypercube_sample
# =============================================================================
# CollocationSolverND - Core PINN Solver
# =============================================================================
class CollocationSolverND:
    """
    Core PINN solver for Reynolds equation with JFO cavitation.

    Manages:
    - Model compilation and training
    - Loss computation (residual + BC + JFO interaction)
    - RAD (Residual-based Adaptive Distribution) refinement
    """

    def __init__(self, assimilate=False, verbose=True, device='auto'):
        self.assimilate = assimilate
        self.verbose = verbose
        # Resolve device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Loss tracking
        self.loss_history = []
        self.loss_all_history = []
        self.epoch_history = [0]

        # State
        self.loss_value_min = 1e12
        self.best_state_dict = None
        self.best_weights_path = 'epochs_best_model.pt'

        # Config flags
        self.two_output = False
        self.none_zero = False

        # Extra models (set externally)
        self.f_model_FB = None
        self.f_model_list = []

    # =========================================================================
    # Compile
    # =========================================================================
    def compile(self, layer_sizes, f_model_list, domain, bcs,
                u_model_switch=1, two_output=False, none_zero=False,
                R_range=None, theta_range=None, batch_size=None,
                core='mlp', kan_grid_size=5, kan_spline_order=3,
                output_head_dim=64):
        """
        Configure the solver.

        Args:
            ...
            batch_size: minibatch size for collocation points; None = full-batch
            core: 'mlp' (standard MLP) or 'pikan' (KAN-based architecture)
            kan_grid_size: B‑spline grid intervals (PIKAN only)
            kan_spline_order: B‑spline polynomial order (PIKAN only)
            output_head_dim: hidden dim inside deep output heads
        """
        self.layer_sizes = layer_sizes
        self.bcs = bcs
        self.domain = domain
        self.two_output = two_output
        self.none_zero = none_zero
        self.R_range = R_range if R_range else []
        self.theta_range = theta_range if theta_range else []
        self.core = core
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order

        # Build model based on switch AND core type
        self.u_model_switch = u_model_switch
        if u_model_switch == 8:
            bc_values = [bcs[0].val, bcs[1].val]

            if core == 'pikan':
                # Auto‑compute PIKAN sizes if the user hasn't manually set them.
                # We detect this by checking if layer_sizes look like MLP sizes
                # (cos_units ≠ hidden_width, or >5 layers) → auto-tune.
                hidden_w = layer_sizes[2:-1]
                if len(set(hidden_w)) > 1 or len(hidden_w) > 4:
                    # MLP-style sizes with varying widths → auto-compute
                    pikan_sizes = auto_pikan_layer_sizes(
                        layer_sizes, kan_grid_size, kan_spline_order
                    )
                else:
                    pikan_sizes = layer_sizes  # user provided explicit PIKAN sizes

                self.layer_sizes = pikan_sizes  # store actual sizes used

                self.u_model = PIKAN_Polar_BC_Two_Output(
                    pikan_sizes, bc_values, self.R_range, self.theta_range,
                    kan_grid_size=kan_grid_size,
                    kan_spline_order=kan_spline_order,
                    output_head_dim=output_head_dim,
                )

                # Print parameter summary
                trainable, total = count_model_params(self.u_model)
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"PIKAN model created (G={kan_grid_size}, K={kan_spline_order})")
                    print(f"Layer sizes: {pikan_sizes}")
                    print(self.u_model.param_summary())
                    print(f"{'='*60}\n")
            else:
                self.u_model = new_neural_period_polar_exactBC_two_output(
                    layer_sizes, bc_values, self.R_range, self.theta_range,
                    output_head_dim=output_head_dim,
                )

                trainable, total = count_model_params(self.u_model)
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"MLP model created")
                    print(f"Layer sizes: {layer_sizes}")
                    print(self.u_model.param_summary())
                    print(f"{'='*60}\n")
        else:
            raise ValueError(f"Unsupported u_model_switch={u_model_switch}")
        self.u_model.to(self.device)

        # Store param counts for later reference
        self.model_param_count = trainable

        # Store PDE residual functions
        self.f_model_list = f_model_list

        # Extract collocation points
        self.domain_X_f = torch.tensor(domain.X_f, dtype=torch.float32, device=self.device)
        self.X_f_len = len(domain.X_f)

        # Optimizer (Adam, matching TF: lr=0.001, betas=(0.99, 0.999))
        self.tf_optimizer = torch.optim.Adam(
            self.u_model.parameters(),
            lr=0.001,
            betas=(0.99, 0.999)
        )

        # Weight sizes for L-BFGS
        sizes_w, sizes_b = [], []
        for name, param in self.u_model.named_parameters():
            if 'weight' in name:
                sizes_w.append(param.numel())
            elif 'bias' in name:
                sizes_b.append(param.numel())
        self.sizes_w = sizes_w
        self.sizes_b = sizes_b

        # Save initial weights
        self._save_best()

        # Store batch settings
        self.batch_size = batch_size
        self.n_batches = 1

    # =========================================================================
    # Loss Functions
    # =========================================================================
    def _get_batch_X_f(self):
        """Return collocation points tensor, optionally subsampled as a minibatch."""
        if self.batch_size is not None and self.batch_size < self.X_f_len:
            idx = torch.randperm(self.X_f_len, device=self.device)[:self.batch_size]
            return self.domain_X_f[idx]
        return self.domain_X_f

    def _get_loss_list(self):
        """
        Get a representative loss list to determine n_loss_terms.
        Must match update_loss_seperate() structure.
        """
        # We have: len(f_model_list) residuals + 1 BC + (1 JFO if two_output)
        n = len(self.f_model_list) + 1
        if self.two_output:
            n += 1
        if self.none_zero:
            n += 1
        return list(range(max(n, 2)))

    def update_loss_res(self, f_model, R, theta):
        """Evaluate one PDE residual model and return MSE losses."""
        f_u_preds = f_model(self.u_model, R, theta)

        if not isinstance(f_u_preds, tuple):
            f_u_preds = (f_u_preds,)

        loss_res = []
        for f_u_pred in f_u_preds:
            loss_r = mse(f_u_pred, torch.zeros_like(f_u_pred))
            loss_res.append(loss_r)

        return loss_res

    def update_loss_bcs(self):
        """Evaluate all boundary condition losses."""
        loss_bcs = []
        for bc in self.bcs:
            if bc.isDirichlet:
                bc_input = torch.tensor(bc.input, dtype=torch.float32, device=self.device)
                bc_val = torch.tensor(np.reshape(bc.val, (-1, 1)), dtype=torch.float32, device=self.device)

                if self.two_output:
                    out = self.u_model(bc_input)
                    loss_bc = mse(out[0], bc_val)
                else:
                    out = self.u_model(bc_input)
                    if isinstance(out, list):
                        out = out[0]
                    loss_bc = mse(out, bc_val)
                loss_bcs.append(loss_bc)
            elif bc.isPeriodic:
                # Periodic BC: not used in current training setup
                pass
            elif bc.isInit:
                # Initial condition: not used
                pass
            elif bc.isNeumann:
                # Neumann BC: not used
                pass

        # TF version: returns [mean(loss_bcs)]
        if loss_bcs:
            return [torch.mean(torch.stack(loss_bcs))]
        return []

    def update_loss_JFO_term_interact(self):
        """Fischer-Burmeister complementarity: P + gamma - sqrt(P^2 + gamma^2) = 0."""
        X_f_batch = self._get_batch_X_f()
        u_preds = self.u_model(X_f_batch)
        if not isinstance(u_preds, list):
            u_preds = [u_preds]

        loss_g_all = torch.tensor(0.0, device=self.device)
        for u_pred in u_preds:
            p, gamma = u_pred[0], u_pred[1]
            loss_g = mse(
                p + gamma - torch.sqrt(p ** 2 + gamma ** 2),
                torch.zeros_like(p)
            )
            loss_g_all = loss_g_all + loss_g

        return loss_g_all

    def update_loss_0_boundary(self):
        """Non-negative penalty: (|u| * (-u) + u^2) / 2."""
        X_f_batch = self._get_batch_X_f()
        u_preds = self.u_model(X_f_batch)
        if not isinstance(u_preds, list):
            u_preds = [u_preds]

        loss_zero_all = torch.tensor(0.0, device=self.device)
        for u_pred in u_preds:
            loss_zero = torch.mean(
                (torch.abs(u_pred) * (-u_pred) + u_pred ** 2) / 2.0
            )
            loss_zero_all = loss_zero_all + loss_zero

        return loss_zero_all

    def update_loss_seperate(self):
        """
        Compute all loss terms separately.
        Returns list: [res_0, ..., res_n, bc_loss(es), jfo_loss, ...]
        Supports minibatch via self.batch_size.
        """
        loss_all = []

        # PDE Residual losses (one per f_model) — use minibatch
        X_f_batch = self._get_batch_X_f()
        R = X_f_batch[:, 0:1].requires_grad_(True)
        theta = X_f_batch[:, 1:2].requires_grad_(True)

        for f_model in self.f_model_list:
            loss_res = self.update_loss_res(f_model, R, theta)
            loss_all = loss_all + loss_res

        # Boundary condition losses
        loss_bcs = self.update_loss_bcs()
        loss_all = loss_all + loss_bcs

        # JFO interaction loss — use minibatch
        if self.two_output:
            loss_g_all = self.update_loss_JFO_term_interact()
            loss_all.append(loss_g_all)

        # Non-negative loss
        if self.none_zero:
            loss_zero_all = self.update_loss_0_boundary()
            loss_all.append(loss_zero_all)

        return loss_all

    def update_loss(self):
        """Compute total loss (for L-BFGS)."""
        loss_all = self.update_loss_seperate()
        loss_total = sum(loss_all)
        return loss_total, loss_all

    # =========================================================================
    # Training
    # =========================================================================
    def train_step(self):
        """Single Adam training step."""
        self.u_model.train()
        self.tf_optimizer.zero_grad()

        loss_all = self.update_loss_seperate()
        loss_total = sum(loss_all)

        loss_total.backward()
        self.tf_optimizer.step()

        return loss_total.detach(), [l.detach() for l in loss_all]

    def fit(self, tf_iter=0, newton_iter=0, batch_sz=None, newton_eager=True, scheduler=None):
        """
        Main training entry point.

        Args:
            tf_iter: number of Adam iterations
            newton_iter: number of L-BFGS iterations (0 to skip)
            batch_sz: batch size for minibatching
            newton_eager: whether to use eager L-BFGS
            scheduler: optional torch LR scheduler (stepped each epoch)
        """
        from .fit import fit as fit_func
        fit_func(self, tf_iter=tf_iter, newton_iter=newton_iter, newton_eager=newton_eager, scheduler=scheduler)

    def get_loss_and_flat_grad(self):
        """Return closure for L-BFGS (matching TF's get_loss_and_flat_grad)."""
        from .utils import get_weights_torch, set_weights_torch

        def loss_and_flat_grad(w_flat):
            # Set flat weights into model
            set_weights_torch(self.u_model, w_flat, self.sizes_w, self.sizes_b)

            # Compute loss
            self.u_model.train()
            loss_value, loss_all = self.update_loss()

            # Compute gradients
            grads = torch.autograd.grad(loss_value, self.u_model.parameters(),
                                        retain_graph=False, create_graph=False)
            grads = [g if g is not None else torch.zeros_like(p)
                     for g, p in zip(grads, self.u_model.parameters())]

            grad_flat = torch.cat([g.reshape(-1) for g in grads])
            return loss_value, grad_flat, loss_all

        return loss_and_flat_grad

    # =========================================================================
    # RAD (Residual-based Adaptive Distribution)
    # =========================================================================
    def RAD_FB(self, f_model_list, N_raw, num_add_points_test=2500,
               num_add_points=None, k=1.0, c=1.0):
        """
        Residual-based Adaptive Distribution refinement.
        Matching TF's RAD_FB exactly.

        Args:
            f_model_list: list of PDE residual functions
            N_raw: number of original points to keep
            num_add_points_test: test points to evaluate
            num_add_points: list of points to add per f_model
            k: power for residual weighting (TF uses k=1)
            c: constant offset in probability
        """
        if num_add_points is None:
            num_add_points = [50]

        # Trim to original points
        self.domain.X_f = self.domain.X_f[0:N_raw, :]

        # Generate test points
        X_f_in = self.domain.generate_collocation_points_old(num_add_points_test)
        X_star = torch.tensor(X_f_in, dtype=torch.float32, device=self.device)
        R = X_star[:, 0:1].requires_grad_(True)
        theta = X_star[:, 1:2].requires_grad_(True)

        for i, f_model in enumerate(f_model_list):
            f_u_pred = f_model(self.u_model, R, theta)

            # Convert to numpy
            if isinstance(f_u_pred, torch.Tensor):
                f_np = f_u_pred.detach().cpu().numpy().flatten()
            else:
                f_np = f_u_pred[0].detach().cpu().numpy().flatten()

            # Remove NaN
            nan_mask = ~np.isnan(f_np)
            f_np_clean = f_np[nan_mask]
            X_clean = X_f_in[nan_mask]

            if len(f_np_clean) == 0:
                continue

            # Square and compute probability
            f_sq = f_np_clean ** 2
            err_eq = np.power(f_sq, k) / (np.power(f_sq, k).mean() + 1e-8) + c
            err_eq[np.isnan(err_eq)] = 0

            err_eq_normalized = err_eq / (err_eq.sum() + 1e-12)

            # Sample new points
            n_add = num_add_points[min(i, len(num_add_points) - 1)]
            n_add = min(n_add, len(X_clean))
            X_ids = np.random.choice(len(X_clean), size=n_add, replace=False, p=err_eq_normalized)
            X_add = X_clean[X_ids]

            # Append to domain
            self.domain.X_f = np.concatenate([self.domain.X_f, X_add], axis=0)

        # Update domain tensors
        self.domain_X_f = torch.tensor(self.domain.X_f, dtype=torch.float32, device=self.device)
        self.X_f_len = len(self.domain.X_f)

    # =========================================================================
    # RAR (Residual Adaptive Refinement)
    # =========================================================================
    def RAR(self, f_model, num_add_points_test=2000, c=0.8):
        """Residual Adaptive Refinement (matching TF's RAR)."""
        X_star = self.domain_X_f

        f_u_pred = f_model(self.u_model, X_star[:, 0:1], X_star[:, 1:2])
        f_np = f_u_pred.detach().cpu().numpy().flatten()

        X_f_in = self.domain.generate_collocation_points_old(num_add_points_test)
        f_test = f_model(
            self.u_model,
            torch.tensor(X_f_in[:, 0:1], dtype=torch.float32, device=self.device),
            torch.tensor(X_f_in[:, 1:2], dtype=torch.float32, device=self.device)
        )
        f_test_np = f_test.detach().cpu().numpy().flatten()

        # Remove NaN
        nan_mask = ~np.isnan(f_test_np)
        f_test_clean = f_test_np[nan_mask]
        X_clean = X_f_in[nan_mask]

        if len(f_test_clean) == 0:
            return

        f_sq = f_test_clean ** 2
        threshold = c * np.max(f_sq)
        mask = f_sq > threshold
        point_add = X_clean[mask]

        if len(point_add) > 0:
            self.domain.X_f = np.concatenate([self.domain.X_f, point_add], axis=0)
            self.domain_X_f = torch.tensor(self.domain.X_f, dtype=torch.float32, device=self.device)
            self.X_f_len = len(self.domain.X_f)

    # =========================================================================
    # Predict, Save, Load
    # =========================================================================
    def predict(self, X_star):
        """Inference on given coordinates."""
        self.u_model.eval()
        X_t = torch.tensor(X_star, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            u_pred = self.u_model(X_t)
        return [u.cpu().numpy() for u in u_pred]

    def _save_best(self):
        """Save current weights as best."""
        self.best_state_dict = {
            k: v.cpu().clone() for k, v in self.u_model.state_dict().items()
        }

    def save_weights(self, path):
        """Save model weights."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.u_model.state_dict(),
            'optimizer_state_dict': self.tf_optimizer.state_dict(),
            'loss_history': self.loss_history,
            'loss_all_history': self.loss_all_history,
            'epoch_history': self.epoch_history,
            'loss_value_min': self.loss_value_min,
        }, path)

    def load_weights(self, path):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.u_model.load_state_dict(checkpoint['model_state_dict'])
        self.loss_history = checkpoint.get('loss_history', [])
        self.loss_all_history = checkpoint.get('loss_all_history', [])
        self.epoch_history = checkpoint.get('epoch_history', [0])
        self.loss_value_min = checkpoint.get('loss_value_min', 1e12)
        self.best_state_dict = {
            k: v.cpu().clone() for k, v in self.u_model.state_dict().items()
        }

    def save(self, path):
        """Save full model."""
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, 'model.pt')
        torch.save({
            'model_state_dict': self.u_model.state_dict(),
            'layer_sizes': self.layer_sizes,
            'R_range': self.R_range,
            'theta_range': self.theta_range,
            'bc_values': [self.bcs[0].val, self.bcs[1].val],
            'core': getattr(self, 'core', 'mlp'),
            'kan_grid_size': getattr(self, 'kan_grid_size', 5),
            'kan_spline_order': getattr(self, 'kan_spline_order', 3),
            'output_head_dim': getattr(self, 'output_head_dim', 64),
        }, save_path)

    def load_model(self, path, compile_model=False):
        """Load full model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.layer_sizes = checkpoint['layer_sizes']
        self.R_range = checkpoint['R_range']
        self.theta_range = checkpoint['theta_range']
        bc_values = checkpoint['bc_values']
        core = checkpoint.get('core', 'mlp')
        kan_grid_size = checkpoint.get('kan_grid_size', 5)
        kan_spline_order = checkpoint.get('kan_spline_order', 3)
        output_head_dim = checkpoint.get('output_head_dim', 64)

        if core == 'pikan':
            self.u_model = PIKAN_Polar_BC_Two_Output(
                self.layer_sizes, bc_values, self.R_range, self.theta_range,
                kan_grid_size=kan_grid_size,
                kan_spline_order=kan_spline_order,
                output_head_dim=output_head_dim,
            )
        else:
            self.u_model = new_neural_period_polar_exactBC_two_output(
                self.layer_sizes, bc_values, self.R_range, self.theta_range,
                output_head_dim=output_head_dim,
            )
        self.u_model.load_state_dict(checkpoint['model_state_dict'])
        self.u_model.to(self.device)
        self.core = core
