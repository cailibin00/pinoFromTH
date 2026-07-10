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
import torch.nn as nn
from .networks import new_neural_period_polar_exactBC_two_output
from .utils import mse, to_torch, latin_hypercube_sample
from .pcgrad import pcgrad


# =============================================================================
# ComputeSum_weight - Adaptive Loss Weighting
# =============================================================================
class ComputeSum_weight:
    """
    Adaptive loss weighting via exponential smoothing of gradient ratios.

    Implements the TF ComputeSum_weight layer:
    - w_new = (1 - alpha) * w_old + alpha * grad_ratio
    - Tracks both instantaneous weights and step-averaged weights
    """

    def __init__(self, input_dim, adaptive_constant_alpha, adaptive_constant_step_alpha=100):
        self.input_dim = input_dim
        self.alpha = adaptive_constant_alpha
        self.step_alpha = adaptive_constant_step_alpha
        # adaptive_constant: exponentially smoothed weights
        self.adaptive_constant = torch.ones(1, input_dim)
        # adaptive_constant_step: step-averaged reference
        self.adaptive_constant_step = torch.ones(1, input_dim)
        self.count = self.step_alpha

    def update(self, adaptive_constant_new):
        """Update weights with new gradient ratios (matching TF call)."""
        if isinstance(adaptive_constant_new, list):
            adaptive_constant_new = torch.tensor(
                [[l.item() if isinstance(l, torch.Tensor) else l for l in adaptive_constant_new]],
                dtype=torch.float32
            )
        if adaptive_constant_new.dim() == 1:
            adaptive_constant_new = adaptive_constant_new.unsqueeze(0)

        self.adaptive_constant = (
            (1.0 - self.alpha) * self.adaptive_constant +
            self.alpha * adaptive_constant_new
        )

        self.count += 1
        if self.count > self.step_alpha:
            self.adaptive_constant_step = self.adaptive_constant.clone()
            self.count = 0

    def get_weights(self):
        """Return current adaptive weights."""
        return self.adaptive_constant


# =============================================================================
# CollocationSolverND - Core PINN Solver
# =============================================================================
class CollocationSolverND:
    """
    Core PINN solver for Reynolds equation with JFO cavitation.

    Manages:
    - Model compilation and training
    - Loss computation (residual + BC + JFO interaction)
    - PCGrad gradient projection
    - Adaptive loss weighting
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
        self.PCGrad_true = False
        self.Boundary_true = True
        self.two_output = False
        self.none_zero = False
        self.adapt_True = False
        self.MTL_adapt = False
        self.balance = False

        # Extra models (set externally)
        self.f_model_FB = None
        self.f_model_list = []

    # =========================================================================
    # Compile
    # =========================================================================
    def compile(self, layer_sizes, f_model_list, domain, bcs,
                u_model_switch=1, two_output=False, none_zero=False,
                adapt_True=False, isAdaptive=False, MTL_adapt=False,
                PCGrad_true=False, Boundary_true=True,
                R_range=None, theta_range=None, batch_size=None):
        """
        Configure the solver.

        Args:
            ...
            batch_size: minibatch size for collocation points; None = full-batch
        """
        self.layer_sizes = layer_sizes
        self.bcs = bcs
        self.domain = domain
        self.PCGrad_true = PCGrad_true
        self.Boundary_true = Boundary_true
        self.two_output = two_output
        self.none_zero = none_zero
        self.adapt_True = adapt_True
        self.MTL_adapt = MTL_adapt
        self.R_range = R_range if R_range else []
        self.theta_range = theta_range if theta_range else []

        # Build model based on switch
        self.u_model_switch = u_model_switch
        if u_model_switch == 8:
            bc_values = [bcs[0].val, bcs[1].val]
            self.u_model = new_neural_period_polar_exactBC_two_output(
                layer_sizes, bc_values, self.R_range, self.theta_range
            )
        else:
            raise ValueError(f"Unsupported u_model_switch={u_model_switch}")
        self.u_model.to(self.device)

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

        # Adaptive loss weighting (matching TF)
        n_loss_terms = len(self._get_loss_list())
        self.adaptive_constant_alpha = 0.2
        self.adaptive_constant_func = ComputeSum_weight(n_loss_terms, self.adaptive_constant_alpha)

        # PCGrad loss weighting (matching TF)
        self.adaptive_constant_alpha_PCGrad_loss = 0.1
        self.adaptive_constant_func_PCGrad_loss = ComputeSum_weight(
            n_loss_terms, self.adaptive_constant_alpha_PCGrad_loss
        )

        self.adaptive_constant_func_list = []

        # MTL_adapt parameters
        if self.MTL_adapt:
            self.MTL_adapt_par = nn.Parameter(torch.ones(n_loss_terms, device=self.device))
            self.MTL_adapt_list = []

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

        # Batch settings
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
        # We have: len(f_model_list) residuals + (1 BC if Boundary_true) + (1 JFO if two_output)
        n = len(self.f_model_list)
        if self.Boundary_true:
            n += 1
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
        Compute all loss terms separately (for PCGrad).
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
        if self.Boundary_true:
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
        """Compute total weighted loss (for L-BFGS)."""
        loss_all = self.update_loss_seperate()
        w = self.adaptive_constant_func.adaptive_constant.to(self.device)

        loss_total = w[0, 0] * loss_all[0]
        for i in range(1, len(loss_all)):
            loss_total = loss_total + w[0, min(i, w.shape[1] - 1)] * loss_all[i]

        return loss_total, loss_all

    # =========================================================================
    # Gradient Computation with PCGrad + Adaptive Weighting
    # =========================================================================
    def grad_separate_all_with_adapt_weight(self):
        """
        Compute gradients with PCGrad projection and adaptive weighting.
        This is the core gradient function matching TF's implementation.
        """
        loss_all = self.update_loss_seperate()

        # Total loss (for standard gradient or as base for PCGrad)
        if self.MTL_adapt:
            # Log-variance weighting: sum(exp(-w_i) * loss_i + w_i)
            loss_total = sum(
                0.5 * torch.exp(-self.MTL_adapt_par[i]) * loss_all[i] + self.MTL_adapt_par[i]
                for i in range(len(loss_all))
            )
        else:
            w = self.adaptive_constant_func.adaptive_constant.to(self.device)
            loss_total = w[0, 0] * loss_all[0]
            for i in range(1, len(loss_all)):
                loss_total = loss_total + w[0, min(i, w.shape[1] - 1)] * loss_all[i]

        # Adaptive weight update (matching TF adapt_True logic)
        if self.adapt_True:
            self._update_adaptive_weights(loss_all)

        # PCGrad gradient projection
        if self.PCGrad_true:
            grads, _ = pcgrad(
                loss_all,
                list(self.u_model.parameters()),
                self.adaptive_constant_func_PCGrad_loss,
                self.balance
            )
            return loss_total, grads, loss_all
        else:
            grads = torch.autograd.grad(
                loss_total, self.u_model.parameters(),
                retain_graph=True, create_graph=False
            )
            grads = [g if g is not None else torch.zeros_like(p)
                     for g, p in zip(grads, self.u_model.parameters())]
            return loss_total, grads, loss_all

    def _update_adaptive_weights(self, loss_all):
        """Update adaptive weights based on gradient magnitude ratios (matching TF)."""
        params = list(self.u_model.parameters())

        # Compute per-loss gradients
        grad_all = []
        for loss in loss_all:
            g = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
            g_clean = [gi if gi is not None else torch.zeros_like(p) for gi, p in zip(g, params)]
            grad_all.append(g_clean)

        # Extract weight (kernel) gradients only
        grad_all_w = []
        for grads in grad_all:
            w_grads = [g for name, g in zip(
                [n for n, _ in self.u_model.named_parameters()], grads
            ) if 'weight' in name]
            grad_all_w.append(w_grads)

        if len(grad_all_w) < 2:
            return

        # Residual gradient mean magnitude
        grad_res_num = sum(g.numel() for g in grad_all_w[0])
        grad_res_sum = sum(torch.sum(torch.abs(g)) for g in grad_all_w[0])
        grads_res_mean = grad_res_sum / (grad_res_num + 1e-12)

        # BC/interaction gradient mean magnitudes
        grad_w_nums = [sum(g.numel() for g in gw) for gw in grad_all_w]
        grad_w_sums = [sum(torch.sum(torch.abs(g)) for g in gw) for gw in grad_all_w]
        grads_mean_list = [
            s / (n + 1e-12) for s, n in zip(grad_w_sums, grad_w_nums)
        ]

        # Compute new weight ratios
        adaptive_constant_new = [grads_res_mean / (grads_mean_list[i] + 1e-12) for i in range(len(grads_mean_list))]
        adaptive_constant_new = [torch.clamp(w, 1e-2, 1e12) for w in adaptive_constant_new]
        self.adaptive_constant_func.update(adaptive_constant_new)

    # =========================================================================
    # Training
    # =========================================================================
    def train_step(self):
        """Single Adam training step with PCGrad."""
        self.u_model.train()

        loss_total, grads, loss_all = self.grad_separate_all_with_adapt_weight()

        # Apply gradients
        self.tf_optimizer.zero_grad()
        for param, grad in zip(self.u_model.parameters(), grads):
            param.grad = grad
        self.tf_optimizer.step()

        return loss_total.detach(), [l.detach() for l in loss_all]

    def fit(self, tf_iter=0, newton_iter=0, batch_sz=None, newton_eager=True):
        """
        Main training entry point (matching TF's CollocationSolverND.fit).

        Args:
            tf_iter: number of Adam iterations
            newton_iter: number of L-BFGS iterations (0 to skip)
            batch_sz: batch size for minibatching
            newton_eager: whether to use eager L-BFGS
        """
        from .fit import fit as fit_func
        fit_func(self, tf_iter=tf_iter, newton_iter=newton_iter, newton_eager=newton_eager)

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
        }, save_path)

    def load_model(self, path, compile_model=False):
        """Load full model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.layer_sizes = checkpoint['layer_sizes']
        self.R_range = checkpoint['R_range']
        self.theta_range = checkpoint['theta_range']
        bc_values = checkpoint['bc_values']
        self.u_model = new_neural_period_polar_exactBC_two_output(
            self.layer_sizes, bc_values, self.R_range, self.theta_range
        )
        self.u_model.load_state_dict(checkpoint['model_state_dict'])
        self.u_model.to(self.device)
