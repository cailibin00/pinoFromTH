import copy
import os
import numpy as np
import torch

from .networks import FourierDecoupledPINN
from .pcgrad import pcgrad
from .utils import mse, to_torch


class TorchCollocationSolver:
    """PyTorch PINN solver, replacing CollocationSolverND from tensordiffeq.

    Designed for the Reynolds equation with JFO cavitation model using
    FourierDecoupledPINN architecture.
    """

    def __init__(self, device=None, verbose=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.loss_history = []
        self.loss_all_history = []
        self.epoch_history = [0]
        self.loss_value_min = float("inf")
        self.best_state_dict = None
        self.best_weights_path = None

    def compile(self, layer_sizes, f_model_list, domain, bcs,
                u_model_switch=13, two_output=True,
                none_zero=False, adapt_True=False,
                isAdaptive=False, MTL_adapt=False, PCGrad_true=True,
                Boundary_true=False,
                R_range=None, theta_range=None,
                bc_switch=1, num_freq=4, embed_dim=64):
        """Compile the solver with model architecture and PDE configuration.

        Args:
            layer_sizes: List of layer sizes for the neural network.
            f_model_list: List of PDE residual functions.
            domain: DomainND object with collocation points.
            bcs: List of boundary condition objects.
            u_model_switch: Network architecture selector (only 13 is supported).
            two_output: Whether the network outputs two values (P, gamma).
            PCGrad_true: Enable PCGrad gradient projection.
            Boundary_true: Whether to include boundary condition losses.
            R_range, theta_range: Domain ranges for coordinate normalization.
            bc_switch: BC handling mode (1=hard BC with g_net, 2=soft BC).
            num_freq: Number of Fourier feature frequencies.
            embed_dim: Embedding dimension for R/theta encoders.
        """
        if u_model_switch != 13:
            raise NotImplementedError(
                f"u_model_switch={u_model_switch} is not supported in torch_pinn. "
                f"Only switch=13 (FourierDecoupledPINN) is implemented."
            )

        self.layer_sizes = layer_sizes
        self.f_model_list = f_model_list
        self.domain = domain
        self.bcs = bcs
        self.two_output = two_output
        self.none_zero = none_zero
        self.PCGrad_true = PCGrad_true
        self.Boundary_true = Boundary_true
        self.R_range = R_range
        self.theta_range = theta_range
        self.bc_switch = bc_switch
        self.num_freq = num_freq
        self.embed_dim = embed_dim

        # Extract domain points
        self.domain_X_f = np.asarray(self.domain.X_f, dtype=np.float32)
        self.N_f_true = len(self.domain_X_f)

        # Build the neural network
        self.u_model = FourierDecoupledPINN(
            layer_sizes,
            self._extract_bc_values(),
            R_range,
            theta_range,
            bc_switch=bc_switch,
            num_freq=num_freq,
            embed_dim=embed_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.u_model.parameters(), lr=1e-3, betas=(0.99, 0.999)
        )
        self.current_lr = 1e-3

        # Additional model references (for RAD_FB, etc.)
        self.f_model_FB = None
        self.set_w_wedge = None

        # For epoch tracking across multiple fit() calls
        self._epoch_base = 0

    def _extract_bc_values(self):
        """Extract boundary values from the BC list."""
        bc_values = []
        for bc in self.bcs:
            if hasattr(bc, 'isDirichlect') and bc.isDirichlect:
                bc_values.append(bc.val)
        # Return [lower_val, upper_val] as expected by FourierDecoupledPINN
        if len(bc_values) >= 2:
            return bc_values[:2]
        elif len(bc_values) == 1:
            return [bc_values[0], bc_values[0]]
        return [0.0, 0.0]

    def set_learning_rate(self, lr):
        """Update the optimizer learning rate."""
        self.current_lr = lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _domain_tensors(self, requires_grad=True):
        """Create tensors for domain collocation points with gradient tracking."""
        x = to_torch(self.domain_X_f, self.device)
        r = x[:, 0:1].clone().detach().requires_grad_(requires_grad)
        theta = x[:, 1:2].clone().detach().requires_grad_(requires_grad)
        return r, theta

    def compute_losses(self):
        """Compute all loss terms: residuals + boundary conditions.

        Returns:
            List of loss tensors, one per term.
        """
        r, theta = self._domain_tensors(requires_grad=True)
        losses = []

        # Residual losses from f_model_list
        for f_model in self.f_model_list:
            residuals = f_model(self.u_model, r, theta)
            if not isinstance(residuals, (list, tuple)):
                residuals = [residuals]
            for res in residuals:
                losses.append(mse(res, torch.zeros_like(res)))

        # Boundary condition losses
        if self.Boundary_true:
            for bc in self.bcs:
                if hasattr(bc, 'isDirichlect') and bc.isDirichlect:
                    bc_input = to_torch(bc.input, self.device)
                    pred = self.u_model(bc_input)
                    if self.two_output:
                        p_pred = pred[0]
                    else:
                        p_pred = pred[0] if isinstance(pred, tuple) else pred
                    target = to_torch(bc.val, self.device)
                    losses.append(mse(p_pred.reshape(-1, 1), target.reshape(-1, 1)))

        # JFO interaction term (complementarity condition)
        if self.two_output:
            x = to_torch(self.domain_X_f, self.device)
            p, gamma = self.u_model(x)
            p = p.reshape(-1, 1)
            gamma = gamma.reshape(-1, 1)
            # Fischer-Burmeister: P + gamma - sqrt(P^2 + gamma^2) = 0
            loss_interact = mse(
                p + gamma - torch.sqrt(p ** 2 + gamma ** 2),
                torch.zeros_like(p)
            )
            losses.append(loss_interact)

        return losses

    def train_step(self):
        """Execute a single training step with PCGrad.

        Returns:
            (total_loss_tensor, list_of_loss_tensors)
        """
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.compute_losses()

        if len(losses) == 0:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return total_loss.detach(), []

        params = [p for p in self.u_model.parameters() if p.requires_grad]

        if self.PCGrad_true and len(losses) > 1:
            proj_grads = pcgrad(losses, params)
            for param, grad_value in zip(params, proj_grads):
                param.grad = grad_value
        else:
            total_loss = sum(losses)
            total_loss.backward()

        self.optimizer.step()

        total_loss = sum(losses).detach()
        return total_loss, [loss.detach() for loss in losses]

    def fit(self, tf_iter=0, newton_iter=0):
        """Run training for tf_iter iterations.

        Args:
            tf_iter: Number of Adam iterations.
            newton_iter: Number of L-BFGS iterations (not supported in torch_pinn).

        Returns:
            self
        """
        if newton_iter > 0:
            print("Warning: L-BFGS is not supported in torch_pinn. Skipping.")

        for epoch in range(tf_iter):
            loss_value, loss_all = self.train_step()
            global_epoch = self._epoch_base + epoch

            if epoch % 10 == 0:
                self.loss_history.append(float(loss_value.cpu().item()))
                self.epoch_history.append(global_epoch)
                self.loss_all_history.append(
                    [float(v.cpu().item()) for v in loss_all]
                )
                if self.verbose and epoch % 500 == 0:
                    loss_names = ['L_Reynolds', 'L_FB', 'L_BC', 'L_Interact']
                    parts = []
                    for i, name in enumerate(loss_names):
                        if i < len(loss_all):
                            parts.append(f'{name}={loss_all[i].cpu().item():.3e}')
                    loss_str = ' | '.join(parts)
                    print(
                        f'  Epoch {global_epoch}: '
                        f'Total={loss_value.cpu().item():.3e} | {loss_str}'
                    )

            if epoch % 100 == 0:
                loss_scalar = float(loss_value.cpu().item())
                if loss_scalar < self.loss_value_min:
                    self.loss_value_min = loss_scalar
                    self.best_state_dict = copy.deepcopy(self.u_model.state_dict())
                    if self.best_weights_path is not None:
                        self.save_weights(self.best_weights_path)

        self._epoch_base += tf_iter
        return self

    def RAD_FB(self, f_model_list, N_raw, num_add_points_test=2500,
               num_add_points=None, k=1.0, c=1.0):
        """Residual Adaptive Distribution for collocation point refinement.

        Adds new collocation points in regions with high PDE residual,
        matching the TF version's RAD_FB logic.

        Args:
            f_model_list: List of PDE residual functions to evaluate.
            N_raw: Number of original points to keep.
            num_add_points_test: Number of test points for residual evaluation.
            num_add_points: List of points to add per residual model.
            k: Power for residual weighting.
            c: Baseline constant for sampling probability.
        """
        if num_add_points is None:
            num_add_points = [50]

        # Truncate to original points
        self.domain.X_f = self.domain.X_f[0:N_raw, :]

        # Generate test points for residual evaluation
        X_f_in = self.domain.generate_collocation_points_old(num_add_points_test)

        for i, f_model in enumerate(f_model_list):
            if f_model is None:
                continue

            # Evaluate residual on test points
            # NOTE: requires_grad=True needed because PDE residual uses autograd
            r_test = to_torch(X_f_in[:, 0:1], self.device, requires_grad=True)
            theta_test = to_torch(X_f_in[:, 1:2], self.device, requires_grad=True)

            f_u_pred = f_model(self.u_model, r_test, theta_test)

            f_u_pred_np = f_u_pred.detach().cpu().numpy().flatten()

            # Remove NaN values
            non_nan_mask = ~np.isnan(f_u_pred_np)
            f_u_pred_nonnan = f_u_pred_np[non_nan_mask]
            X_f_in_nonnan = X_f_in[non_nan_mask, :]

            if len(f_u_pred_nonnan) == 0:
                continue

            # Compute squared error and sampling weights
            err_eq = np.power(f_u_pred_nonnan ** 2, k) / (
                np.power(f_u_pred_nonnan ** 2, k).mean() + 1e-8
            ) + c
            err_eq[np.isnan(err_eq)] = 0
            err_eq_normalized = err_eq / err_eq.sum()

            n_add = num_add_points[i] if i < len(num_add_points) else num_add_points[-1]
            n_add = min(n_add, len(X_f_in_nonnan))

            # Sample points based on residual magnitude
            X_ids = np.random.choice(
                a=len(X_f_in_nonnan), size=n_add, replace=False,
                p=err_eq_normalized
            )
            X_f_in_add = X_f_in_nonnan[X_ids, :]

            # Append new points to domain
            self.domain.X_f = np.concatenate((self.domain.X_f, X_f_in_add), axis=0)

        # Update internal domain representation
        self.domain_X_f = np.asarray(self.domain.X_f, dtype=np.float32)
        self.N_f_true = len(self.domain_X_f)

        return self

    def predict(self, X_star):
        """Run inference on input coordinates.

        Args:
            X_star: numpy array of shape (N, 2), columns are (R, theta).

        Returns:
            (p_numpy, gamma_numpy): Tuple of numpy arrays.
        """
        coords = to_torch(X_star, self.device)
        self.u_model.eval()
        with torch.no_grad():
            p, gamma = self.u_model(coords)
        self.u_model.train()
        return p.cpu().numpy(), gamma.cpu().numpy()

    def save_weights(self, path):
        """Save model state dict to path."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'state_dict': self.u_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'loss_all_history': self.loss_all_history,
            'epoch_history': self.epoch_history,
            'loss_value_min': self.loss_value_min,
        }, path)

    def load_weights(self, path, map_location=None):
        """Load model state dict from path.

        Returns:
            bool: True if best checkpoint was restored, False otherwise.
        """
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.u_model.load_state_dict(checkpoint.get('state_dict', checkpoint))
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        if 'loss_all_history' in checkpoint:
            self.loss_all_history = checkpoint['loss_all_history']
        if 'epoch_history' in checkpoint:
            self.epoch_history = checkpoint['epoch_history']
        if 'loss_value_min' in checkpoint:
            self.loss_value_min = checkpoint['loss_value_min']
        return True

    def save(self, model_path):
        """Save full model (state dict + config)."""
        os.makedirs(model_path, exist_ok=True)
        payload = {
            'state_dict': self.u_model.state_dict(),
            'layer_sizes': self.layer_sizes,
            'bc_values': self._extract_bc_values(),
            'R_range': self.R_range,
            'theta_range': self.theta_range,
            'bc_switch': self.bc_switch,
            'num_freq': self.num_freq,
            'embed_dim': self.embed_dim,
        }
        torch.save(payload, os.path.join(model_path, "model.pt"))

    def load_model(self, model_path):
        """Load a saved model."""
        payload = torch.load(
            os.path.join(model_path, "model.pt"),
            map_location=self.device
        )
        self.layer_sizes = payload['layer_sizes']
        bc_values = payload['bc_values']
        self.R_range = payload['R_range']
        self.theta_range = payload['theta_range']
        self.bc_switch = payload.get('bc_switch', 1)
        self.num_freq = payload.get('num_freq', 4)
        self.embed_dim = payload.get('embed_dim', 64)

        self.u_model = FourierDecoupledPINN(
            self.layer_sizes,
            bc_values,
            self.R_range,
            self.theta_range,
            bc_switch=self.bc_switch,
            num_freq=self.num_freq,
            embed_dim=self.embed_dim,
        ).to(self.device)
        self.u_model.load_state_dict(payload['state_dict'])
        self.u_model.eval()
        return self

    def forward_on_numpy(self, X_star):
        """Forward pass returning tensors (for compatibility)."""
        p, gamma = self.predict(X_star)
        return [torch.from_numpy(p), torch.from_numpy(gamma)]
