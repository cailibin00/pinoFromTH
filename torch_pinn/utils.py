"""
Utility functions for torch_pinn.
Faithful port from tensordiffeq/utils.py (TensorFlow).
"""

import numpy as np
import torch
import sys
import os


# =============================================================================
# MSE
# =============================================================================
def mse(pred, actual, weights=None):
    """Mean squared error."""
    if weights is not None:
        return torch.mean(weights * (pred - actual) ** 2)
    return torch.mean((pred - actual) ** 2)


def find_L2_error(u_pred, u_star):
    """Relative L2 norm error."""
    u_pred = np.array(u_pred).flatten()
    u_star = np.array(u_star).flatten()
    return np.linalg.norm(u_star - u_pred) / np.linalg.norm(u_star)


# =============================================================================
# Latin Hypercube Sampling
# =============================================================================
def latin_hypercube_sample(n_samples, bounds):
    """
    Latin Hypercube Sampling.
    bounds: (n_dims, 2) array of [lower, upper] for each dimension.
    """
    n_dims = bounds.shape[0]
    rng = np.random.default_rng()
    # Generate samples
    samples = np.zeros((n_samples, n_dims))
    for i in range(n_dims):
        # Divide range into n_samples intervals
        cut = rng.permutation(n_samples) + rng.uniform(size=n_samples)
        samples[:, i] = bounds[i, 0] + cut * (bounds[i, 1] - bounds[i, 0]) / n_samples
    return samples


# =============================================================================
# Mesh utilities
# =============================================================================
def multimesh(arrs):
    """Create component arrays of a tensor-product mesh. Like np.meshgrid."""
    lens = list(map(len, arrs))
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz *= s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz_j in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz_j, axis=j)
        ans.append(arr2)

    return ans


def flatten_and_stack(mesh):
    """Flatten and hstack mesh outputs into [N, D] matrix."""
    dims = np.shape(mesh)
    output = np.zeros((len(mesh), np.prod(dims[1:])))
    for i, arr in enumerate(mesh):
        output[i] = arr.flatten()
    return output.T


# =============================================================================
# Weight flattening for L-BFGS
# =============================================================================
def get_sizes(model):
    """Get weight and bias sizes for each layer, matching TF logic."""
    sizes_w = []
    sizes_b = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            sizes_w.append(param.numel())
        elif 'bias' in name:
            sizes_b.append(param.numel())
    return sizes_w, sizes_b


def get_weights_torch(model):
    """Flatten all model weights into a 1D tensor (matching TF get_weights)."""
    w = []
    for name, param in model.named_parameters():
        w.append(param.data.flatten())
    return torch.cat(w)


def set_weights_torch(model, w, sizes_w, sizes_b):
    """Set model weights from a flat tensor (matching TF set_weights)."""
    idx = 0
    w_idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            n = sizes_w[w_idx]
            param.data = w[idx:idx + n].reshape(param.shape).clone()
            idx += n
            w_idx += 1
        elif 'bias' in name:
            n = sizes_b[min(w_idx - 1, len(sizes_b) - 1)] if w_idx > 0 else sizes_b[0]
            param.data = w[idx:idx + n].reshape(param.shape).clone()
            idx += n


# =============================================================================
# PyTorch helpers
# =============================================================================
def to_torch(x, device='cpu', dtype=torch.float32, requires_grad=False):
    """Convert numpy array to torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    t = torch.tensor(x, dtype=dtype, device=device)
    t.requires_grad_(requires_grad)
    return t


def grad(outputs, inputs, grad_outputs=None, retain_graph=True, create_graph=True):
    """Convenience wrapper for torch.autograd.grad."""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(outputs)
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph
    )


# =============================================================================
# Tee for logging
# =============================================================================
class Tee:
    """Duplicate stdout to a file."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def ensure_dir(path):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)
