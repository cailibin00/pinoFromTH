import os
import sys
import numpy as np
import torch


class Tee:
    """Simultaneously write to a file and the original stdout.

    Usage:
        tee = Tee("path/to/log.txt")
        sys.stdout = tee
        ...  # all print() goes to both console and file
        sys.stdout = tee.stdout  # restore
        tee.close()
    """

    def __init__(self, file_path):
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def mse(pred, target, weights=None):
    if weights is not None:
        diff = weights * (pred - target)
    else:
        diff = pred - target
    return torch.mean(diff ** 2)


def latin_hypercube_sample(n_samples, bounds):
    bounds = np.asarray(bounds, dtype=np.float64)
    dim = bounds.shape[0]
    rng = np.random.default_rng()
    result = np.empty((n_samples, dim), dtype=np.float64)
    for j in range(dim):
        perm = rng.permutation(n_samples)
        step = (perm + rng.random(n_samples)) / n_samples
        low, high = bounds[j]
        result[:, j] = low + step * (high - low)
    return result


def multimesh(arrs):
    lens = list(map(len, arrs))
    dim = len(arrs)
    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, size in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(size, axis=j)
        ans.append(arr2)
    return ans


def flatten_and_stack(mesh):
    dims = np.shape(mesh)
    output = np.zeros((len(mesh), np.prod(dims[1:])))
    for i, arr in enumerate(mesh):
        output[i] = arr.flatten()
    return output.T


def piecewise_lr(epoch, boundaries, values):
    for boundary, value in zip(boundaries, values):
        if epoch < boundary:
            return value
    return values[-1]


def ensure_dir(path):
    import os
    os.makedirs(path, exist_ok=True)


def to_torch(x, device, dtype=torch.float32, requires_grad=False):
    tensor = torch.as_tensor(x, dtype=dtype, device=device)
    if requires_grad:
        tensor = tensor.clone().detach().requires_grad_(True)
    return tensor


def grad(outputs, inputs, create_graph=True, retain_graph=True):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=False,
    )[0]


def find_L2_error(u_pred, u_star):
    """Compute relative L2 error between prediction and ground truth."""
    return np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
