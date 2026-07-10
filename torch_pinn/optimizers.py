"""
L-BFGS optimizer for PINN fine-tuning.
Faithful port from tensordiffeq/optimizers.py (TensorFlow eager L-BFGS).

Uses PyTorch's native torch.optim.LBFGS with parameters matching the TF version:
- history_size=50 (matching TF's nCorrection)
- tolerance_change=1e-12 (matching TF's tolFun/tolX)
- max_iter from config
"""

import torch
import numpy as np
import time
from tqdm.auto import trange


def set_weights_flat(model, w_flat, sizes_w, sizes_b):
    """Set model weights from flat tensor."""
    idx = 0
    w_count = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            n = sizes_w[w_count]
            param.data = w_flat[idx:idx + n].reshape(param.shape).clone()
            idx += n
            w_count += 1
        elif 'bias' in name:
            b_count = w_count - 1 if w_count > 0 else 0
            n = sizes_b[min(b_count, len(sizes_b) - 1)]
            param.data = w_flat[idx:idx + n].reshape(param.shape).clone()
            idx += n


def get_weights_flat(model):
    """Get model weights as flat tensor."""
    w = []
    for _, param in model.named_parameters():
        w.append(param.data.flatten())
    return torch.cat(w)


def get_sizes_from_model(model):
    """Extract weight and bias sizes from model."""
    sizes_w = []
    sizes_b = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            sizes_w.append(param.numel())
        elif 'bias' in name:
            sizes_b.append(param.numel())
    return sizes_w, sizes_b


class LBFGS_Trainer:
    """
    L-BFGS trainer wrapping torch.optim.LBFGS.
    Matches the TF eager_lbfgs behavior.
    """

    def __init__(self, model, loss_fn, max_iter=100, learning_rate=0.8,
                 history_size=50, tolerance_change=1e-12):
        self.model = model
        self.loss_fn = loss_fn  # callable returning (loss, loss_all)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.history_size = history_size
        self.tolerance_change = tolerance_change
        self.sizes_w, self.sizes_b = get_sizes_from_model(model)

    def train(self):
        """Run L-BFGS training."""
        loss_history = []
        epoch_history = []
        loss_all_history = []

        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.learning_rate,
            max_iter=1,  # One step per outer iteration (manual loop)
            history_size=self.history_size,
            tolerance_change=self.tolerance_change,
            line_search_fn='strong_wolfe'
        )

        start_time = time.time()
        n_iter = 0

        with trange(self.max_iter) as t:
            for epoch in t:
                n_iter += 1

                def closure():
                    optimizer.zero_grad()
                    loss_value, loss_all = self.loss_fn()
                    loss_value.backward()
                    return loss_value

                loss_value = optimizer.step(closure)
                loss_val = loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value

                # Recompute for logging
                with torch.no_grad():
                    loss_val_full, loss_all = self.loss_fn()

                loss_history.append(loss_val_full.item() if isinstance(loss_val_full, torch.Tensor) else loss_val_full)
                epoch_history.append(n_iter)
                loss_all_history.append([l.item() if isinstance(l, torch.Tensor) else l for l in loss_all])

                t.set_description(f'L-BFGS epoch {n_iter}')
                t.set_postfix(loss=loss_val)

                if n_iter % 10 == 0:
                    elapsed = time.time() - start_time
                    start_time = time.time()

        return loss_history, epoch_history, loss_all_history
