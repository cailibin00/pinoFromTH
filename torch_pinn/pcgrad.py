"""
PCGrad (Projecting Conflicting Gradients) for multi-task learning.
Faithful port from tensordiffeq/models.py and tensordiffeq/PCGrad.py (TensorFlow).

Implements the full PCGrad algorithm:
1. Compute per-loss gradients
2. Flatten gradients
3. Random shuffle + conflict projection (cosine similarity)
4. Gradient reassembly and summation
"""

import torch
import numpy as np


def pcgrad(losses, params, adaptive_constant_func_PCGrad_loss=None, balance=False):
    """
    PCGrad gradient projection for multi-task learning.

    Args:
        losses: list of scalar loss tensors (each requires_grad)
        params: list of model parameters
        adaptive_constant_func_PCGrad_loss: optional ComputeSum_weight for loss balancing
        balance: whether to apply loss-based weight normalization

    Returns:
        projected_grads: list of projected gradients for each parameter (summed across tasks)
        total_loss: scalar, sum of all losses
    """
    num_tasks = len(losses)

    if num_tasks == 1:
        # Single task - just compute gradients normally
        total_loss = losses[0]
        grads = torch.autograd.grad(total_loss, params, retain_graph=True, create_graph=False)
        # Replace None grads with zeros
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
        return grads, total_loss

    # Compute per-task gradients
    grad_all = []
    for loss in losses:
        g = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
        # Replace None gradients with zeros
        g_clean = []
        for gi, p in zip(g, params):
            if gi is None:
                g_clean.append(torch.zeros_like(p))
            else:
                g_clean.append(gi)
        grad_all.append(g_clean)

    # Flatten gradients for each task
    grads_task = []
    for grads in grad_all:
        flat = torch.cat([g.reshape(-1) for g in grads])
        grads_task.append(flat)

    # Optional: loss-based weight normalization (matching TF balance logic)
    if balance and adaptive_constant_func_PCGrad_loss is not None:
        loss_values = [l.detach() for l in losses]
        adaptive_constant_func_PCGrad_loss.update(loss_values)

        loss_all_smooth = adaptive_constant_func_PCGrad_loss.adaptive_constant
        loss_all_smooth_reference = adaptive_constant_func_PCGrad_loss.adaptive_constant_step

        loss_effective = [loss_all_smooth[0, k] / (loss_all_smooth_reference[0, k] + 1e-12) for k in range(num_tasks)]
        loss_effective_01 = [loss_effective[k] / (sum(loss_effective) + 1e-12) for k in range(num_tasks)]

        grads_task_norm = [torch.sqrt(torch.sum(g ** 2)) for g in grads_task]
        mean_norm = sum(grads_task_norm) / num_tasks

        weight_grads_task_norm = [
            (loss_effective_01[k] * (mean_norm - grads_task_norm[k]) + grads_task_norm[k]) / (grads_task_norm[k] + 1e-12)
            for k in range(num_tasks)
        ]

        grads_task = [grads_task[k] * weight_grads_task_norm[k] for k in range(num_tasks)]

    # PCGrad: Random shuffle and project conflicting gradients
    indices = list(range(num_tasks))
    np.random.shuffle(indices)
    grads_task_shuffled = [grads_task[i] for i in indices]

    grads_task_proj = []
    for gi in grads_task_shuffled:
        grad_proj = gi.clone()
        for gj in grads_task:
            inner_product = torch.sum(grad_proj * gj)
            denominator = torch.sum(gj * gj) + 1e-12
            proj_direction = inner_product / denominator
            # Only remove conflicting (negative cosine) components
            grad_proj = grad_proj - torch.min(proj_direction, torch.tensor(0.0, device=gi.device)) * gj
        grads_task_proj.append(grad_proj)

    # Reassemble: sum all projected gradients
    proj_grads_flat = grads_task_proj

    # Reassemble to parameter shapes and sum
    proj_grads = []
    for j in range(num_tasks):
        start_idx = 0
        for idx, param in enumerate(params):
            grad_shape = param.shape
            flatten_dim = param.numel()
            proj_grad = proj_grads_flat[j][start_idx:start_idx + flatten_dim]
            proj_grad = proj_grad.reshape(grad_shape)
            if len(proj_grads) < len(params):
                proj_grads.append(proj_grad)
            else:
                proj_grads[idx] = proj_grads[idx] + proj_grad  # Sum across tasks
            start_idx += flatten_dim

    total_loss = sum(losses)
    return proj_grads, total_loss
