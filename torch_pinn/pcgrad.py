import torch


def _flatten_grads(grads, params):
    flat = []
    for grad, param in zip(grads, params):
        if grad is None:
            flat.append(torch.zeros_like(param).reshape(-1))
        else:
            flat.append(grad.reshape(-1))
    return torch.cat(flat)


def _unflatten_to(grads_flat, params):
    out = []
    start = 0
    for param in params:
        numel = param.numel()
        out.append(grads_flat[start:start + numel].view_as(param))
        start += numel
    return out


def pcgrad(losses, params):
    params = list(params)
    task_grads = []
    for loss in losses:
        grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
        task_grads.append(_flatten_grads(grads, params))

    proj_grads = []
    for grad_i in task_grads:
        proj = grad_i.clone()
        for grad_j in task_grads:
            denom = torch.sum(grad_j * grad_j) + 1e-12
            inner = torch.sum(proj * grad_j)
            coeff = inner / denom
            proj = proj - torch.minimum(coeff, torch.zeros_like(coeff)) * grad_j
        proj_grads.append(proj)

    merged = torch.stack(proj_grads, dim=0).sum(dim=0)
    return _unflatten_to(merged, params)
