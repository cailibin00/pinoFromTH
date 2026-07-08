"""Training orchestration for torch_pinn models.

Provides train_model_torch() which mirrors the multi-stage training loop
from reynold_pinn.py's train_model(), including:
  - Piecewise constant LR schedules
  - w_wedge curriculum learning (wedge term weight ramp-up)
  - RAD_FB residual adaptive point refinement
  - Best model checkpointing
"""

from .utils import piecewise_lr


def train_model_torch(model, cfg, N_f_true):
    """Multi-stage training with LR schedules and adaptive refinement.

    Simplified version: no w_wedge curriculum, plain LR decay.

    Args:
        model: TorchCollocationSolver instance.
        cfg: Config object with training parameters.
        N_f_true: Number of collocation points after initial setup.
                  Used for RAD_FB (adaptive refinement).

    Returns:
        model: Trained solver instance.
    """
    # Learning rate schedules per stage
    lr_schedules = [
        {'boundaries': [20000, 40000], 'values': [1e-3, 1e-4, 1e-5]},
        {'boundaries': [20000, 40000], 'values': [1e-4, 1e-4, 1e-5]},
        {'boundaries': [20000, 40000], 'values': [1e-5, 1e-4, 1e-5]},
        {'boundaries': [20000, 40000], 'values': [1e-5, 1e-5, 1e-6]},
    ]

    # w_wedge schedule: 阶段升温, 从 1% 到 100% 恢复完整楔形物理
    w_wedge_schedule = [1e-2, 5e-2, 2e-1, 1.0]

    for stage_idx, schedule in enumerate(lr_schedules):
        # Update wedge weight for this stage
        if hasattr(model, 'set_w_wedge') and model.set_w_wedge is not None:
            w_val = w_wedge_schedule[stage_idx]
            model.set_w_wedge(w_val)
            print(f"[Stage {stage_idx + 1}/4] w_wedge = {w_val:.0e}")
        N_train = getattr(cfg, 'N_train', 5000)
        NL_train = getattr(cfg, 'NL_train', 4)

        for outer_idx in range(NL_train):
            # Train with piecewise constant LR
            for epoch in range(N_train):
                lr = piecewise_lr(epoch, schedule['boundaries'], schedule['values'])
                model.set_learning_rate(lr)

                loss_value, loss_all = model.train_step()
                global_epoch = model.epoch_history[-1]

                if epoch % 10 == 0:
                    model.loss_history.append(float(loss_value.cpu().item()))
                    if len(model.epoch_history) == 0 or model.epoch_history[-1] != global_epoch:
                        model.epoch_history.append(global_epoch + 1)
                    else:
                        model.epoch_history[-1] = global_epoch + 1
                    model.loss_all_history.append(
                        [float(v.cpu().item()) for v in loss_all]
                    )

                if epoch % 500 == 0:
                    loss_names = ['L_Reynolds', 'L_FB', 'L_BC', 'L_Interact']
                    parts = []
                    for i, name in enumerate(loss_names):
                        if i < len(loss_all):
                            parts.append(f'{name}={loss_all[i].cpu().item():.3e}')
                    loss_str = ' | '.join(parts)
                    print(
                        f'  Epoch {global_epoch} (stage {stage_idx + 1}, outer {outer_idx + 1}/{NL_train}): '
                        f'Total={loss_value.cpu().item():.3e} | {loss_str}'
                    )

                if epoch % 100 == 0:
                    loss_scalar = float(loss_value.cpu().item())
                    if loss_scalar < model.loss_value_min:
                        model.loss_value_min = loss_scalar
                        model.best_state_dict = (
                            {k: v.clone() for k, v in model.u_model.state_dict().items()}
                        )
                        if model.best_weights_path is not None:
                            model.save_weights(model.best_weights_path)

            # Adaptive point refinement after each outer loop
            if hasattr(model, 'f_model_FB') and model.f_model_FB is not None:
                ratio_RAD_list = getattr(cfg, 'ratio_RAD_list', [0.03, 0.01])
                model.RAD_FB(
                    model.f_model_list + [model.f_model_FB],
                    N_f_true,
                    num_add_points_test=round(10 * N_f_true),
                    num_add_points=[round(r * N_f_true) for r in ratio_RAD_list],
                    k=1,
                    c=1e-16,
                )

    return model
