"""
Training loop for PINN solver.
Simplified from the TF version:
- Removed PCGrad (projecting conflicting gradients)
- Removed adaptive loss weighting (ComputeSum_weight)
- Replaced piecewise LR with CosineAnnealingLR scheduler
- Two-phase training: Adam + L-BFGS (optional)
"""

import time
import numpy as np
import torch
from tqdm.auto import trange


def fit(obj, tf_iter=0, newton_iter=0, newton_eager=True, scheduler=None):
    """
    Main training loop.

    Args:
        obj: CollocationSolverND instance
        tf_iter: number of Adam iterations
        newton_iter: number of L-BFGS iterations (0 to skip)
        newton_eager: whether to use eager L-BFGS
        scheduler: optional torch LR scheduler (stepped each epoch after optimizer.step)
    """
    start_time = time.time()
    epoch_base = obj.epoch_history[-1]

    # =========================================================================
    # Phase 1: Adam Training
    # =========================================================================
    if tf_iter > 0:
        print(f"Starting Adam training ({tf_iter} iterations)...")

        with trange(tf_iter) as t:
            for epoch in t:
                # Training step
                loss_value, loss_all = obj.train_step()

                # Step the scheduler after optimizer update
                if scheduler is not None:
                    scheduler.step()

                # Update epoch counter
                current_epoch = epoch + epoch_base + 1

                t.set_description(f'Adam epoch {current_epoch}')

                # Logging every 10 epochs
                if epoch % 10 == 0:
                    obj.loss_history.append(loss_value.item())
                    obj.epoch_history.append(current_epoch)
                    obj.loss_all_history.append([l.item() for l in loss_all])

                    if epoch % 500 == 0 and len(loss_all) >= 2:
                        loss_names = ['L_Reynolds', 'L_BC', 'L_FB']
                        loss_str = ' | '.join([
                            f'{name}={loss_all[i].item():.3e}'
                            for i, name in enumerate(loss_names) if i < len(loss_all)
                        ])
                        current_lr = obj.tf_optimizer.param_groups[0]['lr']
                        print(f'  Epoch {current_epoch}: Total={loss_value.item():.3e} | {loss_str} | lr={current_lr:.2e}')

                    t.set_postfix(loss=loss_value.item())

                # Save best model every 100 epochs
                if epoch % 100 == 0:
                    if loss_value.item() < obj.loss_value_min:
                        obj.loss_value_min = loss_value.item()
                        obj.save_weights(obj.best_weights_path)

    # =========================================================================
    # Phase 2: L-BFGS Training
    # =========================================================================
    if newton_iter > 0:
        print(f"Starting L-BFGS training ({newton_iter} iterations)...")

        if newton_eager:
            from .optimizers import LBFGS_Trainer

            def loss_fn():
                return obj.update_loss()

            lbfgs_trainer = LBFGS_Trainer(
                obj.u_model, loss_fn,
                max_iter=newton_iter + 1,
                learning_rate=0.8,
                history_size=50,
                tolerance_change=1e-12
            )

            loss_hist, epoch_hist, loss_all_hist = lbfgs_trainer.train()

            # Append L-BFGS history to Adam history
            obj.loss_history = obj.loss_history + loss_hist
            obj.epoch_history = obj.epoch_history + list(
                np.array(epoch_hist) + obj.epoch_history[-1]
            )
            obj.loss_all_history = obj.loss_all_history + loss_all_hist
        else:
            # Graph-mode L-BFGS not supported in PyTorch, fallback to eager
            print("Warning: graph-mode L-BFGS not supported in PyTorch, using eager mode.")
            from .optimizers import LBFGS_Trainer

            def loss_fn():
                return obj.update_loss()

            lbfgs_trainer = LBFGS_Trainer(
                obj.u_model, loss_fn,
                max_iter=newton_iter + 1,
                learning_rate=0.8,
                history_size=50,
                tolerance_change=1e-12
            )

            loss_hist, epoch_hist, loss_all_hist = lbfgs_trainer.train()

            obj.loss_history = obj.loss_history + loss_hist
            obj.epoch_history = obj.epoch_history + list(
                np.array(epoch_hist) + obj.epoch_history[-1]
            )
            obj.loss_all_history = obj.loss_all_history + loss_all_hist

    # =========================================================================
    # Completion
    # =========================================================================
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s")
