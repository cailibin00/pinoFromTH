"""
torch_pinn - PyTorch implementation of PINN for Reynolds equation with JFO cavitation.
Faithful 1:1 port from tensordiffeq (TensorFlow).
"""

from .networks import (
    Coslayer_normalization,
    Out_Imp_BC_layer,
    Out_Imp_BC_value_layer,
    new_neural_period_polar_exactBC_two_output,
)

from .models import CollocationSolverND, ComputeSum_weight

from .domains import DomainND

from .boundaries import (
    BC,
    dirichletBC,
    FunctionDirichletBC,
    FunctionNeumannBC,
    IC,
    periodicBC,
)

from .utils import (
    mse,
    find_L2_error,
    latin_hypercube_sample,
    multimesh,
    flatten_and_stack,
    set_weights_torch,
    get_weights_torch,
    get_sizes,
    Tee,
    ensure_dir,
    to_torch,
    grad,
)

from .pcgrad import pcgrad

from .optimizers import LBFGS_Trainer

from .fit import fit as fit_model
