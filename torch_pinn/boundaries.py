import numpy as np
from .utils import multimesh, flatten_and_stack


def get_linspace(dict_):
    """Extract linspace array from a domain dictionary entry."""
    lin_key = "linspace"
    return [val for key, val in dict_.items() if lin_key in key][0]


class BC:
    """Base class for boundary conditions."""

    def __init__(self):
        self.isPeriodic = False
        self.isInit = False
        self.isNeumann = False

    def compile(self):
        self.input = self.create_input()

    def create_input(self):
        raise NotImplementedError("Subclasses must implement create_input()")

    def get_dict(self, var):
        return next(item for item in self.domain.domaindict if item["identifier"] == var)

    def get_not_dims(self, var):
        self.dicts_ = [item for item in self.domain.domaindict if item['identifier'] != var]
        return [get_linspace(dict_) for dict_ in self.dicts_]

    def create_target_input_repeat(self, var, target):
        fidelity_key = "fidelity"
        fids = []
        for dict_ in self.dicts_:
            res = [val for key, val in dict_.items() if fidelity_key in key]
            fids.append(res)
        reps = np.prod(fids)
        if type(target) is str:
            return np.repeat(self.dict_[(var + target)], reps)
        else:
            return np.repeat(target, reps)


class dirichletBC(BC):
    """Dirichlet boundary condition.

    Usage:
        lower_bc = dirichletBC(Domain, val=P_i, var='R', target="lower")
        upper_bc = dirichletBC(Domain, val=P_o, var='R', target="upper")
    """

    def __init__(self, domain, val, var, target):
        self.domain = domain
        self.val = val
        self.var = var
        self.target = target
        super().__init__()
        self.dicts_ = [item for item in self.domain.domaindict if item['identifier'] != self.var]
        self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == self.var)
        self.target = self.dict_[var + target]
        self.isDirichlect = True
        self.compile()

    def create_input(self):
        repeated_value = self.create_target_input_repeat(self.var, self.target)
        repeated_value = np.reshape(repeated_value, (-1, 1))
        mesh = flatten_and_stack(multimesh(self.get_not_dims(self.var)))
        mesh = np.insert(mesh, self.domain.vars.index(self.var), repeated_value.flatten(), axis=1)
        return mesh
