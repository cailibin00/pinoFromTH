import numpy as np
from .utils import latin_hypercube_sample


class DomainND:
    """N-dimensional domain for PINN collocation point generation.

    Ported from tensordiffeq/domains.py. Uses local latin_hypercube_sample
    instead of the TensorFlow-based LHS from tensordiffeq.sampling.
    """

    def __init__(self, var, time_var=None):
        self.vars = var
        self.domaindict = []
        self.domain_ids = []
        self.time_var = time_var

    def generate_collocation_points_old(self, N_f):
        """Simple LHS-based collocation point generation."""
        range_list = [
            [val for key, val in dict_.items() if "range" in key][0]
            for dict_ in self.domaindict
        ]
        limits = np.array(range_list)
        X_f = latin_hypercube_sample(N_f, limits)
        return X_f

    def generate_collocation_points(self, N_f, points_scala):
        """Grid-based + LHS hybrid collocation point generation.

        Uses a structured mesh grid in polar-like (R, theta) coordinates,
        matching the original TensorFlow version's behavior.
        """
        range_list = [
            [val for key, val in dict_.items() if "range" in key][0]
            for dict_ in self.domaindict
        ]
        range_list_2 = [
            [val for key, val in dict_.items() if "range" in key][0]
            for dict_ in self.domaindict
        ]
        limits_middle = np.array(range_list_2)
        X_f_4 = latin_hypercube_sample(round(N_f * 1), limits_middle)

        n_2 = np.round(np.sqrt(N_f * points_scala))
        n_1 = np.round(N_f / n_2)
        R = np.linspace(range_list[0][0], range_list[0][1], int(n_1))
        theta = np.linspace(range_list[1][0], range_list[1][1], int(n_2))
        R, THETA = np.meshgrid(R, theta)
        X_f_2 = np.hstack((R.flatten()[:, None], THETA.flatten()[:, None]))

        # Use X_f_2 (structured grid) as the primary set
        X_f = X_f_2
        return X_f

    def add(self, token, vals, fidel):
        """Register a domain variable with its range and fidelity."""
        self.domain_ids.append(token)
        self.domaindict.append({
            "identifier": token,
            "range": vals,
            (token + "fidelity"): fidel,
            (token + "linspace"): np.linspace(vals[0], vals[1], fidel),
            (token + "upper"): vals[1],
            (token + "lower"): vals[0]
        })
