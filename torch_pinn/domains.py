"""
DomainND - Computational domain and collocation point generation.
Faithful port from tensordiffeq/domains.py (TensorFlow).
"""

import numpy as np
from .utils import latin_hypercube_sample


class DomainND:
    """N-dimensional computational domain for PINN collocation points."""

    def __init__(self, var, time_var=None):
        self.vars = var
        self.domaindict = []
        self.domain_ids = []
        self.time_var = time_var
        self.X_f = None  # Collocation points, set externally

    def add(self, token, vals, fidel):
        """Register a domain variable."""
        self.domain_ids.append(token)
        self.domaindict.append({
            "identifier": token,
            "range": vals,
            (token + "fidelity"): fidel,
            (token + "linspace"): np.linspace(vals[0], vals[1], fidel),
            (token + "upper"): vals[1],
            (token + "lower"): vals[0]
        })

    def generate_collocation_points_old(self, N_f):
        """Generate collocation points using pure LHS."""
        range_list = [
            [val for key, val in dict_.items() if "range" in key][0]
            for dict_ in self.domaindict
        ]
        limits = np.array(range_list)
        X_f = latin_hypercube_sample(N_f, limits)
        return X_f

    def generate_collocation_points(self, N_f, points_scala):
        """
        Generate collocation points using hybrid LHS + structured meshgrid.
        This is the DEFAULT method used in training (matching TF version exactly).

        The method creates a structured grid of (R, theta) points with
        refinement near R=0.92 (groove boundary).
        """
        range_list = [
            [val for key, val in dict_.items() if "range" in key][0]
            for dict_ in self.domaindict
        ]

        # Generate structured meshgrid (this is what gets returned)
        n_2 = np.round(np.sqrt(N_f * points_scala))
        n_1 = np.round(N_f / n_2)
        R = np.linspace(range_list[0][0], range_list[0][1], int(n_1))
        theta = np.linspace(range_list[1][0], range_list[1][1], int(n_2))
        R_mesh, THETA_mesh = np.meshgrid(R, theta)
        X_f_2 = np.hstack((R_mesh.flatten()[:, None], THETA_mesh.flatten()[:, None]))

        # In the TF version, only X_f_2 is returned
        X_f = X_f_2
        return X_f
