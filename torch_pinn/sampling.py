"""
Latin Hypercube Sampling with ESE (Enhanced Stochastic Evolutionary) optimization.
Ported from tensordiffeq/sampling.py (SMT library based).

This module provides LHS sampling capability matching the TF version.
"""

import numpy as np
from scipy.spatial.distance import cdist


class LHS:
    """Latin Hypercube Sampling with optional maximin ESE optimization."""

    def __init__(self, xlimits, criterion='ese', random_state=None):
        self.xlimits = np.array(xlimits)
        self.criterion = criterion
        self.n_dim = len(xlimits)
        if random_state is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = np.random.RandomState(random_state)

    def __call__(self, n_points):
        # Generate initial LHS
        x = self._lhs_initial(n_points)

        if self.criterion == 'ese':
            x = self._maximinESE(x, n_points)
        elif self.criterion == 'maximin':
            x = self._maximin(x, n_points)

        return x

    def _lhs_initial(self, n_points):
        """Generate initial Latin Hypercube sample."""
        x = np.zeros((n_points, self.n_dim))
        for i in range(self.n_dim):
            cut = self.rng.permutation(n_points) + self.rng.uniform(size=n_points)
            x[:, i] = self.xlimits[i, 0] + cut * (self.xlimits[i, 1] - self.xlimits[i, 0]) / n_points
        return x

    def _maximin(self, x, n_points):
        """Basic maximin optimization via random swaps."""
        n_iter = min(1000, 100 * n_points)
        phi = self._mindist(x)

        for _ in range(n_iter):
            i, j = self.rng.choice(n_points, 2, replace=False)
            k = self.rng.randint(self.n_dim)

            x_new = x.copy()
            x_new[i, k], x_new[j, k] = x_new[j, k], x_new[i, k]

            phi_new = self._mindist(x_new)
            if phi_new > phi:
                x = x_new
                phi = phi_new

        return x

    def _maximinESE(self, x, n_points):
        """
        Enhanced Stochastic Evolutionary (ESE) algorithm for maximin LHS.
        Based on Jin, Chen, Sudjianto (2005).
        """
        n_iter = min(500, 50 * n_points)
        T0 = 0.005 * self._phi_p(x)
        T = T0
        phi = self._mindist(x)
        x_best = x.copy()
        phi_best = phi
        n_accept = 0
        n_improve = 0

        for _ in range(n_iter):
            # Random columnwise pair swap
            i, j = self.rng.choice(n_points, 2, replace=False)
            k = self.rng.randint(self.n_dim)

            x_new = x.copy()
            x_new[i, k], x_new[j, k] = x_new[j, k], x_new[i, k]

            phi_new = self._mindist(x_new)

            if phi_new > phi:
                x = x_new
                phi = phi_new
                n_accept += 1
                n_improve += 1
                if phi > phi_best:
                    x_best = x.copy()
                    phi_best = phi
            else:
                # Accept worse solution with probability exp(-delta/T)
                delta = phi - phi_new
                p_accept = np.exp(-delta / (T + 1e-12))
                if self.rng.uniform() < p_accept:
                    x = x_new
                    phi = phi_new
                    n_accept += 1

            # Update temperature
            if n_accept > 100 or _ % 100 == 0:
                n_accept = 0
                if n_improve < 50:
                    T = T0
                else:
                    T = T / 2

        return x_best

    def _mindist(self, x):
        """Minimum distance between any two points."""
        dists = cdist(x, x)
        np.fill_diagonal(dists, np.inf)
        return np.min(dists)

    def _phi_p(self, x):
        """Phi_p criterion (p=50)."""
        p = 50
        dists = cdist(x, x)
        np.fill_diagonal(dists, np.inf)
        dists = dists[np.triu_indices(len(x), 1)]
        return np.sum(dists ** (-p)) ** (1 / p)
