import logging
from itertools import product

import numpy as np
import scipy
import scipy.spatial
from scipy.sparse.csgraph import laplacian

from spectral_metric.lib import find_samples, compute_expectation_with_monte_carlo

log = logging.getLogger(__name__)


class CumulativeGradientEstimator(object):
    def __init__(self, M_sample=250, k_nearest=3):
        """
        The Cumulative Gradient Estimator, estimates the complexity of a dataset.
        Args:
            M_sample (int): Number of sample per class to use
            k_nearest (int): Number of neighbours to look to compute $P(C_c \vert x)$.
        """
        self.M_sample = M_sample
        self.k_nearest = k_nearest

    def fit(self, data, target):
        """
        Estimate the CSG metric from the data
        Args:
            data: data samples, ndarray (n_samples, n_features)
            target: target samples, ndarray (n_samples)
        """
        np.random.seed(None)
        data_x = data.copy()
        self.n_class = len(np.unique(target))

        # Do class sampling
        class_samples = np.array(find_samples(data_x, target, self.n_class, M=self.M_sample))

        self.compute(data_x, target, class_samples)
        return self

    def compute(self, data, target, class_samples):
        """
        Compute the difference matrix and the eigenvalues
        Args:
            data: data samples, ndarray (n_samples, n_features)
            target: target samples, ndarray (n_samples)
            class_samples : class samples, ndarray (n_class, M, n_features)
        """
        # Compute E_{p(x\mid C_i)} [p(x\mid C_j)]
        self.S, self.samples = compute_expectation_with_monte_carlo(
            data, target, class_samples, n_class=self.n_class, k_nearest=self.k_nearest
        )

        # Compute the D matrix
        self.W = np.empty([self.n_class, self.n_class])
        for i, j in product(range(self.n_class), range(self.n_class)):
            self.W[i, j] = 1 - scipy.spatial.distance.braycurtis(self.S[i], self.S[j])

        self.difference = 1 - self.W

        # Get the Laplacian and its eigen values
        self.L_mat, dd = laplacian(self.W, False, True)
        self.evals, self.evecs = np.linalg.eigh(self.L_mat)
        self.csg = self._csg_from_evals(self.evals)

    def _csg_from_evals(self, evals: np.ndarray) -> float:
        # [n_class]
        grads = evals[1:] - evals[:-1]
        ratios = grads / (np.array([list(reversed(range(1, grads.shape[-1] + 1)))]) + 1)
        csg: float = np.maximum.accumulate(ratios, -1).sum(1)
        return csg
