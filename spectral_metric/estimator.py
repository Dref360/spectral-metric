from itertools import product

import numpy as np
import scipy
import scipy.spatial
from scipy.sparse.csgraph import laplacian
from sklearn.manifold import Isomap

from spectral_metric.config import log
from spectral_metric.lib import find_samples, compute_expectation_with_monte_carlo


class CumulativeGradientEstimator(object):
    def __init__(self, M_sample=250, k_nearest=3, isomap=None):
        """
        The Cumulative Gradient Estimator, estimates the complexity of a dataset.
        Args:
            M_sample (int): Number of sample per class to use
            k_nearest (int): Number of neighbours to look to compute $P(C_c \vert x)$.
            isomap (None, int): Optional, number of component of the Isomap
        """
        self.M_sample = M_sample
        self.k_nearest = k_nearest
        self.isomap = isomap

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
        class_samples = np.array(find_samples(data_x, self.n_class, target, M=self.M_sample))

        # Compute isomap
        if self.isomap:
            class_samples, data_x = make_isomap(class_samples, data_x, self.isomap)
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
        self.S = compute_expectation_with_monte_carlo(data, target, class_samples,
                                                      n_class=self.n_class,
                                                      k_nearest=self.k_nearest)

        # Compute the D matrix
        self.W = np.empty([self.n_class, self.n_class])
        for i, j in (product(range(self.n_class), range(self.n_class))):
            self.W[i, j] = 1 - scipy.spatial.distance.braycurtis(self.S[i], self.S[j])

        self.difference = 1 - self.W

        # Get the Laplacian and its eigen values
        self.L_mat, dd = laplacian(self.W, False, True)
        self.evals, self.evecs = np.linalg.eigh(self.L_mat)


def make_isomap(class_samples, data, isomap):
    """
    Make Isomap if enabled
    Args:
        class_samples: ndarray [C, M_sample, ?]
        data: Dataset input
        isomap: Whether to use Isomap or not
    Returns:
        (transformed class samples, transformed data)
    """

    log("Isomap", end='')
    obj = Isomap(n_components=isomap, n_neighbors=120, n_jobs=4)
    k1, k2, _ = class_samples.shape
    class_samples = obj.fit_transform(class_samples.reshape([-1, data.shape[-1]])).reshape([k1, k2, isomap])
    data = obj.transform(data)
    log('Done!')
    return class_samples, data
