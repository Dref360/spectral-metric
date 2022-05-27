import logging
import os
from collections import defaultdict
from typing import Tuple, List, Dict

import numpy as np
import scipy as sp
import scipy.stats
from sklearn.metrics import pairwise_distances

from spectral_metric.types import Array, SimilarityArrays

log = logging.getLogger(__name__)
pjoin = os.path.join


def compute_expectation_with_monte_carlo(
    data: Array,
    target: Array,
    class_samples: Dict[int, Array],
    class_indices: Dict[int, Array],
    n_class: int,
    k_nearest=5,
    distance: str = "euclidean",
) -> Tuple[Array, Dict[int, Dict[int, SimilarityArrays]]]:
    """
    Compute $E_{p(x | C_i)} [p(x | C_j)]$ for all classes from samples
    with a monte carlo estimator.
    Args:
        data: [num_samples, n_features], the inputs
        target: [num_samples], the classes
        class_samples: [n_class, M, n_features], the M samples per class
        class_indices: [n_class, indices], the indices of samples per class
        n_class: The number of classes
        k_nearest: The number of neighbors for k-NN
        distance: Which distance metric to use

    Returns:
        expectation: [n_class, n_class], matrix with probabilities
        similarity_arrays: [n_class, M, SimilarityArrays], dict of arrays with kNN class
                            proportions, raw and normalized by the Parzen-window,
                            accessed via class and sample indices

    """

    def get_volume(dt):
        # [k_nearest, ?]
        # Compute the volume of the parzen-window
        dst = (np.abs(dt[1:] - dt[0]).max(0) * 2).prod()
        # Minimum is 1e-4 to avoid division per 0
        res = max(1e-4, dst)
        return res

    similarity_arrays: Dict[int, Dict[int, SimilarityArrays]] = defaultdict(dict)
    expectation = np.zeros([n_class, n_class])  # S-matrix

    # For each class, we compute the expectation from the samples.
    # Create a matrix of similarity [n_class, M, n_samples]
    # https://scikit-learn.org/stable/modules/metrics.html#metrics
    similarities = lambda k: np.array(pairwise_distances(class_samples[k], data, metric=distance))

    for class_ix in class_samples:
        # Compute E_{p(x\mid C_i)} [p(x\mid C_j)] using a Parzen-Window
        all_similarities = similarities(class_ix)  # Distance arrays for all class samples
        all_indices = class_indices[class_ix]  # Indices for all class samples
        for m, sample_ix in enumerate(all_indices):
            indices_k = all_similarities[m].argsort()[: k_nearest + 1]  # kNN indices (incl self)
            target_k = np.array(target)[indices_k[1:]]  # kNN class labels (self is dropped)
            # Get the Parzen-Window probability
            probability = np.array(
                # kNN class proportions
                [(target_k == nghbr_class).sum() / k_nearest for nghbr_class in range(n_class)]
            )
            probability_norm = probability / get_volume(data[indices_k])  # Parzen-window normalized
            similarity_arrays[class_ix][sample_ix] = SimilarityArrays(
                sample_probability=probability, sample_probability_norm=probability_norm
            )
            expectation[class_ix] += probability_norm

        # Normalize the proportions to facilitate row comparison
        expectation[class_ix] /= expectation[class_ix].sum()

    # Make everything nice by replacing INF with zero values.
    expectation[np.logical_not(np.isfinite(expectation))] = 0

    # Logging
    log.info("----------------Diagonal--------------------")
    log.info(np.round(np.diagonal(expectation), 4))
    return expectation, similarity_arrays


def find_samples(
    data: np.ndarray, target: np.ndarray, n_class: int, M=100, seed=None
) -> Tuple[Dict[int, Array], Dict[int, Array]]:
    """
    Find M samples per class
    Args:
        data: [num_samples, n_features], the inputs
        target: [num_samples], the classes
        n_class: The number of classes
        M: (int, float), Number or proportion of sample per class
        seed: seeding for sampling.


    Returns: Selected items per class and their indices.

    """
    rng = np.random.RandomState(seed)
    class_samples = {}
    class_indices = {}
    indices = np.arange(len(data))
    for k in np.unique(target):
        indices_in_cls = rng.permutation(indices[target == k])
        to_take = min(M if M > 1 else int(M * len(indices_in_cls)), len(indices_in_cls))
        class_samples[k] = data[indices_in_cls[:to_take]]
        class_indices[k] = indices_in_cls[:to_take]
    return class_samples, class_indices


def get_cummax(eigens: np.ndarray) -> Tuple[float, float]:
    """
    Get the confidence interval at 95%.
    Useful if you have run CSG multiple times and wants an average.
    Args:
        eigens (): List of eigenvalues ndarray (?, ?)

    Returns: mu, interval

    """

    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * sp.stats.t._ppf((1 + confidence) / 2.0, n - 1)
        return m, m - h, m + h

    # [?, n_class]
    grads = eigens[:, 1:] - eigens[:, :-1]
    ratios = grads / (np.array([list(reversed(range(1, grads.shape[-1] + 1)))]) + 1)
    # Take the mean of the cummax before summing it.
    cumsums = np.maximum.accumulate(ratios, -1).sum(1)
    mu, lb, ub = mean_confidence_interval(cumsums)
    return mu, ub - mu
