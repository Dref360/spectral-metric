import numpy as np
import pytest
from sklearn.datasets import make_blobs

from spectral_metric.estimator import CumulativeGradientEstimator

N_CLASS = 3


@pytest.fixture
def a_3_class_dataset():
    rng = np.random.RandomState(2021)
    x, y = make_blobs(1000, 256, centers=rng.randn(3, 256), cluster_std=3, random_state=rng)
    return x, y


def test_estimator(a_3_class_dataset):
    x, y = a_3_class_dataset
    csg = CumulativeGradientEstimator(10, 5)
    csg.fit(x, y)
    assert len(csg.evals) == N_CLASS and np.allclose(csg.evals.min(), 0)
    assert csg.evecs.shape == (N_CLASS, N_CLASS)
    assert csg.W.shape == (N_CLASS, N_CLASS)
    assert csg.csg >= 0
    assert csg.difference.shape == (N_CLASS, N_CLASS)
    assert np.allclose(csg.difference.diagonal(), 0)
    assert list(csg.samples.keys()) == list(range(N_CLASS))
    for items in csg.samples.values():
        assert len(items) == 10
        assert all(item.shape == (N_CLASS,) for item in items)
