import numpy as np
import pytest
from sklearn.datasets import make_blobs

from spectral_metric.estimator import CumulativeGradientEstimator
from spectral_metric.types import SimilarityArrays

N_CLASS = 3


@pytest.fixture
def a_3_class_dataset():
    rng = np.random.RandomState(2021)
    x, y = make_blobs(1000, 256, centers=rng.randn(3, 256), cluster_std=3, random_state=rng)
    return x, y


@pytest.fixture
def a_3_class_dataset_with_wrong_indexing(a_3_class_dataset):
    x, y = a_3_class_dataset
    y[y == 2] = 3
    return x, y


@pytest.mark.parametrize('distance', ["euclidean", "cosine"])
def test_estimator(a_3_class_dataset, distance):
    x, y = a_3_class_dataset
    csg = CumulativeGradientEstimator(M_sample=10, k_nearest=5, distance=distance)
    csg.fit(x, y)
    assert len(csg.evals) == N_CLASS and np.allclose(csg.evals.min(), 0)
    assert csg.evecs.shape == (N_CLASS, N_CLASS)
    assert csg.W.shape == (N_CLASS, N_CLASS)
    assert csg.csg >= 0
    assert csg.difference.shape == (N_CLASS, N_CLASS)
    assert np.allclose(csg.difference.diagonal(), 0)
    assert list(csg.similarity_arrays.keys()) == list(range(N_CLASS))
    for items in csg.similarity_arrays.values():
        assert len(items) == 10
        for sample_id_key in items:
            assert isinstance(items[sample_id_key], SimilarityArrays)
            assert (len(items[sample_id_key].sample_probability) ==
                    len(items[sample_id_key].sample_probability_norm) == N_CLASS)
            assert items[sample_id_key].sample_probability.sum() == 1
    for s_matrix_row in csg.S:
        assert s_matrix_row.sum() == pytest.approx(1, 1E-20)