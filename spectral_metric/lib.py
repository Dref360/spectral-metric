import os

import scipy as sp
import scipy.stats
from sklearn.metrics.pairwise import *
from sklearn.neighbors import DistanceMetric

from spectral_metric.config import log, read_ds
from spectral_metric.embedding import *
from spectral_metric.handle_datasets import make_small, randomize

pjoin = os.path.join


def compute_expectation_with_monte_carlo(data, target, class_samples, n_class, k_nearest=5):
    """
    Compute E_{p(x\mid C_i)} [p(x\mid C_j)] for all classes from samples
    with a monte carlo estimator.
    Args:
        data: [num_samples, n_features], the inputs
        target: [num_samples], the classes
        class_samples: [n_class, M, n_features], the M samples per class
        n_class: The number of classes
        k_nearest: number of neighbors for k-NN

    Returns: [n_class, n_class] matrix with probabilities

    """

    def get_volume(dt):
        # [k_nearest, ?]
        # Compute the volume of the parzen-window
        dst = (np.abs(dt[1:] - dt[0]).max(0) * 2).prod()
        # Minimum is 1e-4 to avoid division per 0
        res = max(1e-4, dst)
        return res

    expectation = np.zeros([n_class, n_class])
    # For each class, we compute the expectation from the samples.
    # Create a  matrix of similarity [n_class, M, n_samples]
    # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
    dist = DistanceMetric.get_metric('euclidean')
    similarities = lambda k : np.array(dist.pairwise(class_samples[k], data))

    for k in range(n_class):
        # Compute E_{p(x\mid C_i)} [p(x\mid C_j)] using a Parzen-Window
        all_keep = similarities(k)
        for to_keep in all_keep:
            nearest = to_keep.argsort()[:k_nearest + 1]
            target_k = np.array(target)[nearest[1:]]
            # Get the Parzen-Window probability
            expectation[k] += (np.array([(target_k == ki).sum() / len(target_k) for ki in range(n_class)])) / get_volume(
                data[nearest])

        # Normalize the proportion for Bray-Curtis
        expectation[k] /= expectation[k].sum()

    # Make everything nice by replacing INF with zero values.
    expectation[np.logical_not(np.isfinite(expectation))] = 0

    # Logging
    log("----------------Diagonal--------------------")
    log(np.round(np.diagonal(expectation), 4))
    return expectation


def get_embedding(x_train, config):
    """
    Get the embedding for x_train
    Args:
        x_train: input data
        config: dict, the run configuration

    Returns: The embedded representation of x_train

    """
    ds_name = config['ds_name']
    embd = config['embd']
    tsne = config['tsne']
    x_train, ds_name = embeddings_dict[embd]()(x_train, ds_name)
    if tsne:
        x_train, _ = TSNEEmbedding()(x_train, ds_name)
    return x_train


def get_small(x_train, y_train, config):
    """
    Reduce the number of sample per class
    Args:
        x_train: ndarray
        y_train: ndarray
        config: dict

    Returns: x_train, y_train

    """
    small = config['small']
    if small:
        (x_train, y_train), _ = make_small(((x_train, y_train), (None, None)), float(small))
    return x_train, y_train


def shuffle_classes(x_train, y_train, config):
    """
    Shuffle classes to make the dataset more difficult.
    Args:
        x_train: ndarray
        y_train: ndarray
        config: dict

    Returns: x_train, y_train

    """
    ncls = config['shuffled_class']
    if ncls:
        x_train, y_train = randomize(x_train, y_train.copy(), int(ncls))
    return x_train, y_train


def fetch_dataset(config):
    """
    Get the `ds_name` dataset
    Args:
        ds_name: str: name of the dataset
        embd_param: Type of embedding to use.
            One of [None, 'embd', tsne', 'cnn_embd']

    Returns: (data, target, n_class)

    """
    ds_name = config['ds_name']
    ds = read_ds(ds_name)
    (x_train, y_train), _ = ds

    # get the embedding
    x_train = get_embedding(x_train, config)
    x_train, y_train = get_small(x_train, y_train, config)
    x_train, y_train = shuffle_classes(x_train, y_train, config)

    n_class = len(np.unique(y_train))

    # Flatten the dataset and categorize `y_train`
    data, target = preprocessing(x_train, y_train)
    return data, target, n_class


def find_samples(data, n_class, target, M=100):
    """
    Find M samples per class
    Args:
        data: [num_samples, n_features], the inputs
        target: [num_samples], the classes
        n_class: The number of classes
        M: (int, float), Number or proportion of sample per class


    Returns: [n_class, M, n_features], the M samples per class

    """
    class_samples = []
    for k in range(n_class):
        data_in_cls = data[target == k]
        to_take = M if M > 1 else int(M * len(data_in_cls))
        class_samples.append(data_in_cls[np.random.choice(len(data_in_cls), min(to_take, len(data_in_cls)), replace=False)])
    return np.array(class_samples)


def preprocessing(x_train, y_train, flatten=True):
    """
    Does the processing on the datas
    Args:
        x_train: nd-array like, the inputs
        y_train: nd-array like, the outputs
        flatten: bool, flatten the images or not

    Returns: data, target

    """
    if flatten:
        # Flatten the input
        data = x_train.reshape([x_train.shape[0], -1])
    else:
        data = x_train
    if len(y_train.shape) > 1:
        target = y_train[:, 0]
    else:
        target = y_train

    log("X shape", data.shape)
    log("Y shape", target.shape)
    return data, target


def get_cummax(eigens):
    """
    Get the confidence interval at 95%
    Args:
        eigens (): List of eigenvalues ndarray (?, ?)

    Returns: mu, interval

    """

    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
        return m, m - h, m + h

    # [?, n_class]
    grads = (eigens[:, 1:] - eigens[:, :-1])
    ratios = (grads / (np.array([list(reversed(range(1, grads.shape[-1] + 1)))]) + 1))
    # Take the mean of the cummax before summing it.
    cumsums = np.maximum.accumulate(ratios, -1).sum(1)
    mu, lb, ub = mean_confidence_interval(cumsums)
    return mu, ub - mu
