from config import read_ds, log
from embedding import embeddings_dict, TSNEEmbedding
from handle_datasets import randomize, make_small
import numpy as np


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

    log(f"X shape {data.shape}")
    log(f"Y shape {target.shape}")
    return data, target
