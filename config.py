import os
import types

import numpy as np

from handle_datasets import all_datasets

EMBEDDINGS = ['embd', 'cnn_embd', 'vgg', 'xception']

pjoin = os.path.join

TMP_DIR = pjoin(os.path.dirname(__file__), 'tmp3')
if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)

np.set_printoptions(suppress=True, linewidth=320, precision=5)
VERBOSE = False


def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def make_config(ds_name, embd=None, tsne=False, small=None, shuffled_class=None):
    """
    Create a configuration for a test run.
    Args:
        ds_name: name of the dataset
        embd: type of embedding, one of {None, 'embd', 'cnn_embd', 'vgg', 'xception'}
        tsne: Wheter to apply t-sne or not
        small: Number of samples per class, optional
        shuffled_class: Number of class to shuffle, optional

    Returns:

    """
    assert embd in EMBEDDINGS or embd is None
    return {
        'ds_name': ds_name,
        'embd': embd,
        'tsne': tsne,
        'small': small,
        'shuffled_class': shuffled_class,
    }


def read_ds(ds_name):
    """
    Get the dataset `ds_name`
    Args:
        ds_name: name of the dataset

    Returns: (X_train, Y_train), (X_test, Y_test)

    """
    # Get the mapping function
    ds = all_datasets[ds_name]
    # The dataset may be a tuple (function, arguments)
    if isinstance(ds, tuple):
        ds_f, args = ds
        ds = lambda: ds_f(args)
    # Call the function with its argument
    if isinstance(ds, types.LambdaType):
        ds = arena(ds_name, ds)
    return ds


class MemoryArena():
    """Simple memoization"""

    def __init__(self):
        self.memory = {}

    def __call__(self, name, fn):
        if name not in self.memory:
            self.memory[name] = fn()
        return self.memory[name]


arena = MemoryArena()
