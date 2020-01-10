import cv2
import numpy as np


def unison_shuffled_copies(a, b):
    """
    Shuffle both `a` and `b` in the same fashion
    Args:
        a: ndarray
        b: ndarrau

    Returns: (a,b) shuffled

    """
    assert len(a) == len(b), (len(a), len(b))
    p = np.random.permutation(len(a))
    return a[p], b[p]


def split(imgs, labels, pc=0.9):
    """
    Split the dataset in 2
    Args:
        imgs: Input images
        labels: Groudtruth
        pc: Proportion to keep.

    Returns: (x_train, y_train), (x_test, y_test)

    """
    spl = int(pc * len(imgs))
    imgs, imgs_t = imgs[:spl], imgs[spl:]
    labels, labels_t = labels[:spl], labels[spl:]

    return (np.array(imgs), np.array(labels)), (np.array(imgs_t), np.array(labels_t))


def resize_all(x_train, x_test=None, size=64):
    """
    Resize both `x_train` and `x_test` to `(size,size)`
    Args:
        x_train: ndarray, train images
        x_test: Optional, ndarray, test images
        size: side length

    Returns: x_train, x_test resized

    """
    x_train = np.array([cv2.resize(k, (size, size)) for k in x_train])
    if len(x_train.shape) == 3:
        x_train = x_train[..., np.newaxis]
    if x_test is not None:
        x_test = np.array([cv2.resize(k, (size, size)) for k in x_test], np.float32)
        if len(x_test.shape) == 3:
            x_test = x_test[..., np.newaxis]

    return np.array(x_train, np.float32), x_test


def need_sequence(dat):
    return np.issubdtype(dat.dtype, np.string_) or np.issubdtype(dat.dtype, np.unicode_)
