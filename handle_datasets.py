import os
import warnings
from glob import glob as gl

import cv2
import numpy as np
from keras.datasets import mnist, cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from scipy.io import loadmat

glob = lambda k: sorted(gl(k))
pjoin = os.path.join

if 'DATASET_ROOT' not in os.environ:
    warnings.warn("$DATASET_ROOT is not set, defaulting to author's path.")
    os.environ['DATASET_ROOT'] = '/media/braf3002/hdd2'

DATASET_ROOT = os.environ['DATASET_ROOT']
assert os.path.exists(DATASET_ROOT), '$DATASET_ROOT is not a valid path!'

# Paths for all datasets
notMNIST = pjoin(DATASET_ROOT, 'Datasets/notMNIST_small')
INRIA = pjoin(DATASET_ROOT, 'Datasets/INRIA/INRIAPerson')
SEEFOOD = pjoin(DATASET_ROOT, 'Datasets/seefood/seefood')
SVHN = pjoin(DATASET_ROOT, 'Datasets/SVHN/')
STL10 = pjoin(DATASET_ROOT, 'Datasets/stl10/')
COMP_CARS = pjoin(DATASET_ROOT, 'Datasets/compCars/data')
MIOTCD = pjoin(DATASET_ROOT, 'Datasets/mio_tcd_classification')
CHINA = pjoin(DATASET_ROOT, 'Datasets/ChinaSet_AllFiles')


def read_notMnist(path):
    """Read the notMNIST dataset"""
    imgs = []
    labels = []
    cls = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for pt in glob(pjoin(path, '*', '*.png')):
        img = cv2.imread(pt)
        if img is not None:
            imgs.append(img)
            labels.append(cls.index(pt.split('/')[-2]))

    return (np.array(imgs), np.array(labels)), None


def read_inria(pt):
    """Read the INRIA dataset"""

    def load_data(pt2):
        return [(cv2.resize(cv2.imread(pjoin(pt2, 'pos', k)), (64, 128)), 1) for k in os.listdir(pjoin(pt2, 'pos'))] + [
            (cv2.resize(cv2.imread(pjoin(pt2, 'neg', k)), (64, 128)), 0) for k in os.listdir(pjoin(pt2, 'neg'))]

    (X_train, y_train) = zip(*load_data(pjoin(pt, 'Train')))
    (X_test, y_test) = zip(*load_data(pjoin(pt, 'Test')))
    return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test))


def read_seefood(path):
    """Read the Seefood dataset"""
    imgs = {'train': [], 'test': []}
    labels = {'train': [], 'test': []}
    cls = ['not_hot_dog', 'hot_dog']
    for pt in glob(pjoin(path, '*', '*', '*.jpg')):
        img = cv2.resize(cv2.imread(pt), (200, 200))
        if img is not None:
            mod, cl, _ = pt.split('/')[-3:]
            imgs[mod].append(img)
            labels[mod].append(cls.index(cl))

    return (np.array(imgs['train']), np.array(labels['train'])), (np.array(imgs['test']), np.array(labels['test']))


def read_SVHN(pt):
    """Read the SVHN dataset"""
    data = loadmat(pjoin(pt, 'train_32x32.mat'))
    datat = loadmat(pjoin(pt, 'test_32x32.mat'))
    X, y = data['X'], data['y']
    X = np.rollaxis(X, -1, 0)
    X2, y2 = datat['X'], datat['y']
    X2 = np.rollaxis(X2, -1, 0)
    # From 0 to 10
    y, y2 = y - 1, y2 - 1
    return (X, y), (X2, y2)


def read_STL10(pt):
    """Read the STL-10 dataset"""
    data = loadmat(pjoin(pt, 'train.mat'))
    datat = loadmat(pjoin(pt, 'test.mat'))
    X, y = data['X'], data['y']
    X2, y2 = datat['X'], datat['y']
    # From 0 to 10
    y, y2 = y - 1, y2 - 1
    X = np.reshape(X, (-1, 96, 96, 3))
    X2 = np.reshape(X2, (-1, 96, 96, 3))
    return (X, y), (X2, y2)


def read_compcars(pt):
    """Read the compCars dataset"""
    data = open(pjoin(pt, 'train_test_split', 'classification', 'train.txt'), 'r').readlines()
    clss = [k[0] for k in loadmat(pjoin(pt, 'misc', 'make_model_name.mat'))['make_names'][:, 0]]
    size = (128, 128)
    np.random.seed(200)

    makers = zip(*np.unique([int(k.split('/')[0]) for k in data], return_counts=True))
    makers_k = [k[0] for k in sorted(makers, key=lambda k: k[1])[-11:-1]]
    print(np.array(clss)[np.array(makers_k)])

    imgs, lbls = [], []

    for make, model, yr, file in (k.split('/') for k in data):
        if int(make) not in makers_k:
            continue
        imgs.append(cv2.resize(cv2.imread(pjoin(pt, 'image', make, model, yr, file.strip())), size))
        lbls.append(makers_k.index(int(make)))
    return (np.array(imgs), np.array(lbls)), None


def read_cifar10_deer_dog():
    deer = 4
    dog = 5
    (X_train,y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train[np.isin(y_train[:, 0], [deer, dog]), ...]
    y_train = y_train[np.isin(y_train, [deer, dog])]
    X_test = X_test[np.isin(y_test[:, 0], [deer, dog]), ...]
    y_test = y_test[np.isin(y_test, [deer, dog])]
    y_train[y_train == deer] = 0
    y_train[y_train == dog] = 1

    y_test[y_test == deer] = 0
    y_test[y_test == dog] = 1
    return (X_train, y_train), (X_test, y_test)


def read_china_set(pt):
    shen_paths = glob(os.path.join(pt, 'CXR_png', '*.png'))
    imgs, label = zip(*[(cv2.resize(cv2.cvtColor(cv2.imread(k), cv2.COLOR_BGR2RGB), (224, 224)),
                         int(k.split('_')[-1].split('.')[0]))
                        for k in shen_paths])
    return (np.array(imgs), np.array(label)), None

mio_tcd_clss = ["articulated_truck", "bicycle", "bus", "car", "motorcycle", "non-motorized_vehicle",
                "pedestrian", "pickup_truck", "single_unit_truck", "work_van", "background"]


def read_miotcd(pt):
    """Read the mio-tcd classification dataset"""
    # WARNING : This only return the paths, too big otherwise
    acc = []
    for k in glob(pjoin(pt, 'train', '*', '*.jpg')):
        cls = k.split('/')[-2]
        acc.append((k, mio_tcd_clss.index(cls)))

    return list(map(np.array, zip(*acc))), None


def make_small(data, n_samp=500, seed=None):
    """
    Reduce the number of sample per class
    Args:
        data ((X_train, y_train), (X_test, y_test)):
        n_samp (int, float): Number of sample per class
        seed (int): RNG seed

    Returns:(X_train, y_train), (X_test, y_test)

    """
    if seed:
        np.random.seed(seed)
    (X_train, y_train), test_data = data
    y_train = y_train.reshape([y_train.shape[0], -1])
    sh = X_train.shape[1:]
    acc_x = []
    acc_y = []
    for cl in np.unique(y_train, axis=0):
        dat = X_train[np.all(y_train == cl, -1)]
        s = int(n_samp) if n_samp > 1 else int(n_samp * len(dat))
        acc_x.extend(dat[:s])
        acc_y += [cl] * s
    X_train = np.array(acc_x).reshape((-1,) + sh)
    y_train = np.array(acc_y).reshape([len(acc_y), -1])
    return (X_train, y_train), test_data


def augment(data, samp=500):
    """
    Randomly augment `samp` samples
    Args:
        data ((X_train, y_train), (X_test, y_test)):
        samp (int): Number of samples  to augment

    Returns: (X_train, y_train), (X_test, y_test)

    """
    s = 1337
    (X_train, y_train), (X_test, y_test) = data
    imgd = ImageDataGenerator(rotation_range=10, horizontal_flip=True)
    x_acc, y_acc = [], []
    for idx in np.random.choice(len(X_train), samp):
        x, y = X_train[idx], y_train[idx]
        x = imgd.random_transform(x, seed=s)
        s += 1
        x_acc.append(x)
        y_acc.append(y)
    return (np.vstack((X_train, x_acc)), np.vstack((y_train, y_acc))), (X_test, y_test)


def randomize(X_train, y_train, random_cls):
    """
    Randomize `random_cls` classes
    Args:
        X_train (ndarray):
        y_train (ndarray):
        random_cls (int):  Number of classes to shuffle

    Returns: (X_train, y_train)

    """
    a = np.arange(10)
    np.random.shuffle(a)
    rand_cls = a[:random_cls]
    y_rand = y_train[np.isin(y_train, rand_cls)].copy()
    np.random.shuffle(y_rand)
    y_train[np.isin(y_train, rand_cls)] = y_rand
    return (X_train, y_train)


# All of the possible datasets
all_datasets = {'mnist': lambda: mnist.load_data(),
                'cifar10': lambda: cifar10.load_data(),
                'cifar10aug': lambda: augment(make_small(cifar10.load_data(), 1500, seed=1337), 10 * 2000),
                'cifar100': lambda: cifar100.load_data(),
                'cifar_deer_dog': lambda: read_cifar10_deer_dog(),
                'china': lambda: read_china_set(CHINA),
                'notMNIST': lambda: read_notMnist(notMNIST),
                'svhn': lambda: read_SVHN(SVHN),
                'stl10': lambda: read_STL10(STL10),
                'inria': lambda: read_inria(INRIA),
                'seefood': lambda: read_seefood(SEEFOOD),
                'compcars': lambda: read_compcars(COMP_CARS),
                'miotcd': lambda: read_miotcd(MIOTCD)}

all_datasets.update({'cifar10_small{}'.format(k): lambda: cifar10.load_data() for k in [500, 1000, 2500, 5000, 6000]})
all_datasets.update({'mnist_small{}'.format(k): lambda: mnist.load_data() for k in [500, 1000, 2500, 5000]})

def add_dataset(ds_name, func, *args):
    """
    Add a dataset to the set of known dataset
    Args:
        ds_name: str, name of the dataset
        func: function Any -> (X_train, y_train), Optional(None, (X_test, y_test))
        *args: arguments to  func
    """
    if ds_name in all_datasets:
        raise ValueError(f"{ds_name} already used.")
    all_datasets[ds_name] = lambda: func(*args)



# dataset used in the paper
paper_dataset = ['mnist', 'notMNIST', 'svhn', 'stl10', 'inria', 'seefood', 'cifar10', 'compcars']
two_class_dataset = ['inria', 'seefood']
ten_class_dataset = ['notMNIST', 'svhnfull', 'stl10', 'mnist', 'cifar10', 'compcars']


def main():
    # Sanity check to see if we can read all datasets
    PATHS = [notMNIST, INRIA, SEEFOOD, COMP_CARS, STL10, SVHN]
    PATHS = [CHINA]
    FUNCS = [read_notMnist, read_inria, read_seefood, read_compcars, read_STL10, read_SVHN]
    FUNCS = [read_china_set]

    (X_train, Y_train), k = read_cifar10_deer_dog()
    print(X_train.shape)
    print(np.unique(Y_train, return_counts=True))

    for f, p in zip(FUNCS, PATHS):
        print(p)
        (X_train, Y_train), k = f(p)
        print(X_train.shape)
        print(np.unique(Y_train, return_counts=True))
        print()


if __name__ == '__main__':
    main()
