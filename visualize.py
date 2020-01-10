from itertools import cycle

import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import quantile_transform

from config import log
from estimator import CumulativeGradientEstimator
from handle_datasets import mio_tcd_clss
from lib import fetch_dataset, get_cummax

LOOP = 30
LONG_LABEL = False

cmap = sns.cubehelix_palette(light=0.9, as_cmap=True)
font = FontProperties()
font.set_family('sans-serif')
"""CLASSES"""
cifar10_cls = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
notMNIST_cls = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', ]
stl10_cls = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
clss = {'cifar10': cifar10_cls, 'notMNIST': notMNIST_cls, 'stl10': stl10_cls, 'miotcd': mio_tcd_clss}


def visualize(config, k_nearest=10, isomap=None, M_sample=100):
    """
    Compute and plot the graph for a config
    Args:
        config (dict): run confiuration
        k_nearest (int): hyper-parameter k
        isomap (int): None or number of components to use for the Isomap
        M_sample (int): hyper-parameter M

    Returns:

    """
    font.set_weight('normal')
    print(config)
    data, ds_name, n_class, pca_params, target = get_data(config)

    estimator = CumulativeGradientEstimator(M_sample, k_nearest, isomap)
    estimator.fit(data, target)

    make_graph(estimator.difference, ds_name)
    font.set_weight('normal')


def make_graph(difference, ds_name):
    """
    Plot the graph of `ds_name`
    Args:
        difference: 1 - D matrix computed
        ds_name: str: name of dataset
    """
    from sklearn import manifold
    difference = (difference)
    mds = manifold.MDS(n_components=2, max_iter=1000, eps=1e-9, random_state=1337,
                       dissimilarity="precomputed", n_jobs=1)
    font.set_size(16)

    pos = mds.fit(difference).embedding_
    fig = plt.figure(1, dpi=300)
    ax = plt.axes([0., 0., 1., 1.])
    ax.axis('off')
    ax.set_title(ds_name)
    s = 100
    plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', s=s, lw=0, label='MDS')
    for i, p in enumerate(pos):
        tmp_name = ds_name.split('_')[0]
        ha = 'left' if p[0] < 0 else 'right'
        if tmp_name in clss:
            ax.annotate(clss[tmp_name][i], (p[0], p[1]), fontproperties=font, ha=ha)
        else:
            ax.annotate(str(i), (p[0], p[1]), fontproperties=font, ha=ha)
    similarities = difference.max() / difference * 100
    similarities[np.isinf(similarities)] = 0
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[pos[i, :], pos[j, :]]
                for i in range(len(pos)) for j in range(len(pos))]
    values = np.abs(similarities)
    lc = LineCollection(segments,
                        zorder=0, cmap='Blues',
                        norm=plt.Normalize(0, values.max()))
    lc.set_array(similarities.flatten())
    lc.set_linewidths(1. * np.ones(len(segments)))
    ax.add_collection(lc)
    plt.title(ds_name)
    plt.tight_layout(1.1)
    plt.show()


def process_many(config, k_nearest=3, isomap=None, M_sample=100, loop=1):
    """
    Compute the CSG `loop` time.
    """
    log(config, k_nearest, isomap, M_sample)
    data, ds_name, n_class, embedding, target = get_data(config)

    estimator = CumulativeGradientEstimator(M_sample, k_nearest, isomap)

    return config, Parallel(n_jobs=2)(
        delayed(estimator.fit)(data, target) for _ in (range(loop)))


def get_data(args):
    """
    Get the data for ds_name
    Args:
        args:

    Returns:

    """
    assert type(args) is dict
    embd = args['embd']
    data, target, n_class = fetch_dataset(args)
    if embd in ['cnn_embd', 'vgg', 'embd'] and args['tsne'] is False:
        # This normalize the data otherwise we would get overflow errors.
        data = quantile_transform(data)
    return data, args['ds_name'], n_class, embd, target


def test_job(**kwargs):
    """
    Aggregate the result of a run configuration
    Args:
        **kwargs (dict): params for `process_many`
                         keys: config, k_nearest, isomap, M_sample

    Returns: the result for all runs
             [(config params, eigenvalues)]

    """

    args, vals = list(process_many(**kwargs, loop=LOOP))
    return [([args, v.k_nearest, v.isomap, v.M_sample], v.evals) for v in vals]


def make_label(config, knn, iso, ksamp):
    lbl = config['ds_name']
    embedding_name = str(config['embd']) if config['embd'] else ''
    if config['tsne']:
        embedding_name += '_tsne'
    if config['shuffled_class']:
        lbl += ' {}'.format(config['shuffled_class'])

    if config['small']:
        lbl += ' reduced={}'.format(config['small'])
    if LONG_LABEL:
        lbl += ' ({}, k={}, iso={}, M={})'.format(embedding_name, knn, iso, ksamp)
    return lbl


def plot_with_err(*vals):
    """
    Plot the eigens values and the CSG
    Args:
        *vals ([(config, evals)]): data to plot
    """
    plt.figure(dpi=300)
    plt.subplot(111)
    bars = []
    vals = sorted(vals, key=lambda k: np.array([ki[1] for ki in k]).mean(0).max(), reverse=True)
    for eigens, ls in zip(vals, cycle(['-', '--', '-.', ':'])):
        lbl = eigens[0][0]
        lbl = make_label(*lbl)

        arr = np.array([k[1] for k in eigens])
        arr_mean = arr.mean(0)
        bars.append((lbl, get_cummax(arr)))
        plt.errorbar(x=list(range(arr.shape[-1])), y=arr_mean, yerr=arr.std(0), label=lbl, linestyle=ls, linewidth=1.5)
    axes = plt.gca()
    axes.set_ylim([0, 10])
    plt.ylabel('Eigenvalue')
    plt.xlabel('Node')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    plt.figure(dpi=300)

    plt.subplot(111)
    plt.xlabel('CSG')
    lbls, ratios = zip(*sorted(bars, key=lambda k: k[1][0]))
    ind = np.arange(len(lbls))
    plt.barh(ind, np.array(ratios)[:, 0], xerr=np.array(ratios)[:, 1], log=False)
    plt.yticks(ind, lbls)
    plt.tight_layout()
    plt.show()
