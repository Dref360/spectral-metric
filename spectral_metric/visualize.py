from typing import List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties

LOOP = 30
LONG_LABEL = False

cmap = sns.cubehelix_palette(light=0.9, as_cmap=True)
font = FontProperties()
font.set_family("sans-serif")


def make_graph(difference: np.ndarray, title: str, classes: List[str]):
    """
    Plot the graph of `ds_name`
    Args:
        difference: 1 - D matrix computed
        title: str: name of dataset
        classes: Class names.
    """
    from sklearn import manifold

    difference = difference
    mds = manifold.MDS(
        n_components=2,
        max_iter=1000,
        eps=1e-9,
        random_state=1337,
        dissimilarity="precomputed",
        n_jobs=1,
    )
    font.set_size(16)

    pos = mds.fit(difference).embedding_
    _ = plt.figure(1, dpi=300)
    ax = plt.axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.set_title(title)
    s = 100
    plt.scatter(pos[:, 0], pos[:, 1], color="turquoise", s=s, lw=0, label="MDS")
    for i, p in enumerate(pos):
        ha = "left" if p[0] < 0 else "right"
        ax.annotate(classes[i], (p[0], p[1]), fontproperties=font, ha=ha)
    similarities = difference.max() / difference * 100
    similarities[np.isinf(similarities)] = 0
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[pos[i, :], pos[j, :]] for i in range(len(pos)) for j in range(len(pos))]
    values = np.abs(similarities)
    lc = LineCollection(segments, zorder=0, cmap="Blues", norm=plt.Normalize(0, values.max()))
    lc.set_array(similarities.flatten())
    lc.set_linewidths(1.0 * np.ones(len(segments)))
    ax.add_collection(lc)
    plt.title(title)
    plt.tight_layout()
    plt.show()
