# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

from embedding.common import EmbeddingGetter
from handle_datasets import all_datasets, paper_dataset


def tsne_inner(x_train):
    data = x_train.reshape([x_train.shape[0], -1])
    data = data / data.max()
    tsne = TSNE(n_components=3, random_state=0, verbose=1, n_jobs=2, n_iter=10000)
    Y = tsne.fit_transform(data)
    return Y


class TSNEEmbedding(EmbeddingGetter):
    """Get t-SNE embedding"""

    def get_embedding(self, dat):
        return tsne_inner(dat)

    @classmethod
    def get_embedding_name(cls):
        return 'tsne'


if __name__ == '__main__':
    for ds_name in paper_dataset:
        (x_train, y_train), _ = all_datasets[ds_name]()
        TSNEEmbedding()(x_train, ds_name)
