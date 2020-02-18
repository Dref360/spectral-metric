import os

import keras.backend as K
from keras import Input
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, ZeroPadding2D, BatchNormalization

from spectral_metric.embedding.common import GenericAutoEncoder, EmbeddingGetter
from spectral_metric.embedding.utils import need_sequence
from spectral_metric.handle_datasets import all_datasets, paper_dataset

pjoin = os.path.join


def model_fn(input_shape, force=False):
    """
    Train an AE
    Args:
        input_shape: tuple, [?,width, height, 3]
        force: boolean, whether to ignore the output shape

    Returns: Layer, the embdedding layer.

    """

    inp = Input(input_shape[1:])
    x = Conv2D(64, 3, padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)  # 32
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)  # 16
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)  # 8
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)  # 4
    embd = Conv2D(8, 1, padding='same', name='embd')(x)
    x = Conv2DTranspose(128, 2, strides=2, padding='same')(embd)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(256, 2, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(512, 2, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(512, 2, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(input_shape[-1], 1, padding='same')(x)
    if K.int_shape(x)[1:] != input_shape[1:] and not force:
        x = ZeroPadding2D(2)(x)
    return inp, embd, x


class CNNAutoencoderEmbd(EmbeddingGetter):
    def get_embedding(self, dat):
        if need_sequence(dat):
            return GenericAutoEncoder(model_fn,
                                      flatten=False,
                                      normalize=True).fit_generator(dat,
                                                                    batch_size=32,
                                                                    size=128).reshape([dat.shape[0], -1])
        else:
            return GenericAutoEncoder(model_fn,
                                      flatten=False,
                                      normalize=True).fit(dat,
                                                          batch_size=128).reshape([dat.shape[0], -1])

    @classmethod
    def get_embedding_name(cls):
        return 'cnn_embd'


def main():
    for k in paper_dataset:
        print(k)
        (data, target), _ = all_datasets[k]()
        CNNAutoencoderEmbd()(data, k)
        K.clear_session()


if __name__ == '__main__':
    main()
