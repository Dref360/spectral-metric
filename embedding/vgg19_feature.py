import os

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.applications import VGG19
from keras.applications.imagenet_utils import preprocess_input
from keras.engine import Model
from keras.layers import Input, Lambda

from config import read_ds
from embedding.common import EmbeddingGetter

pjoin = os.path.join


class VGGFeatures(EmbeddingGetter):
    def __init__(self):
        self.built = False

    def __exit__(self):
        self.out = None
        self.mod = None
        K.clear_session()

    def _build_model(self):
        inp = Input([None, None, 3])
        x = Lambda(lambda k: tf.image.resize_bilinear(k, (224, 244)))(inp)
        mod = VGG19(include_top=False, input_tensor=x, weights='imagenet', pooling='avg')
        self.out = mod.output
        self.mod = Model(inp, self.out)

    @classmethod
    def get_embedding_name(cls):
        return 'vgg'

    def get_embedding(self, dat):

        if len(dat.shape) == 3:
            dat = np.expand_dims(dat, -1)
        if dat.shape[-1] == 1:
            dat = np.array([cv2.cvtColor(k, cv2.COLOR_GRAY2RGB) for k in dat])
        dat = preprocess_input(dat.astype(np.float32))
        if not self.built:
            self._build_model()
            self.built = True

        res = self.mod.predict(dat, 5, verbose=1)
        return res


if __name__ == '__main__':
    from handle_datasets import paper_dataset

    v = VGGFeatures()
    for k in paper_dataset:
        print(k)
        pred = v(read_ds(k)[0][0], k)
