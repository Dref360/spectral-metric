import os
import pickle

import cv2
import keras.backend as K
import numpy as np
from keras import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import Sequence

from config import TMP_DIR
from embedding.utils import resize_all, need_sequence

pjoin = os.path.join


class GenericAutoEncoder(object):
    def __init__(self, model_fn, flatten, normalize):
        """
        Class similar to `tf.Estimator` for autoencoders
        Args:
            model_fn: function which creates the autoencoder.
                (input_shape) -> Input, Embedding, Output tensors
            flatten: whether to train on images or on vectors
            normalize: Normalize inputs with imagenet means/variances.
        """
        self.model_fn = model_fn
        self.flatten = flatten
        self.normalize = normalize

    def fit(self, data, batch_size=32):
        """
        Fit the models on the data
        Args:
            data: ndarray, Input data
            batch_size: int, batch size to use.

        Returns: Embedding of data.

        """
        data, _ = resize_all(data, None)
        if self.normalize:
            data = preprocess_input(data) if data.shape[-1] == 3 else data / 255.
        if self.flatten:
            data = data.reshape([data.shape[0], -1])
        inp, embd, out = self.model_fn(data.shape)
        model = Model(inp, out)
        model.compile('adam', 'mse')
        model.fit(data, data, batch_size=batch_size, epochs=100, shuffle=True)
        model = Model(inp, embd)
        return model.predict(data, batch_size)

    def fit_generator(self, data, batch_size=32, size=64):
        assert need_sequence(data)
        tr_seq = AutoEncoderSequence(data, batch_size, size, self.flatten, self.normalize)
        inp, embd, out = self.model_fn([1, size, size, 3], force=True)
        model = Model(inp, out)
        model.compile('adam', 'mse')
        try:
            model.fit_generator(tr_seq, len(tr_seq) // 2, epochs=50, use_multiprocessing=True, workers=4, shuffle=True)
        except KeyboardInterrupt:
            pass
        model = Model(inp, embd)
        return model.predict_generator(tr_seq, use_multiprocessing=True, workers=4)


class AutoEncoderSequence(Sequence):
    def __init__(self, data, batch_size, size, flatten, normalize):
        """
        Standard Sequence to train an Autoencoder
        Args:
            data (ndarray): Sample input
            batch_size (int):
            size (int): Resize the images
            flatten (bool): Flatten the data
            normalize (bool): Normalize the data
        """
        self.data = data
        self.batch_size = batch_size
        self.size = size
        self.flatten = flatten
        self.normalize = normalize

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        dat = self.data[idx * self.batch_size: (idx + 1) * self.batch_size]
        data, _ = resize_all([cv2.imread(f) for f in dat], None, size=self.size)
        if self.normalize:
            data = preprocess_input(data) if data.shape[-1] == 3 else data / 255.
        if self.flatten:
            data = data.reshape([data.shape[0], -1])
        return data.astype(np.float32), data.astype(np.float32)


class EmbeddingGetter():
    """Abstract Class for all embedding"""

    def __call__(self, dat, ds_name):
        pt = self.get_path(ds_name)
        if os.path.exists(pt):
            return pickle.load(open(pt, 'rb')), self.transform_name(ds_name)
        print("WARNING: Need to compute {} for {}".format(self.get_embedding_name(), ds_name))
        res = self.get_embedding(dat)
        K.clear_session()
        pickle.dump(res, open(pt, 'wb'))
        return res, self.transform_name(ds_name)

    def get_path(self, ds_name):
        """Path where to save the embedding"""
        return pjoin(TMP_DIR, self.transform_name(ds_name) + '.pkl')

    def transform_name(self, ds_name):
        """Add the embedding name to a path"""
        return '{}_{}'.format(ds_name, self.get_embedding_name())

    def get_embedding(self, dat):
        """Computes the embedding"""
        raise NotImplementedError

    @classmethod
    def get_embedding_name(cls):
        raise NotImplementedError


class NoneGetter(EmbeddingGetter):
    """The embedding that does nothing."""

    def get_embedding(self, dat):
        pass

    @classmethod
    def get_embedding_name(cls):
        return None

    def __call__(self, dat, ds_name):
        return dat, ds_name
