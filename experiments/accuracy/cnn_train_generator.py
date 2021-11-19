import json
import pickle

import cv2
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import Sequence

from accuracy.cnn_train import opt, pjoin
from accuracy.model_definition import models
from config import TMP_DIR
from embedding.utils import resize_all, unison_shuffled_copies
from handle_datasets import make_small


class LongSeq(Sequence):
    def __init__(self, data, target, batch_size):
        """
        Sequence for huge dataset
        Args:
            data (ndarray): path to image
            target (ndarray): one-hot vector
            batch_size (int):
        """
        self.data = data
        self.target = target
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        dat = self.data[idx * self.batch_size: (idx + 1) * self.batch_size]
        target = self.target[idx * self.batch_size: (idx + 1) * self.batch_size]
        data, _ = resize_all([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in dat], None)
        data = preprocess_input(data) if data.shape[-1] == 3 else data / 255.

        return data.astype(np.float32), np.array(target)


class XGetter(Sequence):
    def __init__(self, s):
        """Get the first element of a Sequence"""
        self.s = s

    def __len__(self):
        return len(self.s)

    def __getitem__(self, item):
        return self.s[item][0]


def fit_generator(ds_name, data, target, xval, yval, x_test, y_test, sample_per_class=None, model_name='alexnet'):
    if sample_per_class:
        (data, target), _ = make_small(((data, target), (None, None)), sample_per_class)
        data,target = unison_shuffled_copies(data,target)

    tr_seq = LongSeq(data, target, 32)
    vl_seq = LongSeq(xval, yval, 32)
    ts_seq = LongSeq(x_test, y_test, 32)
    mod = models[model_name]([64, 64, 3], target.shape[-1])
    mod.compile(opt(), 'categorical_crossentropy', metrics=['accuracy'])
    try:
        mod.fit_generator(tr_seq, len(tr_seq), epochs=100,
                          validation_data=vl_seq,
                          use_multiprocessing=True,
                          workers=4,
                          callbacks=[
                              ModelCheckpoint(pjoin(TMP_DIR, ds_name + '.model'),
                                              monitor='val_acc',
                                              save_best_only=True,
                                              save_weights_only=True),
                              ReduceLROnPlateau(patience=30)],
                          shuffle=True)
    except KeyboardInterrupt:
        pass
    hist = mod.evaluate_generator(ts_seq, use_multiprocessing=True, workers=4)

    # Logging and write results to disk
    print('finished', hist, ds_name, model_name, sample_per_class)

    mod.load_weights(pjoin(TMP_DIR, ds_name + '.model'))
    mod.compile(opt(), 'categorical_crossentropy', metrics=['accuracy'])

    hist = mod.evaluate_generator(ts_seq, use_multiprocessing=True, workers=4)

    # Logging and write results to disk
    print(hist)
    with open(pjoin(TMP_DIR, ds_name + 'res.json'), 'w') as f:
        json.dump(hist, f)

    pickle.dump((y_test, mod.predict_generator(XGetter(ts_seq))), open(pjoin(TMP_DIR, ds_name + 'res.pkl'), 'wb'))