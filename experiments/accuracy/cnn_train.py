import json
import os
import pickle
import sys

import cv2
import keras.backend as K
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.utils import to_categorical

from accuracy.model_definition import models
from config import TMP_DIR
from embedding.utils import unison_shuffled_copies, split, resize_all, need_sequence
from handle_datasets import all_datasets, paper_dataset, make_small

opt = lambda: SGD(0.001, 0.9, nesterov=True)

pjoin = os.path.join

"""
Model building and training
"""


def train_model(X_train, y_train, ds_name, test_data, model_name='alexnet'):
    """
    Train the selected model on the data
    Args:
        X_train (ndarray):
        y_train (ndarray):
        ds_name (str): Name of the dataset
        test_data (ndarray, ndarray): x_test, y_test
        model_name (str): Name of the model

    Returns: Model trained

    """
    mod = models[model_name](X_train.shape[1:], y_train.shape[-1])

    mod.compile(opt(), 'categorical_crossentropy', metrics=['accuracy'])
    mod.summary()
    batch_size = 32
    try:
        mod.fit(X_train, y_train, batch_size, 100, validation_data=test_data,
                callbacks=[ModelCheckpoint(pjoin(TMP_DIR, ds_name + model_name + '.model'),
                                           monitor='val_acc',
                                           save_best_only=True,
                                           save_weights_only=True),
                           ReduceLROnPlateau(patience=30)],
                shuffle=True)
    except KeyboardInterrupt:
        pass
    return mod


def fit(ds_name, data, target, xval, yval, x_test, y_test, sample_per_class=None, model_name='alexnet'):
    # Resize and make RGB
    data, x_test = resize_all(data, x_test)
    xval, _ = resize_all(xval)

    if data.shape[-1] == 1:
        data = np.array([cv2.cvtColor(k, cv2.COLOR_GRAY2RGB) for k in data])
        x_test = np.array([cv2.cvtColor(k, cv2.COLOR_GRAY2RGB) for k in
                           x_test])
        xval = np.array([cv2.cvtColor(k, cv2.COLOR_GRAY2RGB) for k in xval])

    # Preprocessing
    data = preprocess_input(data) if data.shape[-1] == 3 else data / 255.

    if sample_per_class:
        (data, target), _ = make_small(((data, target), (None, None)), sample_per_class)

    if x_test.shape[-1] == 3:
        x_test = preprocess_input(x_test)
    else:
        x_test = x_test / 255.
    if y_test.shape[-1] == 1:
        y_test = to_categorical(y_test, target.shape[-1])

    # Train the model
    model = train_model(data, target, ds_name, (xval, yval), model_name)

    hist = model.evaluate(x_test, y_test, 8)

    # Logging and write results to disk
    print('finish', hist, ds_name, model_name, sample_per_class)

    model.load_weights(pjoin(TMP_DIR, ds_name + model_name + '.model'))
    model.compile(opt(), 'categorical_crossentropy', metrics=['accuracy'])

    hist = model.evaluate(x_test, y_test, 8)

    # Logging and write results to disk
    print('Best val_acc', hist, ds_name, model_name, sample_per_class)
    with open(pjoin(TMP_DIR, ds_name + model_name + 'res.json'), 'w') as f:
        json.dump(hist, f)

    pickle.dump((y_test, model.predict(x_test, 8)), open(pjoin(TMP_DIR, ds_name + model_name + 'res.pkl'), 'wb'))


"""
Main Script
"""


def job(ds_name, model_name='alexnet', n_sample=None):
    d = all_datasets[ds_name]
    if isinstance(d, tuple):
        f, arg = d
        d = lambda: f(arg)
    (x_train, y_train), test_d = d()
    # Shuffle the dataset
    x_train, y_train = unison_shuffled_copies(np.array(x_train), np.array(y_train))

    if test_d is None:
        # We create a fake test set
        (x_train, y_train), (x_test, y_test) = split(x_train, y_train, 0.8)
    else:
        (x_test, y_test) = test_d

    # Create the validation set
    (data, target), (xval, yval) = split(imgs=x_train, labels=y_train, pc=0.9)
    target = to_categorical(target)
    yval = to_categorical(yval)
    y_test = to_categorical(y_test)


    # For datasets like MIO-TCD we must use fit_generator
    if not need_sequence(data):
        fit(ds_name, data, target, xval, yval, x_test, y_test, n_sample, model_name)
    else:
        from spectral_metric.accuracy.cnn_train_generator import fit_generator
        fit_generator(ds_name, data, target, xval, yval, x_test, y_test, n_sample, model_name)


def main():
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'alexnet'
    for k in paper_dataset:
        n = None
        # Special case where you want less samples (ds_name, n_sample_per_class)
        if len(k) == 2:
            k, n = k
        if '_' in k:
            continue
        if os.path.exists(pjoin(TMP_DIR, k + model_name + 'res.json')):
            continue
        print(k)
        job(k, model_name, n)
        K.clear_session()


if __name__ == '__main__':
    main()
