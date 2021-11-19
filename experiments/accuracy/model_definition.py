import keras
import tensorflow as tf
from keras import Sequential, Input, Model
from keras.initializers import Ones, Zeros, Constant, TruncatedNormal
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda

zinit = Zeros()
oinit = Ones()
ginit = TruncatedNormal(0, 0.015)

cinit = Constant(0.1)


def alexnet(input_shape, num_c):
    """Build AlexNet as a Keras model."""

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    # model.add(Conv2D(96, (11,11), strides=(4,4), activation='relu', padding='same', input_shape=(img_height, img_width, channel,)))
    # for original Alexnet
    model.add(Conv2D(96, (3, 3), activation='relu', padding='same', input_shape=input_shape, bias_initializer=cinit
                     , kernel_initializer=ginit))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # Local Response normalization for Original Alexnet
    # model.add(Lambda(lrn))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=cinit, kernel_initializer=ginit))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # Local Response normalization for Original Alexnet
    # model.add(Lambda(lrn))
    model.add(BatchNormalization())

    model.add(Conv2D(384, (3, 3), activation='relu', padding='same', bias_initializer=cinit, kernel_initializer=ginit))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='same', bias_initializer=cinit, kernel_initializer=ginit))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=cinit, kernel_initializer=ginit))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # Local Response normalization for Original Alexnet
    # model.add(Lambda(lrn))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', bias_initializer=cinit, kernel_initializer=ginit))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', bias_initializer=cinit, kernel_initializer=ginit))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_c, activation='softmax', bias_initializer=cinit, kernel_initializer=ginit))
    return model


def top(x, num_c):
    """
    Top of the model
    Args:
        x (Tensor): Last tensor of the backbone
        num_c (int): Nb Classes

    Returns: Tensor, the softmax tensor

    """
    x = Dense(4096, activation='relu', bias_initializer=cinit, kernel_initializer=ginit)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', bias_initializer=cinit, kernel_initializer=ginit)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_c, activation='softmax', bias_initializer=cinit, kernel_initializer=ginit)(x)
    return x


def xception(input_shape, num_c):
    inp = Input([None, None, input_shape[-1]])
    x = Lambda(lambda x: tf.image.resize_bilinear(x, (224, 224)))(inp)
    x = keras.applications.Xception(input_tensor=x, include_top=False, pooling='avg', weights='imagenet').output
    x = top(x, num_c)
    return Model(inp, x)


def resnet50(input_shape, num_c):
    inp = Input([None, None, input_shape[-1]])
    x = Lambda(lambda x: tf.image.resize_bilinear(x, (224, 224)))(inp)
    x = keras.applications.ResNet50(input_tensor=x, include_top=False, pooling='avg', weights='imagenet').output
    x = top(x, num_c)
    return Model(inp, x)


models = {'alexnet': alexnet,
          'xception': xception,
          'resnet50': resnet50}
