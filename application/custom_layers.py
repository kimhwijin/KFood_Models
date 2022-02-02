from tensorflow import keras
import tensorflow as tf


def conv2d_bn(filters, kernel_size, padding='v', strides=1, activation='relu', use_bias=False, **kwargs):

    padding = 'valid' if padding == 'v' else 'same'
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, **kwargs),
    ])
    if not use_bias:
        model.add(keras.layers.BatchNormalization(scale=False))
    if activation:
        model.add(keras.layers.Activation(activation))
    return model

class Conv2D_BN(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='s', strides=1, activation='relu', use_bias=False, **kwargs):
        padding='valid' if padding == 'v' else 'same'
        
        self.conv2d = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, **kwargs),
        self.use_bias = use_bias
        self.activation = activation

    def call(self, inputs):
        x = inputs
        x = self.conv2d(x)
        if not self.use_bias:
            x = keras.layers.BatchNormalization(scale=False)(x)
        if self.activation:
            x = keras.layers.Activation(self.activation)(x)
        return x

def conv2d(filters, kernel_size, padding='v', strides=1, activation='relu', use_bias=False, **kwargs):
    padding = 'valid' if padding == 'v' else 'same'
    return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,activation=activation, use_bias=use_bias, **kwargs)


def max_pool2d(pool_size=2, padding='v', strides=1):
    padding = 'valid' if padding == 'v' else 'same'
    return keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding)

def avg_pool2d(pool_size=2, padding='v', strides=1):
    padding = 'valid' if padding == 'v' else 'same'
    return keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)