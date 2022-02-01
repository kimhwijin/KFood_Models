from tensorflow import keras


def conv2d_bn(filters, kernel_size, padding='v', strides=1, activation='relu', use_bias=False, **kwargs):

    padding = 'valid' if padding == 'v' else 'same'
    x, y = kernel_size.split('x')
    kernel_size = [int(x), int(y)]
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, **kwargs),
    ])
    if not use_bias:
        model.add(keras.layers.BatchNormalization(scale=False))
    if activation:
        model.add(keras.layers.Activation(activation))
    return model

def conv2d(filters, kernel_size, padding='v', strides=1, activation='relu', use_bias=False, **kwargs):
    padding = 'valid' if padding == 'v' else 'same'
    x, y = kernel_size.split('x')
    kernel_size = [int(x), int(y)]
    return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,activation=activation, use_bias=use_bias, **kwargs)


def max_pool2d(pool_size='2x2', padding='v', strides=1):
    x, y = pool_size.split('x')
    pool_size = [int(x), int(y)]
    padding = 'valid' if padding == 'v' else 'same'
    return keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding)

def avg_pool2d(pool_size='2x2', padding='v', strides=1):
    x, y = pool_size.split('x')
    pool_size = [int(x), int(y)]
    padding = 'valid' if padding == 'v' else 'same'
    return keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)