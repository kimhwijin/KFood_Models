from tensorflow import keras
from application.custom_layers import *

def stem(x, filters):

    #299x299x3
    #32
    x = conv2d_bn(x, filters['conv0'], '3x3', 'v', 2) #149x149x32

    #32
    x = conv2d_bn(x, filters['conv1'], '3x3', 'v', 1)
    #64
    x = conv2d_bn(x, filters['conv2'], '3x3', 's', 1)

    branch_pool = max_pool2d(x, '3x3', 'v', 2)
    #96
    branch_0 = conv2d_bn(x, filters['branch0_0'], '3x3', 'v', 2)
    branches = [branch_pool, branch_0]

    x = keras.layers.Concatenate(axis=-1)(branches) #73x73x160


    #64
    branch_0 = conv2d_bn(x, filters['branch1_0'], '1x1', 's', 1)
    #96
    branch_0 = conv2d_bn(branch_0, filters['branch1_1'], '3x3', 'v', 1)

    #64
    branch_1 = conv2d_bn(x, filters['branch2_0'], '1x1', 1)
    branch_1 = conv2d_bn(branch_1, filters['branch2_1'], '7x1', 's', 1)
    branch_1 = conv2d_bn(branch_1, filters['branch2_2'], '1x7', 's', 1)
    #96
    branch_1 = conv2d_bn(branch_1, filters['branch2_3'], '3x3', 'v', 1)

    branches = [branch_0, branch_1]
    x = keras.layers.Concatenate(axis=-1)(branches) #71x71x192

    #192
    branch_0 = conv2d_bn(x, filters['branch3_0'], '3x3', 'v', 2)

    branch_1 = max_pool2d(x, '2x2', 'v', 2)
    branches = [branch_0, branch_1]
    
    return keras.layers.Concatenate(axis=-1)(branches) # 35x35x384


def block35(x, filters, scale=0.17, activation='relu'):
    
    #32
    branch_0 = conv2d_bn(x, filters['branch0'], '1x1', 's', 1)

    #32
    branch_1 = conv2d_bn(x, filters['branch1_0'], '1x1', 's', 1)
    #32
    branch_1 = conv2d_bn(branch_1, filters['branch1_1'], '3x3', 's', 1)

    #32
    branch_2 = conv2d_bn(x, filters['branch2_0'], '1x1', 's', 1)
    #48
    branch_2 = conv2d_bn(branch_2, filters['branch2_1'], '3x3', 's', 1)
    #64
    branch_2 = conv2d_bn(branch_2, filters['branch2_2'], '3x3', 's', 1)

    branches = [branch_0, branch_1, branch_2]

    mixed = keras.layers.Concatenate(axis=-1)(branches)

    skip = x
    shape = keras.backend.int_shape(skip)

    #384    
    up = conv2d_bn(mixed, shape[3], '1x1', 's', 1, activation=None, use_bias=True)
    
    x = keras.layers.Lambda(
        lambda inputs: inputs[0] + inputs[1] * scale,
        output_shape=shape[1:],
    )([skip, up])

    if activation is not None:
        x = keras.layers.Activation(activation)(x)

    return x

def block17(x, filters, scale=0.1, activation='relu'):

    #192
    branch_0 = conv2d_bn(x, filters['branch0'], '1x1', 's', 1)

    #128
    branch_1 = conv2d_bn(x, filters['branch1_0'], '1x1', 's', 1)
    #160
    branch_1 = conv2d_bn(branch_1, filters['branch1_1'], '1x7', 's', 1)
    #192
    branch_1 = conv2d_bn(branch_1, filters['branch1_2'], '7x1', 's', 1)

    branches = [branch_0, branch_1]

    mixed = keras.layers.Concatenate(axis=-1)(branches)

    skip = x
    shape = keras.backend.int_shape(skip)

    up = conv2d_bn(mixed, shape[3], '1x1', 's', 1, activation=None, use_bias=True)
    
    x = keras.layers.Lambda(
        lambda inputs: inputs[0] + inputs[1] * scale,
        output_shape=shape[1:],
    )([skip, up])

    if activation is not None:
        x = keras.layers.Activation(activation)(x)

    return x # 17x17x1152

def block8(x, filters, scale=0.2, activation='relu'):

    #192
    branch_0 = conv2d_bn(x, filters['branch0'], '1x1', 's', 1)

    #192
    branch_1 = conv2d_bn(x, filters['branch1_0'], '1x1', 's', 1)
    #224
    branch_1 = conv2d_bn(branch_1, filters['branch1_1'], '1x3', 's', 1)
    #256
    branch_1 = conv2d_bn(branch_1, filters['branch1_2'], '3x1', 's', 1)

    branches = [branch_0, branch_1]

    mixed = keras.layers.Concatenate(axis=-1)(branches)

    skip = x
    shape = keras.backend.int_shape(skip)

    up = conv2d_bn(mixed, shape[3], '1x1', 's', 1, activation=None, use_bias=True)
    
    x = keras.layers.Lambda(
        lambda inputs: inputs[0] + inputs[1] * scale,
        output_shape=shape[1:],
    )([skip, up])

    if activation is not None:
        x = keras.layers.Activation(activation)(x)
    return x #8x8x2144

def reduction_A(x, filters):
    
    #384
    branch_0 = conv2d_bn(x, filters['branch0'], '3x3', 'v', 2)

    #256
    branch_1 = conv2d_bn(x, filters['branch1_0'], '1x1', 's', 1)
    #256
    branch_1 = conv2d_bn(branch_1, filters['branch1_1'], '3x3', 's', 1)
    #384
    branch_1 = conv2d_bn(branch_1, filters['branch1_2'], '3x3', 'v', 2)


    branch_pool = max_pool2d(x, '3x3', 'v', 2)
    
    branches = [branch_0, branch_1, branch_pool]

    return keras.layers.Concatenate(axis=-1)(branches) #17x17x1152


def reduction_B(x, filters):
    
    #256
    branch_0 = conv2d_bn(x, filters['branch0_0'], '1x1', 's', 1)
    #384
    branch_0 = conv2d_bn(branch_0, filters['branch0_1'], '3x3', 'v', 2)

    #256
    branch_1 = conv2d_bn(x, filters['branch1_0'], '1x1', 's', 1)
    #288
    branch_1 = conv2d_bn(branch_1, filters['branch1_1'], '3x3', 'v', 2)


    #256
    branch_2 = conv2d_bn(x, filters['branch2_0'], '1x1', 's', 1)
    #288
    branch_2 = conv2d_bn(branch_2, filters['branch2_1'], '3x3', 's', 1)
    #320
    branch_2 = conv2d_bn(branch_2, filters['branch2_2'], '3x3', 'v', 2)
    
    branch_pool = max_pool2d(x, '3x3', 'v', 2)
    
    branches = [branch_0, branch_1, branch_2, branch_pool]

    return keras.layers.Concatenate(axis=-1)(branches) # 8x8x2144
