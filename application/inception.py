import tensorflow as tf
from tensorflow import keras
from application.custom_layers import *

class Stem(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)

        #299x299x3
        #32
        self.conv0 = conv2d_bn(filters['conv0'], 3, 'v', 2)

        #32
        self.conv1 = conv2d_bn(filters['conv1'], 3, 'v', 1)
        #64
        self.conv2 = conv2d_bn(filters['conv2'], 3, 's', 1)
        self.max_pool3 = max_pool2d(3, 'v', 2)

        #80
        self.conv4 = conv2d_bn(filters['conv4'], 1, 'v', 1)
        #192
        self.conv5 = conv2d_bn(filters['conv5'], 3, 'v', 1)
        self.max_pool6 = max_pool2d(3, 'v', 2)

        #96
        self.branch0 = conv2d_bn(filters['branch0'], 1, 's', 1)
        #48
        self.branch1_0 = conv2d_bn(filters['branch1_0'], 1, 's', 1)
        #64
        self.branch1_1 = conv2d_bn(filters['branch1_1'], 5, 's', 1)
        #64
        self.branch2_0 = conv2d_bn(filters['branch2_0'], 1, 's', 1)
        #96
        self.branch2_1 = conv2d_bn(filters['branch2_1'], 3, 's', 1)
        #96
        self.branch2_2 = conv2d_bn(filters['branch2_2'], 3, 's', 1)

        self.branch_pool_1 = avg_pool2d(3, 's', 1)
        #64
        self.branch_pool_2 = conv2d_bn(filters['branch_pool_2'], 1, 's', 1)

        self.concat = keras.layers.Concatenate(axis=-1)
    
    def call(self, inputs):
        x = inputs
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max_pool6(x)

        branch_0 = self.branch0(x)
        
        branch_1 = self.branch1_0(x)
        branch_1 = self.branch1_1(branch_1)

        branch_2 = self.branch2_0(x)
        branch_2 = self.branch2_1(branch_2)
        branch_2 = self.branch2_2(branch_2)

        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)

        branches = [branch_0, branch_1, branch_2, branch_pool]

        return self.concat(branches)

    
class Block35(keras.layers.Layer):
    def __init__(self, filters, scale=0.17, activation='relu', **kwargs):
        super().__init__(**kwargs)

        self.act = activation
        self.scale = scale

        #32
        self.branch0 = conv2d_bn(filters['branch0'], 1, 's', 1, activation=activation)
        #32
        self.branch1_0 = conv2d_bn(filters['branch1_0'], 1, 's', 1, activation=activation)
        #32
        self.branch1_1 = conv2d_bn(filters['branch1_1'], 3, 's', 1, activation=activation)

        #32
        self.branch2_0 = conv2d_bn(filters['branch2_0'], 1, 's', 1, activation=activation)
        #48
        self.branch2_1 = conv2d_bn(filters['branch2_1'], 3, 's', 1, activation=activation)
        #64
        self.branch2_2 = conv2d_bn(filters['branch2_2'], 3, 's', 1, activation=activation)

        self.concat = keras.layers.Concatenate(axis=-1)

        #layer define
        self.up = conv2d_bn(320, 1, 's', 1, activation=None, use_bias=True)
        self.scale_mix = keras.layers.Lambda(
            lambda inputs: inputs[0] + inputs[1] * self.scale,
            output_shape=[32, 32, 320],
        )


    def call(self, inputs):

        x = inputs

        #call layers
        branch_0 = self.branch0(x)

        branch_1 = self.branch1_0(x)
        branch_1 = self.branch1_1(branch_1)

        branch_2 = self.branch2_0(x)
        branch_2 = self.branch2_1(branch_2)
        branch_2 = self.branch2_2(branch_2)
        
        branches = [branch_0, branch_1, branch_2]

        mixed = self.concat(branches)
        up = self.up(mixed)

        x = self.scale_mix([x, up])

        if self.act is not None:
            x = keras.layers.Activation(self.act)(x)
        return x

class Block17(keras.layers.Layer):
    def __init__(self, filters, scale=0.1, activation='relu', **kwargs):
        super().__init__(**kwargs)

        self.act = activation
        self.scale = scale

        #192
        self.branch0 = conv2d_bn(filters['branch0'], 1, 's', 1)

        #128
        self.branch1_0 = conv2d_bn(filters['branch1_0'], 1, 's', 1)
        #160
        self.branch1_1 = conv2d_bn(filters['branch1_1'], [1,7], 's', 1)
        #192
        self.branch1_2 = conv2d_bn(filters['branch1_2'], [7,1], 's', 1)

        self.concat = keras.layers.Concatenate(axis=-1)

        #layer define
        self.up = conv2d_bn(1088, 1, 's', 1, activation=None, use_bias=True)
        self.scale_mix = keras.layers.Lambda(
            lambda inputs: inputs[0] + inputs[1] * self.scale,
            output_shape=[17, 17, 1088],
        )
        
    def call(self, inputs):
        x = inputs

        branch_0 = self.branch0(x)

        branch_1 = self.branch1_0(x)
        branch_1 = self.branch1_1(branch_1)
        branch_1 = self.branch1_2(branch_1)
        
        branches = [branch_0, branch_1]

        mixed = self.concat(branches)
        up = self.up(mixed)

        x = self.scale_mix([x, up])

        if self.act is not None:
            x = keras.layers.Activation(self.act)(x)

        return x

class Block8(keras.layers.Layer):
    def __init__(self, filters, scale=0.2, activation='relu', **kwargs):
        super().__init__(**kwargs)

        self.act = activation
        self.scale = scale

        #192
        self.branch0 = conv2d_bn(filters['branch0'], 1, 's', 1)

        #192
        self.branch1_0 = conv2d_bn(filters['branch1_0'], 1, 's', 1)
        #224
        self.branch1_1 = conv2d_bn(filters['branch1_1'], [1,3], 's', 1)
        #256
        self.branch1_2 = conv2d_bn(filters['branch1_2'], [3,1], 's', 1)

        self.concat = keras.layers.Concatenate(axis=-1)
        
        #layer define
        #input channels = filters
        self.up = conv2d_bn(2080, 1, 's', 1, activation=None, use_bias=True)
        self.scale_mix = keras.layers.Lambda(
            lambda inputs: inputs[0] + inputs[1] * self.scale,
            output_shape=[8, 8, 2080],
        )
        
    def call(self, inputs):
        x = inputs

        branch_0 = self.branch0(x)

        branch_1 = self.branch1_0(x)
        branch_1 = self.branch1_1(branch_1)
        branch_1 = self.branch1_2(branch_1)
        
        branches = [branch_0, branch_1]

        mixed = self.concat(branches)
        up = self.up(mixed)

        x = self.scale_mix([x, up])

        if self.act is not None:
            x = keras.layers.Activation(self.act)(x)

        return x


class ReductionA(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)

        #384
        self.branch0 = conv2d_bn(filters['branch0'], 3, 'v', 2)

        #256
        self.branch1_0 = conv2d_bn(filters['branch1_0'], 1, 's', 1)
        #256
        self.branch1_1 = conv2d_bn(filters['branch1_1'], 3, 's', 1)
        #384
        self.branch1_2 = conv2d_bn(filters['branch1_2'], 3, 'v', 2)

        self.branch_pool = max_pool2d(3, 'v', 2)

        self.concat = keras.layers.Concatenate(axis=-1)

    
    def call(self, inputs):
        x = inputs

        branch_0 = self.branch0(x)

        branch_1 = self.branch1_0(x)
        branch_1 = self.branch1_1(branch_1)
        branch_1 = self.branch1_2(branch_1)

        branch_pool = self.branch_pool(x)

        branches = [branch_0, branch_1, branch_pool]

        mixed = self.concat(branches)
        return mixed



class ReductionB(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        
        #256
        self.branch0_0 = conv2d_bn(filters['branch0_0'], 1, 's', 1)
        #384
        self.branch0_1 = conv2d_bn(filters['branch0_1'], 3, 'v', 2)

        #256
        self.branch1_0 = conv2d_bn(filters['branch1_0'], 1, 's', 1)
        #288
        self.branch1_1 = conv2d_bn(filters['branch1_1'], 3, 'v', 2)

        #256
        self.branch2_0 = conv2d_bn(filters['branch2_0'], 1, 's', 1)
        #288
        self.branch2_1 = conv2d_bn(filters['branch2_1'], 3, 's', 1)
        #320
        self.branch2_2 = conv2d_bn(filters['branch2_2'], 3, 'v', 2)

        self.branch_pool = max_pool2d(3, 'v', 2)

        self.concat = keras.layers.Concatenate(axis=-1)

    
    def call(self, inputs):
        x = inputs

        branch_0 = self.branch0_0(x)
        branch_0 = self.branch0_1(branch_0)

        branch_1 = self.branch1_0(x)
        branch_1 = self.branch1_1(branch_1)
        
        branch_2 = self.branch2_0(x)
        branch_2 = self.branch2_1(branch_2)
        branch_2 = self.branch2_2(branch_2)

        branch_pool = self.branch_pool(x)

        branches = [branch_0, branch_1, branch_2, branch_pool]

        mixed = self.concat(branches)
        return mixed
