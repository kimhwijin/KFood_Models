from calendar import c


import tensorflow as tf
from tensorflow import keras
from custom_layers import *


class Stem(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)

        #299x299x3
        #32
        self.conv0 = conv2d_bn(filters['conv0'], '3x3', 'v', 2)

        #32
        self.conv1 = conv2d_bn(filters['conv1'], '3x3', 'v', 1)
        #64
        self.conv2 = conv2d_bn(filters['conv2'], '3x3', 's', 1)
        self.max_pool3 = max_pool2d('3x3', 's', 2)

        #80
        self.conv4 = conv2d_bn(filters['conv4'], '1x1', 'v', 1)
        #192
        self.conv5 = conv2d_bn(filters['conv5'], '3x3', 'v', 1)
        self.max_pool6 = max_pool2d('3x3', 's', 2)

        #96
        self.branch0 = conv2d_bn(filters['branch0'], '1x1', 's', 1)
        #48
        self.branch1_0 = conv2d_bn(filters['branch1_0'], '1x1', 's', 1)
        #64
        self.branch1_1 = conv2d_bn(filters['branch1_2'], '5x5', 's', 1)
        #64
        self.branch2_0 = conv2d_bn(filters['branch2_0'], '1x1', 's', 1)
        #96
        self.branch2_1 = conv2d_bn(filters['branch2_1'], '3x3', 's', 1)
        #96
        self.branch2_2 = conv2d_bn(filters['branch2_2'], '3x3', 's', 1)

        self.branch_pool_1 = avg_pool2d('3x3', 's', 1)
        #64
        self.branch_pool_2 = conv2d_bn(filters['branch_pool_2'], '1x1', 's', 1)

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