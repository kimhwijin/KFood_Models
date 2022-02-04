from tensorflow import keras
import tensorflow as tf

def se_block(x, filters, ratio=16):

    shape = keras.backend.int_shape(x)

    squeeze = keras.layers.GlobalAveragePooling2D()(x)

    excitation = keras.layers.Dense(filters // ratio, activation='relu')(squeeze)
    
    scale = keras.layers.Dense(filters, activation='sigmoid')(excitation) 
    scale = tf.reshape(scale, [-1, 1, 1, shape[-1]])

    return scale * x
    
    



class SE_Block(keras.layers.Layer):
    def __init__(self, filters, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.global_avg_pool = keras.layers.GlobalAveragePooling2D()
        self.squeeze = keras.layers.Dense(filters//ratio, activation='relu')
        self.excitation = keras.layers.Dense(filters, activation='sigmoid')

    def call(self, inputs):
        Z = self.global_avg_pool(inputs)
        Z = self.squeeze(Z)
        return self.excitation(Z)