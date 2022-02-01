from xml.sax.xmlreader import InputSource
from tensorflow import keras
import tensorflow as tf
from application.inception import *
from application.WEIGHTS import *

def InceptionResNetV2(input_shape=[299, 299, 3], n_classes=150):
    img_input = keras.layers.Input(shape=input_shape)

    weights = InceptionResNetV2_Weights()

    x = Stem(weights.STEM)(img_input)

    #35 x 35 x 320
    for _ in range(10):
        x = Block35(weights.BLOCK35)(x)
    
    x = ReductionA(weights.REDUCTION_A)(x)

    #17 x 17 x 1088
    for _ in range(20):
        x = Block17(weights.BLOCK17)(x)

    x = ReductionB(weights.REDUCTION_B)(x)

    #8 x 8 x 2080
    for _ in range(9):
        x = Block8(weights.BLOCK8)(x)
    
    x = Block8(filters=weights.BLOCK8, scale=1., activation=None)(x)

    #8 x 8 x 1536
    x = conv2d_bn(1536, '1x1', 's', 1)(x)

    x = avg_pool2d()(x)
    output = keras.layers.Dense(n_classes, activation='softmax')(x)

    return keras.models.Model(inputs=[img_input], outputs=[output])
