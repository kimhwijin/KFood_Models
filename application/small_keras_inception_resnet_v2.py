from xml.sax.xmlreader import InputSource
from tensorflow import keras
import tensorflow as tf
from application.keras_inception import *
from application.FILTERS import KerasInceptionResNetV2_Filters

def SmallKerasInceptionResNetV2(input_shape=[299, 299, 3], n_classes=150):
    img_input = keras.layers.Input(shape=input_shape)

    weights = KerasInceptionResNetV2_Filters()

    x = stem(img_input, weights.STEM)

    #35 x 35 x 320
    for _ in range(3):
        x = block35(x, weights.BLOCK35)
    
    x = reduction_A(x, weights.REDUCTION_A)

    #17 x 17 x 1088
    for _ in range(7):
        x = block17(x, weights.BLOCK17)

    x = reduction_B(x, weights.REDUCTION_B)

    #8 x 8 x 2080
    for _ in range(3):
        x = block8(x, weights.BLOCK8)
    
    x = block8(x, filters=weights.BLOCK8, scale=1., activation=None)

    #8 x 8 x 1536
    x = conv2d_bn(x, 512, '1x1', 's', 1)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.8)(x)

    output = keras.layers.Dense(n_classes, activation='softmax')(x)

    return keras.models.Model(inputs=[img_input], outputs=[output])
