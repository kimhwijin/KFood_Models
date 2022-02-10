from tensorflow import keras
from application.inception import *
from application.squeeze_excitation_block import *
from application.WEIGHTS import KerasInceptionResNetV2_Weights

def KerasInceptionResNetV2SEBlock(input_shape=[299, 299, 3], n_classes=150, squeeze_ratio=4):
    img_input = keras.layers.Input(shape=input_shape)

    weights = KerasInceptionResNetV2_Weights()

    x = stem(img_input, weights.STEM)

    #35 x 35 x 320
    channel = keras.backend.int_shape(x)[-1]
    for _ in range(10):
        x = block35(x, weights.BLOCK35)

    x = se_block(x, channel, squeeze_ratio)
    x = reduction_A(x, weights.REDUCTION_A)

    #17 x 17 x 1088
    channel = keras.backend.int_shape(x)[-1]
    for _ in range(20):
        x = block17(x, weights.BLOCK17)

    x = se_block(x, channel, squeeze_ratio)
    x = reduction_B(x, weights.REDUCTION_B)

    #8 x 8 x 2080
    channel = keras.backend.int_shape(x)[-1]
    for _ in range(9):
        x = block8(x, weights.BLOCK8)
    x = se_block(x, channel, squeeze_ratio)
    
    x = block8(x, filters=weights.BLOCK8, scale=1., activation=None)
    x = se_block(x, channel, squeeze_ratio)
    
    #8 x 8 x 1536
    channel = keras.backend.int_shape(x)[-1]
    x = conv2d_bn(x, channel, '1x1', 's', 1)

    x = keras.layers.GlobalAveragePooling2D()(x)
    output = keras.layers.Dense(n_classes, activation='softmax')(x)

    return keras.models.Model(inputs=[img_input], outputs=[output])
