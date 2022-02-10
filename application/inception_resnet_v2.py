from tensorflow import keras
from application.inception import *
from application.FILTERS import InceptionResNetV2_Filters

def InceptionResNetV2(input_shape=[299, 299, 3], n_classes=150):
    img_input = keras.layers.Input(shape=input_shape)

    filters = InceptionResNetV2_Filters()

    x = stem(img_input, filters.STEM)

    #35 x 35 x 384
    for _ in range(5):
        x = block35(x, filters.BLOCK35)
    
    #35x35x384
    x = reduction_A(x, filters.REDUCTION_A)
    #17x17x1152

    #17x17x1152
    for _ in range(10):
        x = block17(x, filters.BLOCK17)

    #17x17x1152
    x = reduction_B(x, filters.REDUCTION_B)
    #8x8x2144

    #8x8x2144
    for _ in range(5):
        x = block8(x, filters.BLOCK8)
    
    #8x8x2144
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.8)(x)

    output = keras.layers.Dense(n_classes, activation='softmax')(x)

    return keras.models.Model(inputs=[img_input], outputs=[output])
