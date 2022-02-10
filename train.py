from operator import mod
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from pathlib import Path

class WeightsSaver(keras.callbacks.Callback):
    def __init__(self, weights_save_path, epochs, **kwargs):
        self.epochs = epochs
        self.weights_save_path = weights_save_path
        self.loss = np.array([])
    
    def on_epoch_end(self, epoch, logs={}):

        self.model.save_weights(self.weights_save_path / "epoch:{}_acc:{:.2f}.weights".format(epoch, logs['val_accuracy']))
        self.loss = np.append(self.loss, logs["loss"])
        plt.plot(np.arange(1, epoch+2), self.loss)
        plt.axis([1, self.epochs, 1, self.loss[0] * 1.5])
        plt.savefig(self.weights_save_path / "loss.png", format="png", dpi=300)
        

def train(
    train_set,
    valid_set,
    steps_per_epoch,
    validation_steps,
    epochs=40,
    pretrained=False,
    save_best_weights=True,
    save_weights_per_epoch=True,
    weights_save_path=Path('drive/MyDrive/kfood'),
    train_property={
        'optimizer' : {
            'name' : 'RMSProp',
            'kwargs' : {
                'learning_rate' : 0.001,
                'rho' : 0.9,
                'momentum' : 0.9,
                'epsilon' : 1.0,
                }
        },
        'crop' : 'random',
    },
    model_name='KerasInceptionResNetV2',
    ):

    if model_name=='KerasInceptionResNetV2':
        from application.keras_inception_resnet_v2 import KerasInceptionResNetV2
        model = KerasInceptionResNetV2()
    elif model_name=='KerasInceptionResNetV2SEBlock':
        from application.keras_inception_resnet_v2_se import KerasInceptionResNetV2SEBlock
        model = KerasInceptionResNetV2SEBlock()

    train_property_name = train_property['optimizer']['name'] + train_property['crop']
    weights_save_path = weights_save_path / model_name / train_property_name

    if pretrained:
        model.load_weights(weights_save_path / 'best.weights')

    callbacks = []
    if save_best_weights:
        best_weights_saver = keras.callbacks.ModelCheckpoint(
            filepath=weights_save_path / 'best.weights',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
        )
        callbacks.append(best_weights_saver)

    if save_weights_per_epoch:
        weights_saver = WeightsSaver(weights_save_path=weights_save_path, epochs=epochs)
        callbacks.append(weights_saver)


    if train_property['optimizer']['name'] == 'SGD':
        optimizer = keras.optimizers.SGD(**train_property['optimizer']['kwargs'])
    elif train_property['optimizer']['name'] == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(**train_property['optimizer']['kwargs'])
    elif train_property['optimizer']['name'] == 'Adam':
        optimizer = keras.optimizers.Adam(**train_property['optimizer']['kwargs'])


    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model.fit(train_set, steps_per_epoch=steps_per_epoch,
            validation_data=valid_set, validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks,
    )
        