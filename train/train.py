from operator import mod
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from pathlib import Path

class ModelSaver(keras.callbacks.Callback):
    def __init__(self, now, model_paths, epochs, **kwargs):
        self.now = now
        self.epochs = epochs
        self.model_paths = model_paths
        self.loss = np.array([])
    
    def on_epoch_end(self, epoch, logs={}):
        fileformats = [self.now.year, self.now.month, self.now.day, self.now.hour, self.now.minute, epoch]
        self.model.save(self.model_paths + "/{}_{}_{}_{}_{}_{}.hd5".format(*fileformats))
        
        self.loss = np.append(self.loss, logs["loss"])
        plt.plot(np.arange(1, epoch+2), self.loss)
        plt.axis([1, self.epochs, 1, self.loss[0] * 1.5])
        plt.savefig(self.model_paths + "/loss.png", format="png", dpi=300)
        

def train(
    train_set,
    valid_set,
    steps_per_epoch,
    validation_steps,
    pretrained=False,
    save_best_weights=True,
    save_weights_per_epoch=True,
    weights_save_path=Path('drive/MyDrive/kfood'),
    train_property="SGD_random",
    model_name='KerasInceptionResNetV2',
    epochs=40
    ):

    if model_name=='KerasInceptionResNetV2':
        from application.keras_inception_resnet_v2 import KerasInceptionResNetV2
        model = KerasInceptionResNetV2()
    elif model_name=='KerasInceptionResNetV2SEBlock':
        from application.keras_inception_resnet_v2_se import KerasInceptionResNetV2SEBlock
        model = KerasInceptionResNetV2SEBlock()


    weights_save_path = weights_save_path / model_name / train_property

    if pretrained:
        model.load_weights(weights_save_path / 'best')

    if save_best_weights:
        best_weights_saver = keras.callbacks.ModelCheckpoint(
            filepath=weights_save_path / 'best',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
        )
    if save_weights_per_epoch:

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_set, steps_per_epoch=steps_per_epoch,
            validation_data=valid_set, validation_steps=validation_steps,
            epochs=epochs,

    )
        
    