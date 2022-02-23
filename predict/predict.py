import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from predict.preprocess import preprocess
from tensorflow import keras
import tensorflow as tf
from application.small_keras_inception_resnet_v2 import SmallKerasInceptionResNetV2
import numpy as np

MODEL_PATH = os.path.join(str(Path(__file__).parent.parent), "model", "best", "best.weights")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.getcwd(), "drive", "MyDrive", "Model", "kfood", "best", "best.weights")

model = SmallKerasInceptionResNetV2()
model.load_weights(MODEL_PATH)

LABELS = []
CLASSES = []
with open(os.path.join(str(Path(__file__).parent.parent),'class_to_label.txt'),'r', encoding='utf8') as f:
    for line in f.readlines():
        _label, _class = line.strip().split(',')
        LABELS.append(int(_label))
        CLASSES.append(_class)
LABELS = np.array(LABELS)
CLASSES = np.array(CLASSES)

def predict():
    images = preprocess() # n x 299 x 299 x 3
    print(images.shape)
    predicts = model.predict(images)  # n x 150
    labels = np.argmax(predicts, axis=1) # n
    return images, CLASSES[labels]

