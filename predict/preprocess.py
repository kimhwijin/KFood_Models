import numpy as np
import tensorflow as tf
from pathlib import Path
import os

IMAGE_SIZE = (299, 299)
PREDICT_PATH = os.path.join(str(Path(__file__).parent), "predict_images")

def preprocess_to_npy(image_path, target_dir):
    image_name = Path(image_path).name.split('.')[0]
    image = parse_image(image_path)
    crop_image = crop_and_resize(image)
    np.save("{}/{}.npy".format(target_dir, image_name), crop_image)

def preprocess():
    image_names = os.listdir(PREDICT_PATH)
    return [crop_and_resize(parse_image(os.path.join(PREDICT_PATH, image_name))) for image_name in image_names]
    
def parse_image(filepath):
    image = tf.io.read_file(filepath)
    filepath = tf.compat.path_to_str(filepath)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    return image


def crop_and_resize(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]])
    top_crop = (shape[0] - min_dim) // 2
    bottom_crop = shape[0] - top_crop
    left_crop = (shape[1] - min_dim) // 2
    right_crop = shape[1] - left_crop
    image = image[top_crop : bottom_crop, left_crop : right_crop]
    image = tf.image.resize(image, IMAGE_SIZE, method="nearest")
    image = tf.cast(image, tf.float32) / 255.
    return image


def images_to_samples():
    image_paths = os.listdir("predict/test_images")
    for image_path in image_paths:
        preprocess_to_npy("predict/test_images/" + image_path, "predict/test_samples")

if __name__ == "__main__":
    print(PREDICT_PATH)
    #images_to_samples()