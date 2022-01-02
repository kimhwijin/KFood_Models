import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32
#데이터셋
DATASET_NAME = 'kfood'
DRIVE_PATH = Path(os.getcwd())
DATASET_PATH = DRIVE_PATH / DATASET_NAME
filepath = DATASET_PATH
print(filepath.exists())

#class_label 매칭 딕셔너리로 저장
labels = None
classes = None
with open(filepath / 'class_label.csv','r') as f:
    w = csv.reader(f)
    classes = w.__next__()
    labels = w.__next__()
#print(len(classes), len(labels), classes, labels)
class_to_label = {}
for _class, _label in zip(classes, labels):
    class_to_label[_class] = int(_label)
tf_class_to_label = tf.constant(list(class_to_label.keys()))

def get_image_crop_points(filepath):
    crops = {}
    properties = filepath / "crop_area.properties"
    with open(properties, 'r') as p:
        for row in p:
            name, crop = row.replace("\n", "").replace(" ", "").split("=")
            if name != "" and crop != "":
                #name = name.encode('utf-8')
                crop = crop.split(",")
                if len(crop) >= 4:
                    crop = [int(crop[1]), int(crop[0]), int(crop[3]), int(crop[2])]
                    crops[name] = crop
                elif len(crop) == 2:
                    crop = [0, 0, int(crop[1]), int(crop[0])]
                    crops[name] = crop
            
    return crops

#crop 지점 정보 빼오기
crop_points = {}
class_list = list(filepath.glob("*/*"))
class_list = [class_name for class_name in class_list if class_name.is_dir()]
for class_name in class_list:
    crop_points.update(get_image_crop_points(class_name))

tf_crop_image_names = tf.constant(list(crop_points.keys()), dtype=tf.string)
tf_crop_points = tf.constant(list(crop_points.values()))

#데이터셋의 이미지 경로 및 레이블 저장
from glob import glob
image_paths = sorted(glob("kfood/*/*/*"))
image_paths = [image_path for image_path in image_paths if image_path.split("/")[-1].split(".")[-1].lower() in ("png", "jpg", "jpeg")]
#labels = [class_to_label[Path(image_path).parent.stem] for image_path in image_paths]
print(len(image_paths))#, len(labels))


def parse_and_crop_image_add_label(tf_filepath):
    
    image = tf.io.read_file(tf_filepath) # 이미지 파일 읽기
    filepath = tf.compat.path_to_str(tf_filepath)
    #format decoding
    image_format = tf.strings.lower(tf.strings.split(filepath, ".")[-1])

    if image_format == "jpeg":
        image = tf.image.decode_jpeg(image, channels=3) # JPEG-encoded -> uint8 tensor (RGB format)
    elif image_format == "png":
        image = tf.image.decode_png(image, channels=3, dtype=tf.uint8)
    else:    
        image = tf.image.decode_image(image, channels=3, expand_animations=False)


    #crop
    image_name = tf.strings.split(tf.strings.split(filepath, "/")[-1], ".")[0]
    tf_image_idx = tf.where(tf_crop_image_names == image_name)
    
    #crop 정보가 있으면 크롭
    if tf.reduce_all(tf.not_equal(tf.shape(tf_image_idx), tf.constant((0, 1), dtype=tf.int32))):
        crop_offsets = tf_crop_points[tf.reshape(tf_image_idx, shape=())]
        image = tf.image.crop_to_bounding_box(image, crop_offsets[0], crop_offsets[1], crop_offsets[2], crop_offsets[3])
    
    #labeling
    class_name = tf.strings.split(filepath, "/")[-2]
    tf_class_name_idx = tf.where(tf_class_to_label == class_name)
    try:
        label = tf.reshape(tf_class_name_idx, shape=())
    except:
        label = 0
        print("label error")

    return image, int(label)

def central_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]])
    top_crop = (shape[0] - min_dim) // 2
    bottom_crop = shape[0] - top_crop
    left_crop = (shape[1] - min_dim) // 2
    right_crop = shape[1] - left_crop
    return image[top_crop : bottom_crop, left_crop : right_crop]


def resizing_image(image, label):
    image = central_crop(image)
    image = tf.image.resize(image, [299, 299], method="nearest")
    return image, label

def make_kfood_dataset(filepaths, n_read_threads=5, shuffle_buffer_size=None, n_parse_threads=5, batch_size=32, cache=False):

    filenames_dataset = tf.data.Dataset.from_tensor_slices(filepaths)
    dataset = filenames_dataset.map(parse_and_crop_image_add_label, num_parallel_calls=n_parse_threads)
    dataset = dataset.map(resizing_image, num_parallel_calls=n_parse_threads)
    #dataset = filenames_dataset.map(spa)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)


#데이터셋 생성
#dataset = make_kfood_dataset(image_paths, shuffle_buffer_size=10000, n_parse_threads=tf.data.AUTOTUNE)