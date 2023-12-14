import json
import os
import pickle
from glob import glob

import numpy as np
import tensorflow as tf

    
class CityScapes:

    def __init__(self, 
                 json_path='Utils/colormap_cityscapes.json',
                 label_path='/home/ivlab/dennis_ws/dataset/cityscapes/labelIds/',
                 image_path='/home/ivlab/dennis_ws/dataset/cityscapes/images',
                 mode='labelIds',
                 batch_size=64,
                 channel_first=False,
                 mask_2d=False,
                 one_hot=False,
                 classes=34,
                 image_shape=(256, 512, 3)) -> None:
                 
        with open(json_path, 'r') as f:
            self.color_table = json.load(f) 
        self.name_table = np.array([var[0] for var in self.color_table])
        self.id_table = np.array([var[1] for var in self.color_table])
        self.color_table = np.array([var[-1] for var in self.color_table])

        self.mode = mode
        self.image_path = image_path
        self.label_path = label_path
        self.channel_first = channel_first
        self.mask_2d = mask_2d
        self.onehot = one_hot
        self.classes = classes

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    
    def print_definition(self):
        for (name, id, color) in zip(self.name_table, self.id_table, self.color_table):
            print(f'{name:<20} {id:^4}  {color}')
    
    def read_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[self.image_shape[0], self.image_shape[1]],
                                method=self.method )
        image = tf.cast(image, dtype=tf.float32)
        image = image / 127.5 - 1  # 像素被限制在 [-1, 1] 之间
        if self.channel_first:
            image = tf.transpose(image, (2, 0, 1))
        return image

    def read_mask(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[self.image_shape[0], self.image_shape[1]],
                                method=self.method)
        if self.mask_2d:
            image = tf.squeeze(image)

        if self.onehot:
            image = tf.squeeze(image)
            image = tf.one_hot(image, self.classes)
        return image

    def helper(self, image_list, mask_list):
        image = self.read_image(image_list)
        mask = self.read_mask(mask_list)
        return image, mask

    def gernerate_dataset(self, image_list, mask_list):
        dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
        dataset = dataset.map(self.helper, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def load_data(self):
        train_images = sorted(glob(os.path.join(self.image_path, "train/*")))
        val_images = sorted(glob(os.path.join(self.image_path, "val/*")))

        train_label = sorted(glob(os.path.join(self.label_path, "train/*")))
        val_label = sorted(glob(os.path.join(self.label_path, "val/*")))

        train_dataset = self.gernerate_dataset(train_images, train_label)
        val_dataset = self.gernerate_dataset(val_images, val_label)

        return (train_dataset, val_dataset)

class KITTI_segmantation:

    def __init__(self,
                 json_path='//home/ivlab/dennis_ws/dataset/KITTI/segmantation/data_semantics/colormap_cityscapes.json',
                 dir = '/home/ivlab/dennis_ws/dataset/KITTI/segmantation/data_semantics',
                 batch_size=64,
                 image_shape=(256, 512, 3)) -> None:
        self.json_path = json_path
        self.dir = dir
        self.batch_size = batch_size
        self.image_shape = image_shape

        with open(json_path, 'r') as f:
            self.color_table = json.load(f) 
        self.name_table = np.array([var[0] for var in self.color_table])
        self.id_table = np.array([var[1] for var in self.color_table])
        self.color_table = np.array([var[-1] for var in self.color_table])

        self.method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    
    def print_definition(self):
        for (name, id, color) in zip(self.name_table, self.id_table, self.color_table):
            print(f'{name:<20} {id:^4}  {color}')

    def read_image(self, path, mask=False):
        image = tf.io.read_file(path)
        if mask:
            image = tf.image.decode_png(image, channels=1)
            image.set_shape([None, None, 1])
            image = tf.image.resize(images=image, size=[self.image_shape[0], self.image_shape[1]],
                                    method=self.method )
        else:
            image = tf.image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
            image = tf.image.resize(images=image, size=[self.image_shape[0], self.image_shape[1]],
                                    method=self.method)
            image = tf.cast(image, dtype=tf.float32)
            image = image / 127.5 - 1  # 像素被限制在 [-1, 1] 之间
        return image
    
    def read_test_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[self.image_shape[0], self.image_shape[1]],
                                method=self.method )
        image = tf.cast(image, dtype=tf.float32)
        image = image / 127.5 - 1  # 像素被限制在 [-1, 1] 之间
        return image
    
    def helper(self, image_list, mask_list):
        image = self.read_image(image_list)
        mask = self.read_image(mask_list, True)
        return image, mask

    def gernerate_dataset(self, image_list, mask_list):
        dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
        dataset = dataset.map(self.helper, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def load_data(self):
        train_images = sorted(glob(os.path.join(self.dir, "training/image_2/*")))
        train_label = sorted(glob(os.path.join(self.dir, "training/semantic/*")))
        train_dataset = self.gernerate_dataset(train_images, train_label)
        return train_dataset
    
