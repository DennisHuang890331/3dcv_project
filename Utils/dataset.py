import json
import os
import pickle
from glob import glob

import numpy as np
import tensorflow as tf


class Cifar10:
    """Load cifar10 dataset binary files class."""

    def __init__(self, dir='/home/ivlab/dennis_ws/dataset/cifar10/') -> None:
        """Initialize binary files location.
            The cifar 10 dataset should have following files.
            data_batch_1, data_batch_2, ..., data_batch_5, test_batch files.
            Args:
                dir: cifar10 dataset localtion.
        """
        self.train_files = [dir + 'data_batch_' + str(var) for var in range(1, 6)]
        self.test_file = [dir + 'test_batch']
        self.hashmap = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                         'dog', 'frog', 'horse', 'ship', 'truck']

    def unpickle(self, file):
        """Unpickle the binary files.
            Args:
                file: binary file location.

            Returns:
                Tuple of list arrays: `labels, filenames, filenames`
                ***labels***: data labels
                ***filenames***: image file names
                ***data***: image data, shape=(len(data), 32, 32, 3)
        """
        with open(file, 'rb') as fo:
            gz_dict = pickle.load(fo, encoding='bytes')
        return gz_dict[b'labels'],gz_dict[b'filenames'],gz_dict[b'data']
    
    def load_data(self):
        """Load cifar10 dataset data.
        The classes are:

        | Label | Description |
        |:-----:|-------------|
        |   0   | airplane    |
        |   1   | automobile  |
        |   2   | bird        |
        |   3   | cat         |
        |   4   | deer        |
        |   5   | dog         |
        |   6   | frog        |
        |   7   | horse       |
        |   8   | ship        |
        |   9   | truck       |

        Returns:
            Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.

            **x_train**: uint8 NumPy array of grayscale image data with shapes
            `(50000, 32, 32, 3)`, containing the training data. Pixel values range
            from 0 to 255.

            **y_train**: uint8 NumPy array of labels (integers in range 0-9)
            with shape `(50000, 1)` for the training data.

            **x_test**: uint8 NumPy array of grayscale image data with shapes
            `(10000, 32, 32, 3)`, containing the test data. Pixel values range
            from 0 to 255.

            **y_test**: uint8 NumPy array of labels (integers in range 0-9)
            with shape `(10000, 1)` for the test data.
        """
        x_train = np.zeros((1,32,32,3))
        y_train = np.zeros((1))
        for file in self.train_files:
            labels, names, data = self.unpickle(file)
            data = np.reshape(data,(len(labels), 3, 32, 32))
            data = np.rollaxis(data, 1, 4)
            x_train = np.concatenate((x_train, data), axis=0)
            y_train = np.concatenate((y_train, labels), axis=0)
        x_train = np.delete(x_train, 0, axis=0)
        y_train = np.delete(y_train, 0)

        for file in self.test_file:
            y_test, names, x_test = self.unpickle(file)
            y_test = np.array(y_test)
            x_test = np.reshape(x_test,(len(y_test), 3, 32, 32))
            x_test = np.rollaxis(x_test, 1, 4)
        
        return (x_train, y_train), (x_test, y_test)

class CIHP:
    def __init__(self, 
                 image_size=512,
                 batch_size=64, 
                 train_dir='/home/ivlab/dennis_ws/dataset/CIHP/instance-level_human_parsing/Training',
                 val_dir='/home/ivlab/dennis_ws/dataset/CIHP/instance-level_human_parsing/Validation',
                 test_dir='/home/ivlab/dennis_ws/dataset/CIHP/instance-level_human_parsing/Testing',) -> None:
        self.image_size = image_size
        self.num_classes = 20
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.method = tf.image.ResizeMethod.NEAREST_NEIGHBOR


    def read_image(self, path, mask=False):
        image = tf.io.read_file(path)
        if mask:
            image = tf.image.decode_png(image, channels=1)
            image.set_shape([None, None, 1])
            image = tf.image.resize(images=image, size=[self.image_size, self.image_size],
                                    method=self.method )
        else:
            image = tf.image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
            image = tf.image.resize(images=image, size=[self.image_size, self.image_size],
                                    method=self.method )
            image = tf.keras.applications.resnet50.preprocess_input(image) 
        return image
    
    def read_test_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[self.image_size, self.image_size],
                                method=self.method)
        return image

    def helper(self, image_list, mask_list):
        image = self.read_image(image_list)
        mask = self.read_image(mask_list, True)
        return image, mask

    def gernerate_dataset(self, image_list, mask_list):
        dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
        dataset = dataset.map(self.helper, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset

    def load_data(self):
        train_images = sorted(glob(os.path.join(self.train_dir, "Images/*")))
        train_masks = sorted(glob(os.path.join(self.train_dir, "Category_ids/*")))
        val_images = sorted(glob(os.path.join(self.val_dir, "Images/*")))
        val_masks = sorted(glob(os.path.join(self.val_dir, "Category_ids/*")))
        test_images = sorted(glob(os.path.join(self.test_dir, "Images/*")))
        train_dataset = self.gernerate_dataset(train_images, train_masks)
        val_dataset = self.gernerate_dataset(val_images, val_masks)
        test_images = tf.data.Dataset.from_tensor_slices(test_images)
        test_images = test_images.map(self.read_test_image, num_parallel_calls=tf.data.AUTOTUNE)
        return (train_dataset, val_dataset, test_images)
    
class CityScapes:

    def __init__(self, 
                 json_path='/home/ivlab/dennis_ws/dataset/cityscapes/colormap_cityscapes.json',
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
    
