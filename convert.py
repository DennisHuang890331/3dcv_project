import os
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert

from TensorFlow_model.deeplab_v3plus import *
from TensorFlow_model.segformer import *
from Utils.dataset import CityScapes
from Utils.utils import *


def convert_tensorrt(model_dir, tensorrt_dir):
    converter = trt_convert.TrtGraphConverterV2(
        input_saved_model_dir=model_dir,
        precision_mode=trt_convert.TrtPrecisionMode.FP32
    )
    converter.convert()

    def input_fn():
        input_sizes = [[256, 512]]
        for size in input_sizes:
            inp1 = np.random.normal(size=(1, *size, 3)).astype(np.float32)
            yield [inp1]
        converter.build(input_fn=input_fn)
    converter.save(output_saved_model_dir=tensorrt_dir)

# Convert deeplab_v3+ to tensorrt model
# model = tf.keras.models.load_model('Checkpoints/deeplabv3plus/deeplabv3plus_inception_256x512.h5', compile=False)
# tf.saved_model.save(model, 'Checkpoints/deeplabv3plus/export/deeplabv3plus_inception_256x512')
# convert_tensorrt(model_dir='Checkpoints/deeplabv3plus/export/deeplabv3plus_inception_256x512',
#                  tensorrt_dir="Checkpoints/deeplabv3plus/export/tensorrt/")

# Convert segformer to tensorrt model
model = tf.keras.models.load_model('Checkpoints/segformer/segformerb2_256x512.h5', compile=False)
tf.saved_model.save(model, 'Checkpoints/segformer/export/segformerb2_256x512')

convert_tensorrt(model_dir='Checkpoints/segformer/export/segformerb2_256x512',
                 tensorrt_dir="Checkpoints/segformer/export/tensorrt/")


# test tensorrt
dataset = CityScapes(image_shape=(256, 512, 3), batch_size=16, one_hot=True)

model = tf.keras.models.load_model('Checkpoints/deeplabv3plus/export/tensorrt')
test_path = sorted(glob(os.path.join(dataset.image_path, "test/*")))
index = np.random.choice(len(test_path), 5, False).astype(np.uint8)

test_path = np.array(test_path)[index]
test_img = []

for var in test_path:
    test_img.append(dataset.read_image(var))
test_img = np.array(test_img)
colormap = dataset.color_table[:-1]

for var in test_img:
    pre = infer(model, var, 'tensorrt')
    mask = predict2mask(pre, colormap)
    origin, overlay = get_visualize(var, mask)
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(4, 3))
    axes[0].imshow(origin)
    axes[1].imshow(overlay)
    axes[2].imshow(mask)
    plt.show()

