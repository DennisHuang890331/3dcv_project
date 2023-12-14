## Abstract
**This repository is the subfunctional block in final project for the 2023 fall 3DCV lecture at National Taiwan University**.

This project provides an implementation of **Deeplabv3+** with InceptionV2 ResNet backbone and the **Segformer** model for the semantic segmentation task using the TensorFlow platform.

This project provide TensorRT compile code for each model to accerate inference by using [**TensorFlow-TensorRT (TF-TRT)**](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html).

## Requirement
[**Cityscapes dataset**](https://www.cityscapes-dataset.com/)
or [**KITTI pixel-level Segmentation**](https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015)

**Optional**

If you want to use TensorRt to accelerate your model.
Make sure you have [**TensoRt**](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
[**CUDA**](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
[**cuDNN**](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) in your enviroment.

## Install

Clone repo and install requirements.txt in 
[Python >= 3.10](https://www.python.org/) 
enviroment.

```bash
pip3 install -r requirements.txt
```
## Training
For DeeplabV3+ with Inceptron Resnet Backbone.
```bash
python3 deeplabv3plus_train.py
```
For SegFormer.
```bash
python3 segformer_train.py
```
## TensorFlow-TensorRT (TF-TRT) Acceleration
Use **conver.py** to convert your Tensorflow model to TensorRT.
```bash
python3 convert.py --model  # Tensorflow model location.
                    --model-type # {Deeplabv3+,Segformer} Type of model will convert to Tensorrt. Deeplabv3+ or Segformer
                    --export-dir # Export directory of Tensorrt model.
                    --h5 # If you load .h5 model.
                    --fp16 # Inference fp16 accuracy default is fp32.
```
## Inference
```bash
from Utils.utils import *
model = tf.keras.models.load_model('your TensorFlow or TensorRT model')
image = # your image size must be (256, 512, 3)
# Tensorflow
prediction = infer(model, image, mode='tensorflow')
# TensorRT
prediction = infer(model, image, mode='tensorrt')
```