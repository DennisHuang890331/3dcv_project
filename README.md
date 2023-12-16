# Abstract
**This repository serves as a subfunctional block within the final project for the 2023 fall 3DCV lecture at National Taiwan University.**.

The project offers an implementation featuring **Deeplabv3+** with InceptionResNetV2 and ResNet50 backbones, along with the **Segformer** model, designed for the semantic segmentation task using the TensorFlow platform.


Additionally, the project supplies TensorRT compiled code for each model to enhance inference speed through the utilization of [**TensorFlow-TensorRT (TF-TRT)**](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html).

## Getting Started
All models in this project underwent training with the dataset described below.

[**Cityscapes dataset**](https://www.cityscapes-dataset.com/)
or [**KITTI**](https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015)

To get started with training your model, follow the folder structure shown below to prepare the data.
```bash
# Cityscapes dataset folder structure.
.
├── images
│   ├── test
│   │   ├── berlin_000000_000019_leftImg8bit.png
│   │   ...
│   ├── train
│   │   ├── aachen_000000_000019_leftImg8bit.png
│   │   ...
│   └── val
|       ├── frankfurt_000000_000294_leftImg8bit.png
|       ...
└── labelIds
    ├── test
    │   ├── berlin_000000_000019_gtFine_labelIds.png
    │   ...
    ├── train
    │   ├── aachen_000000_000019_gtFine_labelIds.png
    │   ...
    └── val
        ├── frankfurt_000000_000294_gtFine_labelIds.png
        ...
# KITTI pixel-level segmentation dataset folder structure.
.
└── training
    ├── image_2 
    │   ├── 000000_10.png
    |   ...
    └── semantic
        ├── 000000_10.png
        ...
```

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
python3 deeplabv3plus_train.py --dataset # Cityscapes or KITTI.
                               --image-dir # Dataset images directory.
                               --label-dir # Dataset labels directory.

# You can check other training param by the following command.
python3 deeplabv3plus_train.py --help
```
For SegFormer. 
```bash
python3 segformer_train.py --dataset # Cityscapes or KITTI.
                           --image-dir # Dataset images directory.
                           --label-dir # Dataset labels directory.

# You can check other training param by the following command.
python3 deeplabv3plus_train.py --help
```


## TensorFlow-TensorRT (TF-TRT) Acceleration

If you want to use TensorRt to accelerate your model.

**Make sure you have [**TensoRT**](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
[**CUDA**](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
[**cuDNN**](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) in your enviroment.**

To convert your TensorFlow model to TensorRT, use the command shown below.
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
model = tf.keras.models.load_model('your TensorFlow or TensorRT model', compile=False)
image = # your image size must be (256, 512, 3)
# Tensorflow
prediction = infer(model, image, mode='tensorflow')
# TensorRT
prediction = infer(model, image, mode='tensorrt')
```