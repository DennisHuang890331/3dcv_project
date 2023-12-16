import argparse
import os

import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert

from TensorFlow_model.deeplab_v3plus import *
from TensorFlow_model.segformer import *


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str , required=True , default='', 
                        help='Tensorflow model location.')
    parser.add_argument('--model-type', choices=['Deeplabv3+', 'Segformer'] , required=True , 
                        help='Type of model will convert to Tensorrt. Deeplabv3+ or Segformer')
    parser.add_argument('--export-dir', default='',
                        help='Export directory of Tensorrt model.')
    parser.add_argument('--h5', action='store_true',
                        help='If you load .h5 model.')
    parser.add_argument('--fp16', action='store_true',
                        help='Inference fp16 accuracy default is fp32.')

    return parser.parse_known_args()[0]

def convert_tensorrt(model_dir, tensorrt_dir, fp16=False):
    precision = trt_convert.TrtPrecisionMode.FP32
    if fp16:
        precision = trt_convert.TrtPrecisionMode.FP16

    converter = trt_convert.TrtGraphConverterV2(
        input_saved_model_dir=model_dir,
        precision_mode=precision
    )
    converter.convert()

    def input_fn():
        input_sizes = [[256, 512]]
        for size in input_sizes:
            inp1 = np.random.normal(size=(1, *size, 3)).astype(np.float32)
            yield [inp1]
        converter.build(input_fn=input_fn)
    converter.save(output_saved_model_dir=tensorrt_dir)

if __name__ == '__main__':
    opt = parse_opt()
    fp16 = False
    if opt.fp16:
        fp16 = True
    if opt.model_type == 'Deeplabv3+':
        # Convert deeplab_v3+ to tensorrt model
        SAVEDIR = opt.model
        EXPORTDIR = opt.export_dir
        if not opt.export_dir:
            EXPORTDIR = 'Checkpoints/deeplabv3plus/tensorrt_deeplabv3plus_inception_256x512'
            if fp16: EXPORTDIR += '_FP16'
            else: EXPORTDIR += '_FP32'
            os.makedirs(EXPORTDIR, exist_ok=True)
        if opt.h5:
            model = tf.keras.models.load_model(SAVEDIR, compile=False)
            SAVEDIR = 'Checkpoints/deeplabv3plus/deeplabv3plus_inception_256x512'
            os.makedirs(SAVEDIR, exist_ok=True)
            tf.saved_model.save(model, SAVEDIR)
    elif opt.model_type == 'Segformer':
        # Convert Segformer to tensorrt model
        SAVEDIR = opt.model
        EXPORTDIR = opt.export_dir
        if not opt.export_dir:
            EXPORTDIR = 'Checkpoints/segformer/tensorrt_segformerb2_256x512'
            if fp16: EXPORTDIR += '_FP16'
            else: EXPORTDIR += '_FP32'
            os.makedirs(EXPORTDIR, exist_ok=True)
        if opt.h5:
            model = tf.keras.models.load_model(SAVEDIR, compile=False)
            SAVEDIR = 'Checkpoints/segformer/segformerb2_256x512'
            os.makedirs(SAVEDIR, exist_ok=True)
            tf.saved_model.save(model, SAVEDIR)
    
    convert_tensorrt(model_dir=SAVEDIR, tensorrt_dir=EXPORTDIR, fp16=fp16)
    print("Finish. Your model location: {}".format(EXPORTDIR))