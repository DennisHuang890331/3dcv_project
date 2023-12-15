import argparse
import os

import tensorflow as tf

from TensorFlow_model.segformer import SegFormer_B2
from Utils import callback, lr_schedular, plot
from Utils.dataset import CityScapes, KITTI_segmantation
from Utils.utils import *


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int , default=29, 
                        help='Random seed.')
    parser.add_argument('--batch-size', type=int, default=16 , 
                        help='Training batch size.')
    parser.add_argument('--epoch', type=int, default=250,
                        help='Training epoch.')
    parser.add_argument('--dataset', choices=['Cityscapes', 'KITTI'], default='Cityscapes',
                        help='Trainning dataset.')
    parser.add_argument('--image-dir', type=str, default='',
                        help='Directory of dataset images.')
    parser.add_argument('--label-dir', type=str, default='',
                        help='Directory of dataset images.')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Traning learning rate.')
    parser.add_argument('--warmup-rate', type=float, default=0.1,
                        help='Warm up ratio during training.')
    parser.add_argument('--save-dir', type=str, default='Checkpoints/segformer/',
                        help='Directory for save model.')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stoping patience param.')
    parser.add_argument('--label-smooth', type=float, default=0.5,
                        help='Label smooth param during training.')
    return parser.parse_known_args()[0]

def load_dataset(opt, root):
    CLASSES = 34
    image_shape = (256, 512, 3)
    if opt.dataset == 'Cityscapes':
        colormap = os.path.join(root, 'Utils/colormap_cityscapes.json')
        dataset = CityScapes(
            image_shape=image_shape,
            colormap=str(colormap),
            label_dir=opt.label_dir,
            image_dir=opt.image_dir,
            batch_size=opt.batch_size,
            classes=CLASSES,
            one_hot=True
        )
        (train_dataset, val_dataset) = dataset.load_data()
        return train_dataset, val_dataset
    elif opt.dataset == 'KITTI':
        colormap = os.path.join(root, 'Utils/colormap_cityscapes.json')
        dataset = KITTI_segmantation(
            image_shape=image_shape,
            colormap=colormap,
            label_dir=opt.label_dir,
            image_dir=opt.image_dir,
            batch_size=opt.batch_size,
            classes=CLASSES,
            one_hot=True
        )
        train_dataset = dataset.load_data()
        return train_dataset
    
if __name__ == '__main__':
    opt = parse_opt()
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    tf.random.set_seed(opt.seed)
    config = tf.compat.v1.ConfigProto() 
    config.gpu_options.allow_growth = True  # 设置动态分配 GPU 内存
    sess = tf.compat.v1.Session(config=config)

    CLASSES = 34
    batch_size = opt.batch_size
    epoch = opt.epoch
    image_shape = (256, 512, 3)
    
    model = SegFormer_B2(input_shape=image_shape, num_classes=CLASSES)
    tracker = callback.LearningRateTracker()
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=opt.patience, monitor='val_loss'),
                tracker,
                tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, 'segformerb2_256x512.h5'),
                                                    monitor='val_loss', save_best_only=True, verbose=1)]
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=opt.label_smooth)

    if opt.dataset == 'Cityscapes':
        train_dataset, val_dataset = load_dataset(opt, ROOT_DIR)
        lr_base = opt.lr
        total_steps = int(len(train_dataset) * epoch)
        warmup_epoch_percentage = opt.warmup_rate
        warmup_steps = int(total_steps * warmup_epoch_percentage)
        scheduled_lrs = lr_schedular.WarmUpCosine(lr_base, total_steps, 0, warmup_steps)
        optimizer = tf.keras.optimizers.Lion(scheduled_lrs)
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        history =  model.fit(train_dataset, validation_data=val_dataset, callbacks=callbacks, epochs=epoch, workers=8)
        plot.loss_plot(history, save_dir, lr_tracker=tracker)

    elif opt.dataset == 'KITTI':
        train_dataset = load_dataset(opt, ROOT_DIR)
        train_dataset, val_dataset = load_dataset(opt)
        lr_base = opt.lr
        total_steps = int(len(train_dataset) * epoch)
        warmup_epoch_percentage = opt.warmup_rate
        warmup_steps = int(total_steps * warmup_epoch_percentage)
        scheduled_lrs = lr_schedular.WarmUpCosine(lr_base, total_steps, 0, warmup_steps)
        optimizer = tf.keras.optimizers.Lion(scheduled_lrs)
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        history =  model.fit(train_dataset, callbacks=callbacks, epochs=epoch, workers=8)
