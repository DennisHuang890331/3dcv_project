import tensorflow as tf

from TensorFlow_model.deeplab_v3plus import *
from Utils import callback, losses, lr_schedular, plot
from Utils.dataset import CityScapes
from Utils.utils import *

tf.random.set_seed(29)
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True  # 设置动态分配 GPU 内存
sess = tf.compat.v1.Session(config=config)

CLASSES = 34
batch_size = 16
epoch = 50
image_shape = (256, 512, 3)
dataset = CityScapes(image_shape=image_shape, batch_size=batch_size, one_hot=True)
(train_dataset, val_dataset) = dataset.load_data()

model = build_model(image_shape=image_shape, backbone_trainable=True, num_classes=CLASSES,
                    backbone='inceptionresnetv2', rate_dropout=0.2)

lr_base = 1e-5

total_steps = int(len(train_dataset) * epoch)
warmup_epoch_percentage = 0.1
warmup_steps = int(total_steps * warmup_epoch_percentage)
scheduled_lrs = lr_schedular.WarmUpCosine(lr_base, total_steps, 0, warmup_steps)

optimizer = tf.keras.optimizers.Lion(scheduled_lrs, weight_decay=0.001)

tracker = callback.LearningRateTracker()
callbacks = [tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss'),
             tracker,
             tf.keras.callbacks.ModelCheckpoint('Checkpoints/deeplabv3plus/deeplabv3plus_inception_256x512.h5',
                                                monitor='val_loss', save_best_only=True, verbose=1)]
loss = losses.KL_Dice_Loss()

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
history =  model.fit(train_dataset, validation_data=val_dataset, callbacks=callbacks, epochs=epoch, workers=8)
plot.loss_plot(history, 'Checkpoints/deeplabv3plus/', lr_tracker=tracker)


