import tensorflow as tf

from TensorFlow_model.segformer import SegFormer_B2
from Utils import callback, lr_schedular, plot
from Utils.dataset import CityScapes
from Utils.utils import *

tf.random.set_seed(29)
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True  # 设置动态分配 GPU 内存
sess = tf.compat.v1.Session(config=config)

batch_size = 16
epoch = 250
image_shape = (256, 512, 3)
dataset = CityScapes(image_shape=image_shape, batch_size=batch_size, one_hot=True)
(train_dataset, val_dataset) = dataset.load_data()

model = SegFormer_B2(input_shape=image_shape, num_classes=34)


lr_base = 1e-4


total_steps = int(len(train_dataset) * epoch)
warmup_epoch_percentage = 0.1
warmup_steps = int(total_steps * warmup_epoch_percentage)
scheduled_lrs = lr_schedular.WarmUpCosine(lr_base, total_steps, 0, warmup_steps)

optimizer = tf.keras.optimizers.Lion(scheduled_lrs)


tracker = callback.LearningRateTracker()
callbacks = [tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss'),
             tracker,
             tf.keras.callbacks.ModelCheckpoint('Checkpoints/segformer/segformerb2_256x512.h5', 
                                                save_best_only=True, verbose=1, monitor='val_loss')]
loss = tf.keras.losses.KLDivergence()

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
history =  model.fit(train_dataset, validation_data=val_dataset, callbacks=callbacks, epochs=epoch, workers=8)
plot.loss_plot(history, 'Checkpoints/segformer/', lr_tracker=tracker)

