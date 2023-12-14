from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def timer_func(func): 
    # This function shows the execution time of  
    # the function object passed 
    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func 

@timer_func
def infer(model, image_tensor, mode='tensorflow'):
    if mode == 'tensorflow':
        predictions = model.predict(np.expand_dims(image_tensor, axis=0))
        predictions = np.squeeze(predictions)
        predictions = np.argmax(predictions, axis=-1)
    elif mode == 'tensorrt':
        # print("The signature keys are: ",list(model.signatures.keys())) 
        inference = model.signatures["serving_default"]
        predictions = inference(tf.constant(np.expand_dims(image_tensor, axis=0) ,dtype=float))['segmentation_output']
        predictions = tf.squeeze(predictions)
        predictions = tf.argmax(predictions, axis=-1).numpy()
        
    return predictions

def predict2mask(predict, colormap, num_classes=34):

    r = np.zeros_like(predict).astype(np.uint8)
    g = np.zeros_like(predict).astype(np.uint8)
    b = np.zeros_like(predict).astype(np.uint8)

    for i in range(num_classes):

        # current_class 是一个布尔数组，形状为 (512, 1024)，属于当前类别的位置将为 True。
        current_class = (predict == i)

        # r_value，是 3 个整数，分别表示当前类别的 RGB 数值。
        r_value, g_value, b_value = colormap[i]

        # 对 mask 中属于当前类别的区域，涂上当前类别的颜色，也就是赋予 RGB 值。
        r[current_class] = r_value
        g[current_class] = g_value
        b[current_class] = b_value

    rgb_mask = np.stack([r, g, b], axis=2)

    return rgb_mask

def get_visualize(image_tensor, rgb_mask):
    image_tensor = (image_tensor + 1) * 127.5
    image_tensor = image_tensor.astype(np.uint8)
    overlay = cv2.addWeighted(image_tensor, 0.2, rgb_mask, 0.8, 0)

    return image_tensor, overlay

def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.utils.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()
