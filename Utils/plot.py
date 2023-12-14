import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Utils.callback import *


def loss_plot(history:tf.keras.callbacks.History, path:str, withlr=False, lr_tracker:LearningRateTracker=None):
    """Plot loss and metrics.

    Args:
        history: tf.keras.callbacks.History object.
        path: Save plot path.
    """
    name = np.array(list(history.history.keys()))
    if withlr:
        lr, name = name[-1], name[:-1]
        [name, val_name] = np.split(name, int(len(name) / 2))
        for i in range(len(name)):
            plt.figure(i)
            plt.plot(history.history[name[i]])
            plt.plot(history.history[val_name[i]])
            plt.title(name[i])
            plt.ylabel(name[i])
            plt.xlabel('epochs')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(path + name[i] + '.png')
        plt.figure(len(name))
        plt.plot(history.history[lr])
        plt.title(lr)
        plt.ylabel('learning rate')
        plt.xlabel('epochs')
        plt.savefig(path + 'learning_rate.png')
        plt.show()
    elif lr_tracker:
        [name, val_name] = np.split(name, int(len(name) / 2))
        for i in range(len(name)):
            plt.figure(i)
            plt.plot(history.history[name[i]])
            plt.plot(history.history[val_name[i]])
            plt.title(name[i])
            plt.ylabel(name[i])
            plt.xlabel('epochs')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(path + name[i] + '.png')
        plt.figure(len(name))
        plt.plot(lr_tracker.lr_arr)
        plt.title('LearningRateTracker')
        plt.ylabel('learning rate')
        plt.xlabel('epochs')
        plt.savefig(path + 'learning_rate.png')
        plt.show()
    else:
        [name, val_name] = np.split(name, int(len(name) / 2))
        for i in range(len(name)):
            plt.figure(i)
            plt.plot(history.history[name[i]])
            plt.plot(history.history[val_name[i]])
            plt.title(name[i])
            plt.ylabel(name[i])
            plt.xlabel('epochs')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(path + name[i] + '.png')
        plt.show()
        
