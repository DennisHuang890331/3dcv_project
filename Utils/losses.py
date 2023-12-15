import tensorflow as tf
from keras.src.utils import losses_utils


class MulticlassDiceLoss(tf.losses.Loss):

    def __init__(self,label_smoothing=0 ,reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction, name)
        self.label_smoothing = label_smoothing

    def label_smooth(self, y):
        num_classes = tf.shape(y)[-1]
        return y * (1.0 - self.label_smoothing) + (
            self.label_smoothing / num_classes
        )
    
    def generalized_dice_coefficient(self, y_true, y_pred):
        smooth = 1.
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
        return score      

    def call(self, y_true, y_pred):
        y_true = self.label_smooth(self.label_smoothing)
        return 1 - self.generalized_dice_coefficient(y_true, y_pred)
    
class SSIMLoss(tf.losses.Loss):

    def call(self, y_true, y_pred):
        return 1 - tf.image.ssim(y_true, y_pred, max_val=1)
    
class KL_Dice_Loss(tf.losses.Loss):

    def __init__(self,label_smoothing=0 ,reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction, name)
        self.label_smoothing = label_smoothing

    def generalized_dice_coefficient(self, y_true, y_pred):
        smooth = 1.
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
        return score
    
    def call(self, y_true, y_pred):
        num_classes = len(y_true)
        y_true = y_true * (1.0 - self.label_smoothing) + (
            self.label_smoothing / num_classes
        )
        return tf.keras.losses.kl_divergence(y_true, y_pred) + 1 - self.generalized_dice_coefficient(y_true, y_pred)
    
class KL_Focal_Loss(tf.losses.Loss):

    def __init__(self,label_smoothing=0.0 ,reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction, name)
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        num_classes = len(y_true)
        y_true = y_true * (1.0 - self.label_smoothing) + (
            self.label_smoothing / num_classes
        )
        return tf.keras.losses.categorical_focal_crossentropy(y_true, y_pred) + tf.keras.losses.kl_divergence(y_true, y_pred)

