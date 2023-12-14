import tensorflow as tf


class MulticlassDiceLoss(tf.losses.Loss):

    def generalized_dice_coefficient(self, y_true, y_pred):
        smooth = 1.
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
        return score      

    def call(self, y_true, y_pred):
        return 1 - self.generalized_dice_coefficient(y_true, y_pred)
    
class SSIMLoss(tf.losses.Loss):

    def call(self, y_true, y_pred):
        return 1 - tf.image.ssim(y_true, y_pred, max_val=1)
    
class KL_Dice_Loss(tf.losses.Loss):

    def generalized_dice_coefficient(self, y_true, y_pred):
        smooth = 1.
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
        return score
    
    def call(self, y_true, y_pred):
        return tf.keras.losses.kl_divergence(y_true, y_pred) + 1 - self.generalized_dice_coefficient(y_true, y_pred)
    
class KL_Focal_Loss(tf.losses.Loss):

    def call(self, y_true, y_pred):
        return tf.keras.losses.categorical_focal_crossentropy(y_true, y_pred) + tf.keras.losses.kl_divergence(y_true, y_pred)

