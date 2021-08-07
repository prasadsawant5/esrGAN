import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, Loss

class DiscriminatorLoss(Loss):

    def __init__(self, gan_type='ragan', name='discriminator_loss'):
        super().__init__(name=name)
        self.gan_type = gan_type


    def call(self, y_true, y_pred):
        if self.gan_type == 'ragan':
            return 0.5 * (
                BinaryCrossentropy(from_logits=False)(tf.ones_like(y_true), tf.sigmoid(y_true - tf.reduce_mean(y_pred))) + 
                BinaryCrossentropy(from_logits=False)(tf.zeros_like(y_pred), tf.sigmoid(y_pred - tf.reduce_mean(y_true)))
            )
        else:
            return (
                BinaryCrossentropy(from_logits=False)(tf.ones_like(y_true), tf.sigmoid(y_true)) + 
                BinaryCrossentropy(from_logits=False)(tf.zeros_like(y_pred), tf.sigmoid(y_pred))
            )