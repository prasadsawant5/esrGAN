import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

class BatchNormalization(BatchNormalization):
    """
    Make trainable=False freeze BN for real (the og version is sad).
    ref: https://github.com/zzh8829/yolov3-tf2
    """
    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-5, center=True, scale=True, name=None, **kwargs):
        super(BatchNormalization, self).__init__(axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale, name=name, **kwargs)

    
    def call(self, inputs, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(inputs, training=training)