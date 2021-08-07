import tensorflow as tf

def _kernel_init(scale=1.0, seed=None):
    """
    He normal initializer with scale.
    """
    scale = 2. * scale
    return tf.keras.initializers.VarianceScaling(
        scale=scale, mode='fan_in', distribution='truncated_normal', seed=seed)


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)