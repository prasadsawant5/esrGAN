import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, LeakyReLU
from utils import _kernel_init, _regularizer

class ResidualDenseBlock(Layer):
    """
    Residual dense block
    """
    def __init__(self, name, nf=64, gc=32, res_beta=0.2, wd=0., **kwargs):
        super(ResidualDenseBlock, self).__init__(name=name, **kwargs)
        # gc: growth channel, intermediate channels
        self.res_beta = res_beta

        self.conv1 = Conv2D(
            filters=gc, 
            kernel_size=3, 
            padding='same', 
            kernel_initializer=_kernel_init(0.1), 
            bias_initializer='zeros', 
            kernel_regularizer=_regularizer(wd),
            activation=LeakyReLU(alpha=0.2)
            )

        self.conv2 = Conv2D(
            filters=gc, 
            kernel_size=3, 
            padding='same', 
            kernel_initializer=_kernel_init(0.1), 
            bias_initializer='zeros', 
            kernel_regularizer=_regularizer(wd),
            activation=LeakyReLU(alpha=0.2)
            )

        self.conv3 = Conv2D(
            filters=gc, 
            kernel_size=3, 
            padding='same', 
            kernel_initializer=_kernel_init(0.1), 
            bias_initializer='zeros', 
            kernel_regularizer=_regularizer(wd),
            activation=LeakyReLU(alpha=0.2)
            )

        self.conv4 = Conv2D(
            filters=gc, 
            kernel_size=3, 
            padding='same', 
            kernel_initializer=_kernel_init(0.1), 
            bias_initializer='zeros', 
            kernel_regularizer=_regularizer(wd),
            activation=LeakyReLU(alpha=0.2)
            )

        self.conv5 = Conv2D(
            filters=nf, 
            kernel_size=3, 
            padding='same', 
            kernel_initializer=_kernel_init(0.1), 
            bias_initializer='zeros', 
            kernel_regularizer=_regularizer(wd),
            activation=LeakyReLU(alpha=0.2)
            )


    def call(self, inputs, **kwargs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(tf.concat([inputs, x1], 3))
        x3 = self.conv3(tf.concat([inputs, x1, x2], 3))
        x4 = self.conv4(tf.concat([inputs, x1, x2, x3], 3))
        x5 = self.conv5(tf.concat([inputs, x1, x2, x3, x4], 3))

        return x5 * self.res_beta + inputs