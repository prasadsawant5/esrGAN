from os import name
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.ops.gen_array_ops import size
from utils import _kernel_init, _regularizer
from model.generator.layers.residual_in_residual_dense_block import ResidualInResidualBlock

class Generator:
    def __init__(self, num_filters: int, height: int = 90, width: int = 120, channels: int = 3, nb: int = 8, gc: int = 32, wd: float = 0., name: str = 'generator') -> None:
        self.gc = gc
        self.wd = wd
        self.height = height
        self.width = width
        self.channels = channels
        self.num_filters = num_filters
        self.name = name
        self.nb = nb

        self.alpha = 0.2

    
    def get_conv_2d(self, filters: int, name: str, use_bias: bool = True, activation: tf.keras.activations = None) -> Conv2D:
        return Conv2D(
            filters=filters, kernel_size=3, padding='same', use_bias=use_bias, 
            bias_initializer='zeros', kernel_initializer=_kernel_init(), 
            kernel_regularizer=_regularizer(self.wd), name=name, activation=activation)

    
    def build(self) -> tf.keras.Model:
        x = inputs = tf.keras.Input(shape=(self.height, self.width, self.channels), name='input_image')

        with tf.name_scope(self.name):
            rrdb_trunk_f = tf.keras.Sequential(
                [ResidualInResidualBlock('RRDB_{}'.format(i)) for i in range(self.nb)],
                name='RRDB_trunk'
            )

            fea = self.get_conv_2d(self.num_filters, 'conv_base')(x)
            rrdb = rrdb_trunk_f(fea)
            conv_trunk = self.get_conv_2d(self.num_filters, 'conv_trunk')(rrdb)
            fea = fea + conv_trunk

            # upsampling
            size_fea_h = tf.shape(fea)[1] if self.height is None else self.height
            size_fea_w = tf.shape(fea)[2] if self.width is None else self.width
            fea_resize = tf.image.resize(fea, [size_fea_h * 2, size_fea_w * 2], method='nearest', name='upsample_0')

            fea = self.get_conv_2d(self.num_filters, 'upsample_conv_0', activation=LeakyReLU(self.alpha))(fea_resize)

            fea_resize = tf.image.resize(fea, [size_fea_h * 4, size_fea_w * 4], method='nearest', name='upsample_1')

            fea = self.get_conv_2d(self.num_filters, 'upsample_conv_1', activation=LeakyReLU(self.alpha))(fea_resize)

            fea = self.get_conv_2d(self.num_filters, 'upsample_conv_2', activation=LeakyReLU(self.alpha))(fea)

            out = self.get_conv_2d(self.channels, 'conv_last')(fea)

            return Model(inputs, out, name='generator')





        
