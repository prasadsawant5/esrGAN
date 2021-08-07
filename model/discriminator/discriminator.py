import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dense, Flatten
from tensorflow.keras import Input
from tensorflow.python.keras.engine.training import Model
from model.batch_normalization import BatchNormalization
from utils import _regularizer, _kernel_init


class Discriminator:
    def __init__(self, height: int = 360, width: int = 480, channels: int = 3, num_filters: int = 64, wd: float = 0., name: str = 'discriminator') -> None:
        self.channels = channels
        self.num_filters = num_filters
        self.wd = wd
        self.name = name

        self.height = height
        self.width = width

        self.alpha = 0.2


    def get_conv_2d(self, filters: int, name: str, k_size: int = 3, strides: int = 1, activation: LeakyReLU = None, use_bias: bool = False) -> Conv2D:
        return Conv2D(
            filters=filters, kernel_size=k_size, strides=strides, padding='same', 
            kernel_initializer=_kernel_init(), kernel_regularizer=_regularizer(self.wd),
            activation=activation, name=name, use_bias=use_bias
        )


    def build(self):
        x = inputs = Input((self.height, self.width, self.channels), name='discriminator_input')

        with tf.name_scope(self.name):
            x = self.get_conv_2d(self.num_filters, 'conv0_0', use_bias=True)(x)             # height: 360 width: 480   360x480x64
            x = self.get_conv_2d(self.num_filters, 'conv0_1', k_size=4, strides=2)(x)       # height: 180 width: 240   180x240x64
            x = LeakyReLU(self.alpha)(BatchNormalization(name='bn0_0')(x))

            x = self.get_conv_2d(self.num_filters * 2, 'conv1_0')(x)                        # height: 180 width: 240   180x240x128
            x = LeakyReLU(self.alpha)(BatchNormalization(name='bn1_0')(x))                  
            x = self.get_conv_2d(self.num_filters * 2, 'conv1_1', k_size=4, strides=2)(x)   # height: 90 width: 240    90x120x128
            x = LeakyReLU(self.alpha)(BatchNormalization(name='bn1_1')(x))

            x = self.get_conv_2d(self.num_filters * 4, 'conv2_0')(x)                        # height: 90 width: 120    90x120x256
            x = LeakyReLU(self.alpha)(BatchNormalization(name='bn2_0')(x))
            x = self.get_conv_2d(self.num_filters * 4, 'conv2_1', k_size=4, strides=2)(x)   # height: 45 width: 60     45x60x256
            x = LeakyReLU(self.alpha)(BatchNormalization(name='bn2_1')(x))

            x = self.get_conv_2d(self.num_filters * 8, 'conv3_0')(x)                        # height: 45 width: 60     45x60x256
            x = LeakyReLU(self.alpha)(BatchNormalization(name='bn3_0')(x))
            x = self.get_conv_2d(self.num_filters * 8, 'conv3_1', k_size=4, strides=2)(x)   # height: 22 width: 29     22x29x256
            x = LeakyReLU(self.alpha)(BatchNormalization(name='bn3_1')(x))

            x = self.get_conv_2d(self.num_filters * 8, 'conv4_0')(x)                        # height: 22 width: 29     22x29x256
            x = LeakyReLU(self.alpha)(BatchNormalization(name='bn4_0')(x))
            x = self.get_conv_2d(self.num_filters * 8, 'conv4_1', k_size=4, strides=2)(x)   # height: 11 width: 14     11x14x256
            x = LeakyReLU(self.alpha)(BatchNormalization(name='bn4_1')(x))

            x = Flatten()(x)
            x = Dense(100, kernel_regularizer=_regularizer(self.wd), activation=LeakyReLU(self.alpha), name='linear1')(x)
            out = Dense(units=1, name='output')(x)

            return Model(inputs, out, name=self.name)