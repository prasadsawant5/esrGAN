import tensorflow as tf
from tensorflow.keras.layers import Layer
from model.generator.layers.residual_dense_block import ResidualDenseBlock

class ResidualInResidualBlock(Layer):
    """
    Residual in Residual Dense Block
    """
    def __init__(self, name, num_filters=64, gc=32, res_beta=0.2, wd=0., **kwargs):
        super(ResidualInResidualBlock, self).__init__(name=name, **kwargs)
        self.res_beta = res_beta

        self.rdb1 = ResidualDenseBlock('rdb_1', num_filters, gc, res_beta, wd)
        self.rdb2 = ResidualDenseBlock('rdb_2', num_filters, gc, res_beta, wd)
        self.rdb3 = ResidualDenseBlock('rdb_3', num_filters, gc, res_beta, wd)


    def call(self, inputs, **kwargs):
        out = self.rdb1(inputs)
        out = self.rdb2(out)
        out = self.rdb3(out)

        return out * self.res_beta + inputs