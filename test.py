from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Layer, Add, Flatten
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

class resnet_block(Layer):
    def __init__(self, **kwargs):
        super(resnet_block, self).__init__(**kwargs)

    def build(self, input_shape):
        super(resnet_block, self).build(input_shape)
        
    def call(self, x):
        conv = Conv2D(3, 3, padding='same')(x)
        out = Add()([x, conv])
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()

def resnet_func(x):
    conv = Conv2D(64, 3, padding='same')(x)
    out = Add()([x, conv])
    return out

if __name__ == '__main__':
    K.set_image_data_format('channels_first')
    # shape = (128, 128, 3)
    shape = (3, 128, 128)
    input = Input(shape=shape)
    mid = Conv2D(64, kernel_size=4, strides=2, padding='same')(input)
    # mid = resnet_block()(input)
    mid = resnet_func(mid)
    for i in range(2):
        mid = Conv2D(10, 3)(mid)
    mid = Flatten()(mid)
    out = Dense(1)(mid)
    model = Model(inputs=input, outputs=out)
    model.summary()