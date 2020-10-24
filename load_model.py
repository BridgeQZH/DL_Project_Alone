from keras import backend as K
from keras.models import load_model
from instancenormalization import InstanceNormalization
from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU
from keras.models import Model
# from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
import numpy as np
import time, os
from data_loader import DataLoader
from args_parser import Parser

class LoadModel(object):
    def __init__(self, args):
        super().__init__()

        # image shape
        self.img_rows = args.img_rows
        self.img_cols = args.img_cols
        self.img_channels = args.img_channels
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

        # build network
        self.learning_rate = args.learning_rate
        self.beta_1 = args.beta_1
        optimizer = Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)

        # training parameters
        self.n_epochs = args.n_epochs_load
        self.batch_size = args.batch_size_load

        # data loader
        self.data_loader = DataLoader(args)

        # model loading
        # self.single_dis_model = load_model('/home/zihan/Downloads/model/single_discri_epoch_18', custom_objects={'InstanceNormalization': InstanceNormalization})
        # self.combined_model = load_model('/home/zihan/Downloads/model/monet2photo_data_aug_model/nnnew_monet2photo_fliped_epoch_60', custom_objects={'InstanceNormalization': InstanceNormalization})
        self.single_dis_model = load_model('/home/zihan/Downloads/model/single_discri_epoch_18', custom_objects={'InstanceNormalization': InstanceNormalization})
        self.combined_model = load_model('/home/zihan/Downloads/model/monet2photo_data_aug_model/nnnew_monet2photo_fliped_epoch_60', custom_objects={'InstanceNormalization': InstanceNormalization})
        layer_name = 'model_3'
        self.intermediate_layer_model = Model(inputs=self.combined_model.input, outputs=self.combined_model.get_layer(layer_name).get_output_at(-3))
        self.intermediate_layer_model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.intermediate_layer_model.summary()
        print("successfully loading the model")
        # self.discri = self.build_network()
        # self.discri.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # self.discri.summary()

    def build_network(self, filters=64, n_layers=3):
        kernel_size = 4
        leaky_alpha = 0.2

        img = Input(shape=self.img_shape)
        mid = Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same')(img)
        mid = LeakyReLU(leaky_alpha)(mid)

        for i in range(1, n_layers):
            multi = min(2**i, 8)
            mid = Conv2D(filters * multi, kernel_size=kernel_size, strides=2, padding='same')(mid)
            mid = InstanceNormalization()(mid)
            mid = LeakyReLU(leaky_alpha)(mid)

        multi = min(2**n_layers, 8)
        mid = Conv2D(filters * multi, kernel_size=kernel_size, strides=1, padding='same')(mid)
        mid = InstanceNormalization()(mid)
        mid = LeakyReLU(leaky_alpha)(mid)
    
        mid = Conv2D(1, kernel_size=kernel_size, strides=1, padding='same')(mid)
        mid = Flatten()(mid)
        out = Dense(1, activation='sigmoid')(mid)
        return Model(img, out)

    def test(self, save_interval=1):
        start_time = time.time()
        real = np.ones((self.batch_size, 1))
        # fake = np.zeros((self.batch_size, 1))
        # Y = np.concatenate((real, fake), axis=0)
        fake_list=[]
        for i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(self.batch_size)):
            intermediate_output = self.intermediate_layer_model.predict([imgs_A, imgs_B]) 
            intermediate_output = intermediate_output[0]    
            fake_list.append(intermediate_output)
            print("%d" %(i))     
        dis_result = self.single_dis_model.evaluate(np.array(fake_list), np.ones((len(fake_list), 1)), batch_size=1)
        print('test loss, test acc:', dis_result)
        
if __name__ == '__main__':
    args = Parser().parse() 
    lm = LoadModel(args)
    lm.test()