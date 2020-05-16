from keras import backend as K
from instancenormalization import InstanceNormalization
from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU
from keras.models import Model
from tensorflow.keras.optimizers import Adam
# from keras.optimizers import Adam
import numpy as np
import time, os
from data_loader import DataLoader
from args_parser import Parser

class Discriminator(object):
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
        self.n_epochs = args.n_epochs_dis
        self.batch_size = args.batch_size_dis

        # data loader
        self.data_loader = DataLoader(args)

        self.discri = self.build_network()
        self.discri.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.discri.summary()

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

    def train(self, save_interval=2):
        start_time = time.time()
        real = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        Y = np.concatenate((real, fake), axis=0)

        for epoch in range(self.n_epochs):
            for i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(self.batch_size)):
                X = np.concatenate((imgs_A, imgs_B), axis=0)
                loss = self.discri.train_on_batch(X, Y)

                elapse = time.time() - start_time
                print("[Epoch %d/%d] [Batch %d/%d] [loss: %05f, acc: %3d%%] time: %05f "
                    % (epoch, self.n_epochs, i, self.data_loader.n_batches, loss[0], 100*loss[1], elapse))
            if epoch % save_interval == 0:
                self.discri.save('single_discri_epoch_%d' %epoch)
   

if __name__ == '__main__':
    args = Parser().parse()
    discriminator = Discriminator(args)
    discriminator.train()