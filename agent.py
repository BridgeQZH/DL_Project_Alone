import pickle
from keras import backend as K
from instancenormalization import InstanceNormalization
from keras.layers import Input, Lambda, Add, Conv2D, Conv2DTranspose, Dropout, ReLU, LeakyReLU
from keras.models import Model
# from keras.models import model_from_json
# from keras.models import load_model
# from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
import numpy as np
import time, os
import matplotlib.pyplot as plt
from data_loader import DataLoader # from a python file's name import a class

''' 
Some notations in this program
    mse: mean_squared_error
    mae: Mean Absolute Error
'''
class Agent(object):
    def __init__(self, args):
        '''
        @param args: Arguments for the network and training process
        '''
        super().__init__()
        # use channel last image format
        # K.set_image_data_format('channels_first')

        # image shape
        self.img_rows = args.img_rows
        self.img_cols = args.img_cols
        self.img_channels = args.img_channels
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

        # weights for losses; different from the model's weights in the training process
        self.lambda_cycle = args.lambda_cycle
        self.lambda_identity = args.lambda_identity * self.lambda_cycle

        # build optimizer
        self.learning_rate = args.learning_rate
        self.beta_1 = args.beta_1
        optimizer = Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)

        # training parameters
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size

        # dataset path
        self.dataset_dir = args.dataset_dir
        self.dataset_name = args.dataset_name
        self.save_dir = args.save_dir
        self.modelsave_dir = args.modelsave_dir

        # data loader
        self.data_loader = DataLoader(args)

        # build networks
        # discriminators
        self.d_A = self.build_disciminator()
        self.d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_B = self.build_disciminator()
        self.d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # self.d_A.summary()

        # patchGAN
        out_D = self.d_A.output.shape
        self.patch_D = (self.batch_size, out_D[1], out_D[2], out_D[3])

        # generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # self.g_AB.summary()

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        # for cycle loss
        fake_A = self.g_BA(img_B)
        fake_B = self.g_AB(img_A)
        rec_A = self.g_BA(fake_B)
        rec_B = self.g_AB(fake_A)
        # for identity loss
        idt_A = self.g_BA(img_A)
        idt_B = self.g_AB(img_B)

        self.d_A.trainable = False
        self.d_B.trainable = False
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, rec_A, rec_B, idt_A, idt_B])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'], 
            loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_identity, self.lambda_identity], optimizer=optimizer)

    def build_generator(self, filters=64, n_blocks=9):
        # TODO weights initialization
        def resnet_block(layer_input, filters, kernel_size=3, dropout_rate=0.0):
            # TODO should be reflection padding here
            block_1 = Conv2D(filters, kernel_size=kernel_size, padding='same')(layer_input)
            norm_1 = InstanceNormalization()(block_1)
            out_1 = ReLU()(norm_1)
            if dropout_rate != 0: out_1 = Dropout(dropout_rate)(out_1)

            # TODO should be reflection padding here
            block_2 = Conv2D(filters, kernel_size=kernel_size, padding='same')(layer_input)
            norm_2 = InstanceNormalization()(block_2)

            # skip connection
            out = Add()([layer_input, norm_2])
            return out

        # pre process
        # TODO should be reflection padding here
        img = Input(shape=self.img_shape)
        mid = Conv2D(filters, kernel_size=7, padding='same')(img)
        mid = InstanceNormalization()(mid)
        mid = ReLU()(mid)

        sample = 2
        # downsampling
        for i in range(1, sample+1):
            multi = 2**i
            mid = Conv2D(filters * multi, kernel_size=3, strides=2, padding='same')(mid)
            mid = InstanceNormalization()(mid)
            mid = ReLU()(mid)

        # resnet blocks
        for i in range(n_blocks):
            mid = resnet_block(mid, filters * 2**sample)

        # upsampling
        for i in range(1, sample+1):
            multi = 2**(sample-i)
            mid = Conv2DTranspose(filters * multi, kernel_size=3, strides=2, padding='same')(mid)
            mid = InstanceNormalization()(mid)
            mid = ReLU()(mid)

        # after process
        # TODO should be reflection padding here
        out = Conv2D(self.img_channels, kernel_size=7, padding='same', activation='tanh')(mid)
        return Model(img, out)

    def build_disciminator(self, filters=64, n_layers=3):
        kernel_size = 4
        leaky_alpha = 0.2

        img = Input(shape=self.img_shape)
        mid = Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same')(img) # zero padding
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
    
        out = Conv2D(1, kernel_size=kernel_size, strides=1, padding='same')(mid)
        return Model(img, out)

    def train(self, save_interval=10):
        # save_path = os.path.join(self.modelsave_dir, self.dataset_name)
        # os.makedirs(save_path, exist_ok=True)
        start_time = time.time()
        real = np.ones(self.patch_D)
        fake = np.zeros(self.patch_D)

        for epoch in range(self.n_epochs):
            alldata = []
            for i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(self.batch_size)):
                # train discriminator
                fake_A = self.g_BA.predict(imgs_B)
                fake_B = self.g_AB.predict(imgs_A)
                # loss for d_A
                # note we set metric in the model, thus the output will be loss, accuracy (tuple)
                loss_d_A_real = self.d_A.train_on_batch(imgs_A, real)
                loss_d_A_fake = self.d_A.train_on_batch(fake_A, fake)
                loss_d_A = 0.5 * np.add(loss_d_A_real, loss_d_A_fake)
                # loss for d_B
                # note we set metric in the model, thus the output will be loss, accuracy (tuple)
                loss_d_B_real = self.d_B.train_on_batch(imgs_B, real)
                loss_d_B_fake = self.d_B.train_on_batch(fake_B, fake)
                loss_d_B = 0.5 * np.add(loss_d_B_real, loss_d_B_fake)
                # total loss and accuracy
                loss_d = 0.5 * np.add(loss_d_A, loss_d_B)

                # train generator. self.combined is the model
                loss_g = self.combined.train_on_batch([imgs_A, imgs_B], [real, real, imgs_A, imgs_B, imgs_A, imgs_B])

                elapse = time.time() - start_time
                temp = {}
                temp['epoch'] = epoch  # existing key, so overwrite
                temp['n_epochs'] = self.n_epochs  # new key, so add
                temp['i'] = i
                temp['batch_size'] = self.batch_size
                temp['D_loss']=loss_d[0]
                temp['acc'] = 100*loss_d[1]
                temp['G_loss'] = float("{:.5f}".format(loss_g[0]))
                temp['adv'] = float("{:.5f}".format(np.mean(loss_g[1:3])))
                temp['recon'] = float("{:.5f}".format(np.mean(loss_g[3:5])))
                temp['id'] = float("{:.5f}".format(np.mean(loss_g[5:7])))
                alldata.append(temp)
                # print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %05f "
                #     % (epoch, self.n_epochs, i, self.batch_size, loss_d[0], 100*loss_d[1], 
                #     loss_g[0], np.mean(loss_g[1:3]), np.mean(loss_g[3:5]), np.mean(loss_g[5:7]), elapse)) # two numbers' mean
                print("Epoch %d, Batch %d" %(epoch, i))
                # print(loss_g) # It will print out seven values
                # save generated image samples
                if i % save_interval == 0:
                    self.sample_images(epoch, i) # Where it saves the generated images
                    # self.combined.save('apple2orange_%d' %i) # Delete this for running the whole epoch
                    # self.save_model(i)
            if epoch % save_interval == 0:
                self.save_model(epoch)
            self.save_data(epoch, alldata)
    
    def save_model(self, epoch): 
        self.combined.save('monet2photo_epoch_%d' %epoch)
        print("Saved model to disk")

    def save_data(self, epoch, alldata): 
        pickle.dump( alldata, open( "data%d.p" %epoch, "wb" ) )
        print("Saved data to disk")

    def sample_images(self, epoch, batch):
        save_path = os.path.join(self.save_dir, self.dataset_name)
        save_path = os.path.join(save_path, '256monet')
        os.makedirs(save_path, exist_ok=True)
        r, c = 2, 3 # 2 rows and 3 columns

        imgs_A = self.data_loader.load_random(domain='A', batch_size=1)
        imgs_B = self.data_loader.load_random(domain='B', batch_size=1)
        # imgs_A = self.data_loader.load_index(domain='A', indices=[1])
        # imgs_B = self.data_loader.load_index(domain='B', indices=[1])
        fake_A = self.g_BA.predict(imgs_B)
        fake_B = self.g_AB.predict(imgs_A)
        rec_A = self.g_BA.predict(fake_B)
        rec_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, rec_A, imgs_B, fake_A, rec_B])
        gen_imgs = 0.5 * gen_imgs + 0.5     # scale to 0 - 1
        # gen_imgs = (gen_imgs + 1) * 127.5

        titles = ['Original', 'Fake', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(save_path+"/%d_%d.png" % (epoch, batch))
        plt.close()
