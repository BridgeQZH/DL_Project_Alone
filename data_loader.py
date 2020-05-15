from args_parser import Parser
import os
from glob import glob
import cv2
import numpy as np

class DataLoader(object):
    def __init__(self, args):
        super().__init__()

        # image shape
        self.img_rows = args.img_rows
        self.img_cols = args.img_cols
        self.img_size = (self.img_rows, self.img_cols) # which is a tuple
        self.img_channels = args.img_channels
        # self.img_shape = (self.img_channels, self.img_rows, self.img_cols) # channel first?

        # dataset path
        self.dataset_dir = args.dataset_dir
        self.dataset_name = args.dataset_name

    def load_index(self, domain, indices): # domain 
        path = os.path.join(self.dataset_dir, self.dataset_name, 'train'+domain+'/*')
        train = glob(path) # all files in a list
        selected = [train[i] for i in indices]
        # if i = 2
        # print(type(selected)) # <class 'list'>
        # print(selected) # ['./datasets/apple2orange/trainA/n07740461_9687.jpg', './datasets/apple2orange/trainA/n07740461_2798.jpg']
        imgs = self.read(selected)
        # print(type(imgs)) # <class 'numpy.ndarray'>
        return imgs

    def load_random(self, domain, batch_size=1):
        path = os.path.join(self.dataset_dir, self.dataset_name, 'train'+domain+'/*')
        train = glob(path)
        selected = np.random.choice(train, size=batch_size, replace=False) # Generates number of batch_size random samples from a given 1-D array
        imgs = self.read(selected)
        return imgs

    def load_batch(self, batch_size=1):
        path_A = os.path.join(self.dataset_dir, self.dataset_name, 'trainA/*')
        train_A = glob(path_A)
        path_B = os.path.join(self.dataset_dir, self.dataset_name, 'trainB/*')
        train_B = glob(path_B)

        # number of times we return batches
        n_batches = min(len(train_A), len(train_B)) // batch_size
        # permute samples by random choice
        total_samples = n_batches * batch_size
        selected_A = np.random.choice(train_A, size=total_samples, replace=False)
        selected_B = np.random.choice(train_B, size=total_samples, replace=False)

        for i in range(n_batches - 1):
            batch_A = selected_A[i*batch_size:(i+1)*batch_size]
            batch_B = selected_B[i*batch_size:(i+1)*batch_size]
            imgs_A = self.read(batch_A)
            imgs_B = self.read(batch_B)
            yield imgs_A, imgs_B

    # @param l: list of path of images to read
    def read(self, l, resize=True, flip = False):
        imgs = []
        for p in l:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            # convert channels
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Because the default sequence of opencv reading an image is BGR
            # print(type(img)) # <class 'numpy.ndarray'>
            # print(img.shape) # (256,256,3)
            img = img.astype(np.float)
            # flip
            if flip:
                imgflipVertical = cv2.flip(img, 0)
                imgflipHorizontal = cv2.flip(img, 1)
                imgflipBoth = cv2.flip(img, -1)
            # resize
            if resize:
                img = cv2.resize(img, self.img_size)
            
            # TODO whether to flip the image or not
            imgs.append(img)
        # scale to [-1, 1]
        imgs = np.array(imgs) / 127.5 - 1
        return imgs

if __name__ == '__main__':
    args = Parser().parse()
    data_loader = DataLoader(args)
    # for i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(args.batch_size)):
    #     print(i, imgs_A.shape, imgs_B.shape)
    imgs_loaded = data_loader.load_index('A', [1,2])
    # cv2.imshow('image',imgs_loaded[1])
    # k = cv2.waitKey(0)

    # if k == 27:
    #     cv2.destroyAllWindows()
    # elif k == ord('s'):
    #     cv2.imwrite('/home/zihan/Pictures/blue_copy.png',img)
    #     cv2.destroyAllWindows()