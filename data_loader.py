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
        self.img_size = (self.img_rows, self.img_cols)
        self.img_channels = args.img_channels
        self.img_shape = (self.img_channels, self.img_rows, self.img_cols)

        # image augmentation
        self.resize = args.resize
        self.resize_size = args.resize_size
        self.crop = args.crop
        self.crop_size = args.crop_size
        self.flip = args.flip
        self.flip_prob = args.flip_prob

        # dataset path
        self.dataset_dir = args.dataset_dir
        self.dataset_name = args.dataset_name

    def load_index(self, domain, indices):
        path = os.path.join(self.dataset_dir, self.dataset_name, 'train'+domain+'/*')
        train = glob(path)
        selected = [train[i] for i in indices]
        imgs = self.read(selected)
        return imgs

    def load_random(self, domain, batch_size=1):
        path = os.path.join(self.dataset_dir, self.dataset_name, 'train'+domain+'/*')
        train = glob(path)
        selected = np.random.choice(train, size=batch_size, replace=False)
        imgs = self.read(selected)
        return imgs

    def load_batch(self, batch_size=1):
        path_A = os.path.join(self.dataset_dir, self.dataset_name, 'trainA/*')
        train_A = glob(path_A)
        path_B = os.path.join(self.dataset_dir, self.dataset_name, 'trainB/*')
        train_B = glob(path_B)

        # number of times we return batches
        self.n_batches = min(len(train_A), len(train_B)) // batch_size
        # permute samples by random choice
        total_samples = self.n_batches * batch_size
        selected_A = np.random.choice(train_A, size=total_samples, replace=False)
        selected_B = np.random.choice(train_B, size=total_samples, replace=False)

        for i in range(self.n_batches - 1):
            batch_A = selected_A[i*batch_size:(i+1)*batch_size]
            batch_B = selected_B[i*batch_size:(i+1)*batch_size]
            imgs_A = self.read(batch_A)
            imgs_B = self.read(batch_B)
            yield imgs_A, imgs_B

    # @param l: list of path of images to read
    def read(self, l):
        imgs = []
        for p in l:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            # convert channels
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float)
            # resize
            if self.resize:
                img = cv2.resize(img, (self.resize_size, self.resize_size))
            if self.crop:
                img = self.random_crop(img)
            if self.flip:
                if np.random.random() < self.flip_prob:
                    img = cv2.flip(img, 0)
            img = cv2.resize(img, self.img_size)
            # TODO whether to flip the image or not
            imgs.append(img)
        # scale to [-1, 1]
        imgs = np.array(imgs) / 127.5 - 1
        return imgs

    def random_crop(self, img):
        X, Y, channels = img.shape
        x = np.random.choice(X - self.crop_size)
        y = np.random.choice(Y - self.crop_size)
        return img[x:x+self.crop_size, y:y+self.crop_size]

if __name__ == '__main__':
    args = Parser().parse()
    data_loader = DataLoader(args)
    # for i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(args.batch_size)):
    #     print(i, imgs_A.shape, imgs_B.shape)
    data_loader.load_index('A', [1])