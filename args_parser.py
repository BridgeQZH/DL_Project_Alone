import argparse

class Parser(object):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(description='CycleGAN')

    def parse(self):
        self.parser.add_argument('--img_rows', type=int, default=256, help='# of rows in input images')
        self.parser.add_argument('--img_cols', type=int, default=256, help='# of columns in input images')
        self.parser.add_argument('--img_channels', type=int, default=3, help='# of channels in input images, 3 for RGB and 1 for grayscale')

        self.parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for forward and backward cycle loss')
        self.parser.add_argument('--lambda_identity', type=float, default=0.5, help='weight for identity loss')

        self.parser.add_argument('--learning_rate', type=float, default=0.0002, help='initial learning rate for Adam')
        self.parser.add_argument('--beta_1', type=float, default=0.5, help='momentum term for Adam')

        self.parser.add_argument('--dataset_dir', type=str, default='./datasets', help='directory of all datasets')
        self.parser.add_argument('--dataset_name', type=str, default='monet2photo', help='dataset that network trains on')
        self.parser.add_argument('--save_dir', type=str, default='./images', help='directory to save generated images')

        self.parser.add_argument('--n_epochs', type=int, default=70, help='# of training epochs')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')

        self.parser.add_argument('--n_epochs_dis', type=int, default=20, help='# of training epochs for separate discriminator')
        self.parser.add_argument('--batch_size_dis', type=int, default=20, help='batch size for separate discriminator')

        self.parser.add_argument('--resize', type=bool, default=True, help='whether to resize the image')
        self.parser.add_argument('--resize_size', type=int, default=286, help='resize image to this size')
        self.parser.add_argument('--crop', type=bool, default=True, help='whether to crop the image')
        self.parser.add_argument('--crop_size', type=int, default=256, help='crop image to this size')
        self.parser.add_argument('--flip', type=bool, default=True, help='whether to flip the image')
        self.parser.add_argument('--flip_prob', type=float, default=0.5, help='the probability to flip the image')

        return self.parser.parse_args()