import argparse

class Parser(object):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(description='CycleGAN')

    def parse(self):
        self.parser.add_argument('--img_rows', type=int, default=128, help='# of rows in input images')
        self.parser.add_argument('--img_cols', type=int, default=128, help='# of columns in input images')
        self.parser.add_argument('--img_channels', type=int, default=3, help='# of channels in input images, 3 for RGB and 1 for grayscale')

        self.parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for forward and backward cycle loss')
        self.parser.add_argument('--lambda_identity', type=float, default=0.5, help='weight for identity loss')

        self.parser.add_argument('--learning_rate', type=float, default=0.0002, help='initial learning rate for Adam')
        self.parser.add_argument('--beta_1', type=float, default=0.5, help='momentum term for Adam') # usually this value shoule be close to 1

        self.parser.add_argument('--dataset_dir', type=str, default='./datasets', help='directory of all datasets')
        self.parser.add_argument('--dataset_name', type=str, default='apple2orange', help='dataset that network trains on')
                # If we want to change the dataset we want to train on. We need to modify this above
 
        self.parser.add_argument('--save_dir', type=str, default='./images', help='directory to save generated images')
        self.parser.add_argument('--modelsave_dir', type=str, default='./model', help='directory to save the model')

        self.parser.add_argument('--n_epochs', type=int, default=2, help='# of training epochs')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')

        return self.parser.parse_args()