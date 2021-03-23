import argparse


# configuration for training
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1, help="learning rate for CNN model")
parser.add_argument("--batch", type=int, default=32, help="batch size")
parser.add_argument("--epoch", type=int, default=100, help="epoch")
parser.add_argument("--classNum", type=int, default=10, help="class number")
parser.add_argument("--imgSize", type=int, default=32, help="size of input image")
parser.add_argument("--data_root", type=str, default="/usr/share/dataset/cifar10", help="root dir of dataset")
parser.add_argument("--dir_save_ckpt", type=str, default="ckpts/cifar10", help="dir for saving checkpoints")
parser.add_argument("--dir_save_log", type=str, default="logs/cifar10", help="dir for saving log file")
parser.add_argument("--worker", type=int, default=4, help="number of subprocess for loading data")
parser.add_argument("--decay", type=float, default=5e-4, help="weight decay for optimizer")
parser.add_argument("--momentnum", type=float, default=0.9, help="momentnum for optimizer")
parser.add_argument("--rot", type=int, default=4, help="how many different angles for rotation.")
parser.add_argument("--pretrain", type=str, default="x", help="path to pretrained model (default = No)")
args = parser.parse_args()