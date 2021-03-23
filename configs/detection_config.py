import argparse


# params for detection
parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=200, help="batch size")
parser.add_argument("--classNum", type=int, default=10, help="class number")
parser.add_argument("--imgSize", type=int, default=32, help="size of input image")
parser.add_argument("--InData", type=str, default="/usr/share/dataset/cifar10", help="root dir of dataset")
parser.add_argument("--ckpt", type=str, default="home/majic/ckpts/cifar10/last.pt")
parser.add_argument("--worker", type=int, default=4, help="number of subprocess for loading data")
parser.add_argument("--store_dir", type=str, default='home/majic/log')
parser.add_argument("--model", type=str, default="w402", help="model used for training")
parser.add_argument("--num_eval", type=int, default=1, help="number of testing")
parser.add_argument("--vis", type=int, default=0, help="visualize loader")
parser.add_argument("--method", type=str, default='MSP', help="method for eval")
parser.add_argument("--rot", type=int, default=4, help="how many different angles for rotations")
parser.add_argument("--pretrain", type=str, default="x", help="path to pretrained model (default = No)")
parser.add_argument("--assume", type=int, default=0)
det_cfg = parser.parse_args()