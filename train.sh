!/bin/bash

echo "Training rot-1 model"
python3 rotate.py --dir_save_ckpt ./ckpts/noFlip/rot1/cifar10 --dir_save_log ./ckpts/noFlip/rot1/cifar10 --rot 1

echo "Training rot-2 model"
python3 rotate.py --dir_save_ckpt ./ckpts/noFlip/rot2/cifar10 --dir_save_log ./ckpts/noFlip/rot2/cifar10 --rot 2

echo "Training rot-3 model"
python3 rotate.py --dir_save_ckpt ./ckpts/noFlip/rot3/cifar10 --dir_save_log ./ckpts/noFlip/rot3/cifar10 --rot 3

echo "Training rot-4 model"
python3 rotate.py --dir_save_ckpt ./ckpts/noFlip/rot4/cifar10 --dir_save_log ./ckpts/noFlip/rot4/cifar10 --rot 4

echo "Training rot-5 model"
python3 rotate.py --dir_save_ckpt ./ckpts/noFlip/rot5/cifar10 --dir_save_log ./ckpts/noFlip/rot5/cifar10 --rot 5

echo "Training rot-6 model"
python3 rotate.py --dir_save_ckpt ./ckpts/noFlip/rot6/cifar10 --dir_save_log ./ckpts/noFlip/rot6/cifar10 --rot 6