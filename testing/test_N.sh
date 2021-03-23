#!/bin/bash
echo "Test for N = 1"
cd ./ckpts/noFlip/rot1/cifar10/
./test_cifar10.sh &
echo "Test for N = 2"
cd ./ckpts/noFlip/rot2/cifar10/
./test_cifar10.sh &
echo "Test for N = 3"
cd ./ckpts/noFlip/rot3/cifar10/
./test_cifar10.sh &
echo "Test for N = 4"
cd ./ckpts/noFlip/rot4/cifar10/
./test_cifar10.sh &
echo "Test for N = 5"
cd ./ckpts/noFlip/rot5/cifar10/
./test_cifar10.sh &
echo "Test for N = 6"
cd ./ckpts/noFlip/rot6/cifar10/
./test_cifar10.sh &
