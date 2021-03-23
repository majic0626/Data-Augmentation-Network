# rotation.py
"""
given a set of augmentation for an image...then training a model.

"""
import torch
import torch.nn as nn
import numpy as np
import time
from configs.train_config import train_cfg
from data.build import buildLoader
from modeling.build import buildModel
from solver.build import getOptim
from solver.lr_schduler import adjust_lr
from engine.trainer import Train
from engine.tester import Test
from utils.tools import timeConvert, getDevice, record_saver


if __name__ == "__main__":

    # for reproductivity
    torch.manual_seed(1)
    np.random.seed(1)
    args = train_cfg
    record = {"train_loss": [],
              "train_acc": [],
              "test_loss": [],
              "test_acc": []}

    # ========== [data] ==========
    trainloader = buildLoader(args, mode='train')
    testloader = buildLoader(args, mode='test')

    # ========== [device (cpu / gpu)] =============
    device = getDevice()

    # ========== [model] ==========
    model = buildModel(args)
    model.to(device)

    # ========== [optim (SGD)] ==========
    opt = getOptim(args, model)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    epoch = 1
    for i in range(1, args.epoch + 1):
        s = time.time()
        print("========== [Training] ==========")
        print("[epoch {}/{}]".format(i, args.epoch))
        Train(args, i, trainloader, model, opt, adjust_lr, criterion, record)
        print("========== [Testing] ==========")
        Test(args, i, testloader, model, criterion, record, best_acc)
        e = time.time()
        t_cost = e - s
        hc, mc, sc = timeConvert(t_cost)  # h:m:s cost
        ht, mt, st = timeConvert(t_cost * (args.epoch - i))  # h:m:s total
        print("Time: {}h:{}m:{}s / Total time: {}h:{}m:{}s".format(hc, mc, sc, ht, mt, st))
        record_saver(record, args)
