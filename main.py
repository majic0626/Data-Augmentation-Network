# rotation.py
"""
given a set of augmentation for an image...then training a model.

"""
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import json
import time
from configs.train_config import train_cfg
from data.build import buildLoader
from modeling.build import buildModel
from solver.build import getOptim
from solver.lr_schduler import adjust_lr
from utils.tools import timeConvert, getDevice


def Train(ep):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for ix, sample in enumerate(trainloader):
        opt.zero_grad()
        # get data and label from loader
        data = []
        label = []
        for t in sample:  # ensure order
            data.append(sample[t]["image"])
            label.append(sample[t]["label"])
        # concat data and label
        data = torch.cat(data, 0)
        label = torch.cat(label, 0)
        data, label = data.cuda(), label.cuda()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        opt.step()
        lr = adjust_lr(
            opt=opt,
            it=(ep - 1) * len(trainloader) + ix,
            lr_init=args.lr,
            lr_end=1e-6,
            total_it=args.epoch * len(trainloader),
            warmup=0
        )
        train_loss += loss.item()
        _, predict = output.max(1)
        total += label.size(0)
        correct += predict.eq(label).sum().item()
        if (ix + 1) % 100 == 0:
            print("Learning Rate: {}".format(lr))
            print("L-train loss:{} / L-acc:{}".format(
                train_loss / (ix + 1),
                100 * correct / total))
    record["train_loss"].append(train_loss / (ix + 1))
    record["train_acc"].append(100 * correct / total)


def Test():
    model.eval()
    total = 0
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for ix, sample in enumerate(testloader):
            data = []
            label = []
            for t in sample:  # ensure the order
                data.append(sample[t]["image"])
                label.append(sample[t]["label"])
            data = torch.cat(data, 0)
            label = torch.cat(label, 0)
            data = data.cuda()
            label = label.cuda()
            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item()
            _, predict = output.max(1)
            total += label.size(0)
            correct += predict.eq(label).sum().item()
        print("L-test loss:{} / L-acc:{}".format(
            test_loss / (ix + 1),
            100 * correct / total))
        record["test_loss"].append(test_loss / (ix + 1))
        record["test_acc"].append(100 * correct / total)
    return 100 * correct / total


def record_saver(record, path):
    with open(path, 'w') as f:
        json.dump(record, f)


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
        Train(ep=i)
        print("========== [Testing] ==========")
        test_acc = Test()
        if test_acc > best_acc:
            print("save the new best model: {} || old best model: {}".format(test_acc, best_acc))
            best_acc = test_acc
            torch.save({"cnn": model.state_dict(), "epoch": i}, args.dir_save_ckpt + '/' + "best.pt")
        print("save the last model: {} || best model: {}".format(test_acc, best_acc))
        torch.save({"cnn": model.state_dict(), "epoch": i}, args.dir_save_ckpt + '/' + "last.pt")
        record_saver(record, args.dir_save_log + '/' + "clf.txt")
        e = time.time()
        t_cost = e - s
        hc, mc, sc = timeConvert(t_cost)  # h:m:s cost
        ht, mt, st = timeConvert(t_cost * (args.epoch - i))  # h:m:s total
        print("Time: {}h:{}m:{}s / Total time: {}h:{}m:{}s".format(hc, mc, sc, ht, mt, st))
