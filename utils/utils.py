# utils.py
import numpy as np
import cv2
import json


def visualize(loader, path, size=8):
    for ix, sample in enumerate(loader):
        b, c, h, w = sample["image"].size()
        if b < 64:
            raise ValueError("batch size < 64 cannot generate gallery for train data!!!")
        img = sample["image"].numpy()[:64].transpose(0, 2, c, 1).reshape(size, size, h, w, c)
        img = np.swapaxes(img, 1, 2).reshape(size * h, size * w, c)
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(path, img)
        return None


def adjust_lr(opt, it, lr_init, lr_end, total_it, warmup):
    """
    it: iteration
    total_it: total iteration
    warmup = your epoch * len(trainloader)

    """
    if it < warmup:
        lr = lr_init * ((1 / warmup) * it)
    else:
        lr = lr_end + (lr_init - lr_end) * 0.5 * (1 + np.cos((it - warmup) / (total_it - warmup) * np.pi))

    if lr < lr_end:  # boundary
        lr = lr_end
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    return lr


def timeConvert(s):
    """
    given second return hour:minute:second
    """
    s = int(s)
    H = s // 3600
    M = s % 3600 // 60
    S = s % 60
    return H, M, S


def record_saver(record, path):
    with open(path, 'w') as f:
        json.dump(record, f)
