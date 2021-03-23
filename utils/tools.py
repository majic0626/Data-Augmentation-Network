# utils.py
import torch
import numpy as np
import cv2


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





def timeConvert(s):
    """
    return hour:minute:second by given second

    """
    s = int(s)
    H = s // 3600
    M = s % 3600 // 60
    S = s % 60
    return H, M, S


def getDevice():
    # return device the computer will use
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def record_saver(record, args):
    with open(args.dir_save_log, 'w') as f:
        json.dump(record, f)


