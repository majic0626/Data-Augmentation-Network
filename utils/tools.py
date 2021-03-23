# utils.py
import torch
import json
import cv2


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


def record_saver(args, record):
    with open(args.dir_save_log + '/' + 'log.txt', 'w') as f:
        json.dump(record, f)


