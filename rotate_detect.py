"""
1. Load in-dist data e.g. CIFAR10, and out-dist data e.g. Places365
2. Feed them into model and get score
3. Evaluate the score with out-of-distribution detector.

"""
import torch
import torch.nn as nn
import numpy as np
from configs.detection_config import det_cfg
from utils.tools import getDevice
from data.build import buildLoader
from modeling.build import buildModel
import matplotlib.pyplot as plt
from collections import OrderedDict
from data.augment import Rot
from torch.utils.data import DataLoader
from modeling.build import buildModel
from utils.metric import FPR, AUPR_Out, AUROC, DetectErr


def validate_assumption(score, name):
    """
    This function test our assumption
    Input:
        score (dict) --> {"in": score_in, "out": score_out}
        name (str) --> name for figure ("MSP" or "MeanMax")
    Return:
        None
    """
    n1, _, _ = plt.hist(
        score["in"],
        bins=np.arange(0, 1.05, 0.05),
        density=False,
        facecolor='g',
        alpha=0.3,
        label='in-distribution')
    n2, _, _ = plt.hist(
        score["out"],
        bins=np.arange(0, 1.05, 0.05),
        density=False,
        facecolor='r',
        alpha=0.3,
        label='out-of-distribution')
    plt.yticks(np.arange(0, 10000, step=1000))
    plt.ylabel('count')
    plt.xlabel('confidence score')
    plt.legend(loc='upper left')
    plt.savefig(name)
    plt.close()


def histogram(score):
    """
    in_score: scores for in-distribution (np.array)
    return:
        n1
    """
    # the histogram of the data
    n1, _, _ = plt.hist(score, bins=np.arange(0, 1.2, 0.2), density=False, facecolor='g', alpha=0.5)
    plt.close()
    return n1


def CreateDataLoader(dataName):
    '''
    create a in-distribution or out-of-distribution dataloader e.g. cifar10 / svhn...

    '''
    root = '/usr/share/dataset'
    args.data_root = root + '/' + dataName
    return buildLoader(args, mode='test')


def CreateSyntheticOutDistLoader(dataName):
    '''
    create out-dist (synthetic) dataloader e.g. gaussian

    '''
    root = '/usr/share/dataset'
    args.data_root = root + '/' + dataName
    return buildLoader(args, mode='syn')


def Test(loader, in_dist=False):
    model.eval()
    total = 0
    correct = 0
    stats = {}  # stats: {"rot_0": {"prob": np.array(10000, 10)}, "rot_90": {"prob": np.array(10000, 10)}, ...}

    # states for out-dist
    if not in_dist:
        for t in AugName:
            stats[t] = {"prob": np.zeros((num_OutData, args.classNum), dtype=np.float32)}
    # states for in-dist
    else:
        for t in AugName:
            stats[t] = {"prob": np.zeros((num_InData, args.classNum), dtype=np.float32)}
    with torch.no_grad():
        for ix, sample in enumerate(loader):
            if (ix * args.batch >= num_OutData) and (not in_dist):  # get only 2000 samples from out-dist
                break
            data = []
            label = []
            for t in AugName:  # ensure order
                data.append(sample[t]["image"])
                label.append(sample[t]["label"])
            data = torch.cat(data, 0)
            label = torch.cat(label, 0)
            data, label = data.cuda(), label.cuda()
            output = model(data)
            prob = nn.functional.softmax(output, dim=1)
            _, predict = torch.max(prob, dim=1)
            total += label.size(0)
            correct += predict.eq(label).sum().item()
            for i, t in enumerate(AugName):  # ensure order
                BEGIN = ix * args.batch
                END = BEGIN + (data.size(0) // len(AugName))
                if END > stats[t]["prob"].shape[0]:
                    END = stats[t]["prob"].shape[0]
                begin = i * args.batch
                end = begin + (END - BEGIN)
                stats[t]["prob"][BEGIN:END] = prob.cpu().numpy()[begin:end]

    print("Test Acc: {}%".format(100 * correct / total))
    return stats


def GetScore(stats, method='MSP'):
    """
    stats: {"rot_0": {"prob": np.array(10000, 10)}, "rot_90": {"prob": np.array(10000, 10)}, ...}
    given a set of prob score and return the confidence score.

    """
    dataNum = stats[AugName[0]]["prob"].shape[0]  # number of data
    if method == 'MSP':
        """
        basline method by D.Hendrycks
        score := max(p0, axis=1), p0 means original image

        """
        score = np.max(stats[AugName[0]]["prob"], axis=1)

    elif method == 'JSD':
        H = np.zeros(args.classNum, dtype=np.float32)
        H[0] = 1  # one-hot probability
        U = np.ones(args.classNum, dtype=np.float32) / args.classNum  # uniform
        M = (H + U) / 2  # middle probability
        k1 = np.sum((H + 1e-15) * np.log2((H + 1e-15) / M))
        k2 = np.sum(U * np.log2(U / M))
        UB = (k1 + k2) / 2  # upper bound
        print("upper bound: ", UB)
        score_set = np.zeros((dataNum, len(stats)), dtype=np.float32)
        for ix, t in enumerate(AugName):
            m = (stats[t]["prob"] + U) / 2
            stats[t]["prob"] += 1e-15
            k1 = np.sum(stats[t]["prob"] * np.log2(stats[t]["prob"] / m), axis=1)  # KL(p, m)
            k2 = np.sum(U * np.log2(U / m), axis=1)
            score_set[:, ix] = (k1 + k2) / 2

        score_set /= UB  # normalize confidence score (0 ~ 1)
        score_set[score_set < 0] = 0
        score_set[score_set > 1] = 1
        score = np.mean(score_set, axis=1)
        return score

    elif method == 'MaxMax':
        """
        score := max(concat(max(pi, axis=1)), axis=1)
        """
        score_set = np.zeros((dataNum, len(stats)), dtype=np.float32)
        for ix, t in enumerate(AugName):
            score_set[:, ix] = np.max(stats[t]["prob"], axis=1)

        score = np.max(score_set, axis=1)

    elif method == 'MeanMax':
        """
        score := mean(concat(max(pi, axis=1)), axis=1)
        """
        score_set = np.zeros((dataNum, len(stats)), dtype=np.float32)
        for ix, t in enumerate(AugName):
            score_set[:, ix] = np.max(stats[t]["prob"], axis=1)

        score = np.mean(score_set, axis=1)

    elif method == "MeanPosMax":
        """
        i = argmax(p0), position where contains maximum value in p0
        score := mean(concat(pji))
        """
        index = np.argmax(stats[AugName[0]]["prob"], axis=1)
        score_set = np.zeros((dataNum, len(stats)), dtype=np.float32)
        for ix, t in enumerate(AugName):
            score_set[:, ix] = stats[t]["prob"][np.arange(dataNum), index]
        score = np.mean(score_set, axis=1)

    return score


if __name__ == "__main__":
    args = det_cfg
    torch.manual_seed(1)  # for reprodictivity
    np.random.seed(1)
    mean = [0.5] * 3  # data normal mean
    std = [0.5] * 3  # data normal std
    num_InData = 10000  # number of in-dist data
    num_OutData = num_InData // 5  # number of out-dist data to be evaluated
    DetResult = {}  # out-of-distribution detection result

    # prepare model
    device = getDevice()
    model = buildModel(args)

    # uncomment the following for load weights
    # ckpt = torch.load(args.ckpt)  # checkpoint
    # model.load_state_dict(ckpt["cnn"])  # load weights

    model.to(device)  # model to gpu
    criterion = nn.CrossEntropyLoss()  # loss funcution
    AugName = []
    AugDict = OrderedDict()
    for i in range(args.rot):
        AugName.append("rot_{}".format(int(i * 360 / args.rot)))
        AugDict["rot_{}".format(int(i * 360 / args.rot))] = Rot(int(i * 360 / args.rot))

    print(AugName)
    print(AugDict)

    # prepare in-dist data ans its score
    IndataName = args.InData.split("/")[-1]  # cifar10
    In_loader = CreateDataLoader(IndataName)
    all_score = OrderedDict()  # contain score for each dataset
    stats_in = Test(loader=In_loader, in_dist=True)  # predict prob for in-dist data
    score_in = GetScore(stats=stats_in, method=args.method)  # score for in-dist data
    scores_assumption = {"in": list(score_in), "out": []}
    all_score[IndataName] = histogram(score_in)

    if IndataName == 'cifar10':
        OutdataNameList = ['texture', 'svhn', 'places365', 'lsun', 'cifar100']
    elif IndataName == 'cifar100':
        OutdataNameList = ['texture', 'svhn', 'places365', 'lsun', 'cifar10']
    elif IndataName == 'tinyimagenet200':
        OutdataNameList = ['texture', 'svhn', 'places365', 'lsun']

    # test out-of-distribution detection performance for natural dataset
    for d in OutdataNameList:
        print("processing {} dataset...".format(d))
        DetResult[d] = {"auroc": np.zeros(args.num_eval, dtype=np.float32),
                        "aupr": np.zeros(args.num_eval, dtype=np.float32),
                        "deterr": np.zeros(args.num_eval, dtype=np.float32),
                        "fpr": np.zeros(args.num_eval, dtype=np.float32)}
        for i in range(args.num_eval):
            Out_loder = CreateDataLoader(d)
            stats_out = Test(loader=Out_loder, in_dist=False)  # predict prob for out-dist data
            score_out = GetScore(stats=stats_out, method=args.method)  # score for out-dist data
            label = np.zeros(score_in.shape[0] + score_out.shape[0], dtype=np.int32)
            label[:score_in.shape[0]] += 1  # in-dist as positive class
            score = np.concatenate((score_in, score_out))
            auroc = AUROC(y_true=label, y_score=score)
            aupr = AUPR_Out(y_true=label, y_score=score)
            fpr = FPR(y_true=label, y_score=score)
            deterr = DetectErr(y_true=label, y_score=score)
            DetResult[d]["auroc"][i] = auroc
            DetResult[d]["aupr"][i] = aupr
            DetResult[d]["fpr"][i] = fpr
            DetResult[d]["deterr"][i] = deterr
            scores_assumption['out'] += list(score_out)

    # test for out-of-distribution detection performance for synthetic dataset
    SyntheticNameList = ['gaussian', 'rademacher', 'blob']
    for d in SyntheticNameList:
        print("processing {} dataset".format(d))
        DetResult[d] = {"auroc": np.zeros(args.num_eval, dtype=np.float32),
                        "aupr": np.zeros(args.num_eval, dtype=np.float32),
                        "deterr": np.zeros(args.num_eval, dtype=np.float32),
                        "fpr": np.zeros(args.num_eval, dtype=np.float32)}
        for i in range(args.num_eval):
            Out_loder = CreateSyntheticOutDistLoader(d)
            stats_out = Test(loader=Out_loder, in_dist=False)  # predict prob for out-dist data
            score_out = GetScore(stats=stats_out, method=args.method)  # score for out-dist data
            label = np.zeros(score_in.shape[0] + score_out.shape[0], dtype=np.int32)
            label[:score_in.shape[0]] += 1
            score = np.concatenate((score_in, score_out))
            auroc = AUROC(y_true=label, y_score=score)
            aupr = AUPR_Out(y_true=label, y_score=score)
            fpr = FPR(y_true=label, y_score=score)
            deterr = DetectErr(y_true=label, y_score=score)
            DetResult[d]["auroc"][i] = auroc
            DetResult[d]["aupr"][i] = aupr
            DetResult[d]["fpr"][i] = fpr
            DetResult[d]["deterr"][i] = deterr
            scores_assumption['out'] += list(score_out)

    # init "avg"
    DetResult["avg"] = {"auroc": 0.0,
                        "aupr": 0.0,
                        "deterr": 0.0,
                        "fpr": 0.0}
    AllOutdataNameList = OutdataNameList + SyntheticNameList

    # average performance for each dataset
    for d in DetResult:
        if d == 'avg':
            continue
        for m in DetResult[d]:
            DetResult[d][m] = np.mean(DetResult[d][m])
            DetResult["avg"][m] += float(DetResult[d][m])

    # store average result
    for m in DetResult["avg"]:
        DetResult["avg"][m] = DetResult["avg"][m] / len(AllOutdataNameList)

    # record to a .txt file
    AllOutdataNameList.append('avg')
    with open("{}/{}.txt".format(args.store_dir, args.method), 'w') as f:
        f.write("Dataset    AUROC    AUPR    FPR    DETERR\n")
        for d in AllOutdataNameList:
            f.write("{}  ".format(d))
            for m in ['auroc', 'aupr', 'fpr', 'deterr']:  # ensure order
                f.write("{:.2f}  ".format(DetResult[d][m]))
            f.write("\n")

    print(DetResult["avg"])

    # test our assumption using natural datasets
    # if args.assume:
    #     print("length for in: ", len(scores_assumption["in"]))
    #     print("length for out: ", len(scores_assumption["out"]))
    #     validate_assumption(score=scores_assumption, name="assume_histogram_{}.png".format(args.method))
