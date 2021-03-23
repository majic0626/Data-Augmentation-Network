# metric.py
# some function to calculate the performance on out-of-distribution detection
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def FPR(y_true, y_score):
    # calculate the falsepositive error when tpr is 95%
    Y1 = y_score[y_true.sum():]  # other
    X1 = y_score[:y_true.sum()]  # cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1), np.min(Y1)])
    gap = (end - start) / 100000  # precision:100000

    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(X1 >= delta) / np.float(len(X1))
        error2 = np.sum(Y1 > delta) / np.float(len(Y1))
        if tpr <= 0.96 and tpr >= 0.94:  # 0.95
            fpr += error2
            total += 1
    if total == 0:
        print('corner case')
        fprBase = 1
    else:
        fprBase = fpr / total

    return 100 * fprBase


def AUPR_In(y_true, y_score):
    """
    out-of-distribution data as negative!
    input: y_true, y_score
    output: AUPR score

    """
    return 100 * average_precision_score(y_true, y_score)


def AUPR_Out(y_true, y_score):
    """
    out-of-distribution data as positive!
    input: y_true, y_score
    output: AUPR score

    """
    return 100 * average_precision_score(1 - y_true, 1 - y_score, pos_label=1)


def AUROC(y_true, y_score):
    """
    input: y_true, y_score
    output: AUROC score

    """
    return 100 * roc_auc_score(y_true, y_score)


def DetectErr(y_true, y_score):
    # suppose P(A) = P(B) = 0.5
    # P(Err) = P(A n B) + P(A n B) = P(Err|A)P(A) + P(Err|B)P(B)

    # cifar = np.loadtxt('%s/confidence_Base_In.txt'%dir_name, delimiter=',')
    # other = np.loadtxt('%s/confidence_Base_Out.txt'%dir_name, delimiter=',')
    Y1 = y_score[y_true.sum():]  # other
    X1 = y_score[:y_true.sum()]  # cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1), np.min(Y1)])
    gap = (end - start) / 100000

    err = 1.0
    prior = 0.5
    for delta in np.arange(start, end, gap):
        err1 = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        err2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        err = np.minimum(err, (err1 + err2) * prior)

    # detection accuracy = 1-err
    # detection err = err
    return 100 * err
