import torch
from scipy.stats import gmean
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def recall(output, target):
    """
    same as sensitivity
    :param output: tensor
    :param target: tensor
    :return:
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        r = recall_score(target.data.cpu().numpy(), pred.data.cpu().numpy(), average='macro')
    return r


def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        p = precision_score(target.data.cpu().numpy(), pred.data.cpu().numpy(), average='macro')
    return p


def f1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        f = f1_score(target.data.cpu().numpy(), pred.data.cpu().numpy(), average="macro")
    return f


def g_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        sensitivities = recall_score(target.data.cpu().numpy(), pred.data.cpu().numpy(), average=None, zero_division=0)
        with np.errstate(divide='ignore'):
            g = gmean(sensitivities)
    return g