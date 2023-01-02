import logging

import numpy as np
from scipy.stats import gmean
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

from .util import np_one_hot

logger = logging.getLogger(__name__)


def accuracy(target, output):
    correct = 0
    correct += np.sum(output == target)
    return correct / len(target)


def recall(target, output, average='macro'):
    """
    same as sensitivity
    :param output: tensor
    :param target: tensor
    :return:
    """
    r = recall_score(target, output, average=average)
    return r


def precision(target, output, average='macro'):
    p = precision_score(target, output, average=average)
    return p


def f1(target, output, average='macro'):
    f = f1_score(target, output, average=average)
    return f


def g_score(target, output, average=None):
    sensitivities = recall_score(target, output, average=average, zero_division=0)
    with np.errstate(divide='ignore'):
        g = gmean(sensitivities)
    return g


def roc_auc(target, output, average="macro", multi_class="ovr"):
    num_classes = (np.concatenate([output, target])).max() + 1

    one_hot_target = np_one_hot(target, num_classes=num_classes)

    one_hot_output = np_one_hot(output, num_classes=num_classes)

    try:
        score = roc_auc_score(one_hot_target, one_hot_output, average=average, multi_class=multi_class)
    except ValueError as e:
        if 'Only one class present' not in str(e):
            raise ValueError(e)
        logger.warning("Only one class present in y_true. ROC AUC score is not defined in that case")
        score = 0

    return score
