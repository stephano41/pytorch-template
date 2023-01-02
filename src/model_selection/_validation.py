from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ..utils import set_seed

if TYPE_CHECKING:
    from ..trainer.neural_net import NeuralNet


def cross_validate(neural_net: NeuralNet, X, Y, cv=5, random_seed=None, call_backs=None):
    splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_seed)
    if random_seed is not None:
        set_seed(random_seed)

    train_scores = []
    test_scores = []

    for fold, (train_idx, test_idx) in enumerate(splits.split(np.arange(len(X)), Y)):
        fit_results = neural_net.fit(X[train_idx], Y[train_idx])
        test_score = neural_net.score(X[test_idx], Y[test_idx])

        train_scores.append(fit_results.scores)
        test_scores.append(test_score)
        if call_backs is not None:
            for call_back in call_backs:
                call_back(fold, fit_results.scores, test_score)

    return CVResult(train_scores, test_scores)


class CVResult:
    __slots__ = ['train_scores', 'test_scores']

    def __init__(self, train_scores, test_scores):
        def unpack_score(packed_dict):
            return_d = defaultdict(list)
            for d in packed_dict:
                for k, v in d.items():
                    return_d[k].append(v)
            return return_d

        self.train_scores = unpack_score(train_scores)
        self.test_scores = unpack_score(test_scores)

        # self.scores = {key: np.array(value) for key, value in self.scores.items()}

    @property
    def avg_test(self):
        return {key: np.average(value) for key, value in self.test_scores}

    @property
    def std_test(self):
        return {key: np.std(value) for key, value in self.test_scores}
