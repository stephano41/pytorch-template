from __future__ import annotations

import torch
from sklearn.base import BaseEstimator, TransformerMixin


class TorchStandardScaler:
    __slots__ = ['mean', 'std', '_dim', '_epsilon']

    def __init__(self, mean=None, std=None, dim=0, epsilon=1e-7):
        self.mean = mean
        self.std = std
        self._dim = dim
        self._epsilon = epsilon

    def fit(self, x, y=None):
        self.mean = torch.mean(x, dim=self._dim)
        self.std = torch.std(x, unbiased=False, dim=self._dim)

    def transform(self, x, y=None):
        return (x - self.mean) / (self.std + self._epsilon)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class Flatten(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.numpy().reshape(len(X), -1)
