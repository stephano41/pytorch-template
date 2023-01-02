from __future__ import annotations

import logging
import os.path
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import get_scorer

logger = logging.getLogger('logger')


class MetricTracker:
    __slots__ = ['keys', '_data']

    def __init__(self, *keys):
        # self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.keys = keys
        self._data: Dict[str:List] = None
        self.reset()

    def reset(self):
        self._data = {key: [] for key in self.keys}

    def update(self, key: str, value: np.float):
        self._data[key].append(value)

    def result(self):
        return {key: np.average(value) for key, value in self._data.items()}


class MetricCollection:
    __slots__ = ['metric_funcs', 'prefix', 'postfix', 'include_loss', 'metric_tracker']

    def __init__(self, metric_names: List[str | Dict] | str, include_loss=False, prefix='', postfix=''):
        self.metric_funcs: Dict[str: partial] = {}
        self.prefix = prefix
        self.postfix = postfix
        self.include_loss = include_loss

        if isinstance(metric_names, str):
            scorer = get_scorer(metric_names)
            self.metric_funcs.update(
                {self.prefix + metric_names + self.postfix: partial(scorer._score_func, **scorer._kwargs)})
        else:
            for f in metric_names:
                if isinstance(f, (Dict, DictConfig)):
                    scorer = instantiate(f, _partial_=True)
                    self.metric_funcs.update({self.prefix + scorer.func.__name__ + self.postfix: scorer})
                else:
                    scorer = get_scorer(f)
                    self.metric_funcs.update(
                        {self.prefix + f + self.postfix: partial(scorer._score_func, **scorer._kwargs)})

        if self.include_loss:
            self.metric_tracker = MetricTracker(
                *(list(self.metric_funcs.keys()) + [self.prefix + 'loss' + self.postfix]))
        else:
            self.metric_tracker = MetricTracker(*list(self.metric_funcs.keys()))

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor, loss=None):
        if self.include_loss:
            self.metric_tracker.update(self.prefix + 'loss' + self.postfix, loss.item())
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        # y_pred = np.argmax(y_pred, axis=1)
        assert y_pred.shape[0] == len(y_true)
        for name, met in self.metric_funcs.items():
            result = met(y_true, y_pred)
            self.metric_tracker.update(name, result)

    def result(self):
        return self.metric_tracker.result()

    def reset(self):
        self.metric_tracker.reset()


@dataclass
class FitResults:
    keys: List[str] = field(repr=False)
    scores: Dict[str:List] = field(init=False)

    def __post_init__(self):
        self.scores = {key: [] for key in self.keys}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.scores[key].append(value)


@dataclass
class PipelineResults:
    model_cfg: Dict | DictConfig
    best_params: Dict | DictConfig
    best_tune_score: float
    bs_lower: float
    bs_upper: float

    def __post_init__(self):
        self._saved_dir = None

    def save(self, output_dir):
        torch.save(self, output_dir)
        if isinstance(output_dir, Path):
            self._saved_dir = output_dir
        else:
            self._saved_dir = Path(output_dir)

    @property
    def saved_dir(self):
        return self._saved_dir
