from __future__ import annotations

from collections import deque


class EarlyStopper:
    __slots__ = ['_mode', '_grace_period', '_n_iter_no_change', '_iter', '_trial_results', '_modifier']

    def __init__(self, mode: str, grace_period: int = 5, n_iter_no_change: int = 10):
        """ Early stop single trials when they don't improve for num_results

        :param metric: Metric to monitor for
        :param mode: whether metric is to be minimised or maximised, default max. must be one of [max, min]
        :param grace_period: Minimum number of timesteps before a trial can be early stopped
        :param num_results: Number of results to consider for improvement
        """
        if mode is None:
            mode = 'max'
        assert mode in ['max', 'min'], 'mode must be min or max'

        self._mode = mode
        self._grace_period = grace_period
        self._n_iter_no_change = n_iter_no_change

        self._iter = 0
        self._trial_results = deque(maxlen=self._n_iter_no_change)

        self._modifier = 1. if self._mode == 'max' else -1.

    def __call__(self, result):
        self._trial_results.append(result)
        self._iter += 1

        # If still in grace period, do not stop yet
        if self._iter < self._grace_period:
            return False

        # If not enough results yet, do not stop yet
        if len(self._trial_results) < self._n_iter_no_change:
            return False

        # using modifier is slightly faster than if statements
        return self._trial_results[0] * self._modifier >= self._trial_results[-1] * self._modifier
