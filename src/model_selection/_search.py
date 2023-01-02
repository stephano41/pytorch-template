from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
from omegaconf import OmegaConf
from ray import tune
from ray.tune import ExperimentAnalysis, CLIReporter
from ray.util.client import ray

from src.model_selection._validation import cross_validate
from src.utils import set_seed
from src.utils.files import valid_file_name


def hp_search_trainer(config, estimator, X, Y, cv=5, seed=None):
    """
    function passed into ray tune to train models
    """

    old_params = OmegaConf.create(estimator.get_params(), flags={"allow_objects": True})
    new_params = OmegaConf.merge(old_params, OmegaConf.create(config))

    neural_net = estimator.__class__(**new_params)

    tune_callback = TuneCallback(cv, monitor=re.sub(r'^.*?/', "test/", neural_net.monitor), mode=neural_net.mode)

    cross_validate(neural_net, X, Y, cv=cv, call_backs=[tune_callback], random_seed=seed)


class HPSearch:
    def __init__(self, estimator, param_grid, cv=5, seed=None, output_dir=None, **kwargs):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.seed = seed
        self.output_dir = output_dir
        self.kwargs = kwargs

        self.scoring = re.sub(r'^.*?/', "", self.estimator.monitor)
        self.mode = self.estimator.mode
        self.analysis: ExperimentAnalysis = None

    def fit(self, X, Y):
        ray.shutdown()

        try:
            if self.output_dir is None:
                self.output_dir = Path(hydra.utils.HydraConfig.get().run.dir)
            ray.init(runtime_env={"env_vars": OmegaConf.to_container(hydra.utils.HydraConfig.get().job.env_set)})
        except ValueError as e:
            if "HydraConfig was not set" in str(e):
                if self.output_dir is None:
                    self.output_dir = Path('./')
            else:
                raise e

        if self.seed:
            set_seed(self.seed)

        val_met_names = [f"test/{self.scoring}",
                         # f"predicted/{self.scoring}"
                         ]

        reporter = CLIReporter(metric_columns=["training_iteration"] + val_met_names)

        self.kwargs.update(dict(progress_reporter=reporter,
                                config=self.param_grid,
                                local_dir=self.output_dir.parent,
                                name=self.output_dir.name if self.kwargs.get("resume") is None else self.kwargs[
                                    "resume"],
                                trial_dirname_creator=trial_name,
                                resume=self.kwargs.get("resume") is not None))
        tune.Tuner()
        self.analysis = tune.run(
            tune.with_parameters(hp_search_trainer, estimator=self.estimator, X=X, Y=Y, cv=self.cv, seed=self.seed),
            **self.kwargs
        )
        ray.shutdown()

    @property
    def best_estimator_(self):
        return self.estimator.set_params(**self.best_params_)

    @property
    def best_params_(self):
        return self.analysis.get_best_config(metric=f"test/{self.scoring}", mode=self.mode, scope="all")

    @property
    def best_score_(self):
        return self.analysis.get_best_trial(metric=f"test/{self.scoring}", mode=self.mode, scope="all").metric_analysis[
            f"test/{self.scoring}"][self.mode]


def trial_name(trial):
    params = str(trial.evaluated_params)
    name = str(trial) + params
    # make it a valid file name
    name = valid_file_name(name)
    if len(name) > 50:
        name = name[:50]
    return name


class TuneCallback:
    def __init__(self, num_folds: int, monitor: str = 'test/accuracy', mode: str = 'max'):
        self.monitor = monitor
        self.mode = mode
        self.actual_scores = defaultdict(list)
        self.num_folds = num_folds

    def __call__(self, fold: int, train_scores: Dict, test_scores: Dict):
        """
        gets a set of scores after all train and test
        generate predicted scores and report predicted scores and actual scores
        tune should monitor predicted/monitor, search algorithm should monitor val/monitor
        """
        for k, v in test_scores.items():
            self.actual_scores[k].append(v)
        log = {k: np.average(v) for k, v in self.actual_scores.items()}

        tune.report(**log)