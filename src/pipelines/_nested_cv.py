import logging

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate

from ..utils import set_seed

logger = logging.getLogger(__name__)


def nested_cv(config, model=None, param_grid=None, X=None, Y=None, inner_folds=5, outer_folds=10):
    """
    does one nested CV on one model
    :param param_grid:
    :param outer_folds:
    :param inner_folds:
    :param config:
    :param model:
    :param X:
    :param Y:
    :return:
    """
    if model is None:
        model = instantiate(config.arch)
    if param_grid is None:
        param_grid = config.search_alg.param_grid

    cv = StratifiedKFold(n_splits=config.search_alg.get("cv", inner_folds), shuffle=True)
    clf = instantiate(config.search_alg, estimator=model, param_grid=param_grid,
                      cv=cv, n_jobs=int(config.n_cpu / 2), refit=True, _convert_='all')

    return train(clf, X, Y, folds=outer_folds, scoring=config.monitor, num_cpu=2)


def compare_nested_cv(config, X, Y, inner_folds=5, outer_folds=10):
    if type(X) == type(None) or type(Y) is type(None):
        dataset = instantiate(config.dataset)
        X = dataset.data_x
        Y = dataset.data_y

    model_confs = [OmegaConf.load(model_yaml) for model_yaml in config.algorithms]
    models = [instantiate(model_conf.arch) for model_conf in model_confs]
    param_grids = [model_conf.search_alg.param_grid for model_conf in model_confs]

    # do nested_cv and compare between average for best
    best_model = None
    best_model_param_grid = None
    best_average = 0
    best_std = 0
    for model, param_grid in zip(models, param_grids):
        set_seed(config.seed)
        nested_cv_score = nested_cv(config, model, param_grid, X, Y, inner_folds=inner_folds, outer_folds=outer_folds)
        test_score = nested_cv_score["test_score"]
        average = np.average(test_score)
        std = np.std(test_score)
        logger.info(f"{model} on nested kfold achieved: {average} STD {std}")
        logger.info(f"Other scores: {nested_cv_score}")
        if average > best_average:
            best_model = model
            best_model_param_grid = param_grid
            best_std = std
            best_average = average
    return {"model": best_model, "param_grid": best_model_param_grid, "score": best_average, "std": best_std}


def repeated_kfold(model, X, Y, n_splits=5, n_repeats=10, num_cpu=2, scoring=None):
    assert isinstance(scoring, list)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    score = cross_validate(model, X, Y, n_jobs=num_cpu, scoring=scoring, cv=cv, verbose=1)
    return score


def train(model, X, Y, folds=5, scoring=None, num_cpu=1) -> dict:
    """
    stratified k-folds a model in config,
    by default will use all data to k-fold, specify X and Y to override this
    :param num_cpu:
    :param scoring:
    :param folds:
    :param model:
    :param X:
    :param Y:
    :return:
    """
    # set_seed(config.get("seed"))
    logger.info(f"training {model}")

    cv = StratifiedKFold(n_splits=folds, shuffle=True)

    score = cross_validate(model, X, Y, cv=cv,
                           n_jobs=num_cpu,
                           scoring=scoring)

    return score
