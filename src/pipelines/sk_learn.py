import logging

import numpy as np
from hydra.utils import instantiate

from ..utils import set_seed

logger = logging.getLogger(__name__)


def ideal_tune(config):
    set_seed(config.seed)

    data_loader = instantiate(config.data_loader)
    X = data_loader.grf_dataset.data_x.numpy()
    Y = data_loader.grf_dataset.data_y.numpy()

    # flatten data
    X = X.reshape(len(X), -1)

    # get best model
    print("Start algorithm comparison:")
    best = instantiate(config.arch_comparison, config, X=X, Y=Y)

    # do hyperparameter tuning on entire dataset
    print("start hyperparameter tuning on best found algorithm")
    search_alg = instantiate(config.hp_search, param_grid=best["param_grid"], estimator=best["model"], _convert_='all')
    search_alg.fit(X, Y)

    logger.info(f"best hyperparameters were {search_alg.best_estimator_} {search_alg.best_params_} with "
                f"{search_alg.best_score_} on {config.monitor}")

    print("start evaluation")
    # get estimation of generalisation performance
    test_scores = instantiate(config.performance_estimate,
                              model=search_alg.best_estimator_,
                              X=X, Y=Y,
                              num_cpu=config.n_cpu,
                              scoring=config.metrics,
                              _convert_="all")

    test_score = test_scores[f'test_{config.monitor}']
    logger.info(f"performance estimate of {search_alg.best_estimator_} with k-fold cross validation: "
                f"{np.average(test_score)} std: {np.std(test_score)} ")
    # logger.info(f"other scores: {test_scores}")

    if config.get("bootstrap"):
        lower, upper = instantiate(config.bootstrap,
                                   model=search_alg.best_estimator_,
                                   X=X, Y=Y)
        logger.info(f"confidence intervals are {lower}, {upper}")


# TODO add wandb

