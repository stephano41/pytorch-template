from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from hydra.utils import instantiate, get_class
from tabulate import tabulate

from src.evaluation.ftest_5x2cv import multi_combined_ftest_5x2cv
from src.loggers.result_trackers import PipelineResults
from src.utils import set_seed, loop_cfg
from src.utils.files import write_conf

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def pt_tune(config, _test_mod_func: Any = None, X=None , Y=None, output_dir=None):
    set_seed(config.seed)

    if X is None or Y is None:
        dataset = instantiate(config.dataset)
        X = dataset.data_x
        Y = dataset.data_y

    if output_dir is None:
        base_output_dir = Path(hydra.utils.HydraConfig.get().run.dir)
    else:
        base_output_dir = output_dir

    overall_results = {}
    best_models = []
    for model_cfg in loop_cfg(config, config.algorithms):
        if _test_mod_func is not None:
            model_cfg = _test_mod_func(model_cfg)

        model_output_dir = base_output_dir / model_cfg.name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        write_conf(model_cfg, model_output_dir / "model_cfg.yaml")

        if "output_dir" in inspect.signature(get_class(model_cfg.hp_search._target_)).parameters.keys():
            search_alg = instantiate(model_cfg.hp_search, instantiate(model_cfg.arch),
                                     output_dir=model_output_dir,
                                     _convert_="partial")
        else:
            search_alg = instantiate(model_cfg.hp_search, instantiate(model_cfg.arch), _convert_="partial")
        search_alg.fit(X, Y)

        best_model = search_alg.best_estimator_
        logger.info(f"{repr(best_model)} achieved {search_alg.best_score_} with {best_model.get_params()}")

        best_models.append(best_model)

        lower, upper = instantiate(model_cfg.bootstrap, best_model, X, Y)
        logger.info(f"{best_model} confidence intervals are {lower}, {upper}")

        pipeline_results = PipelineResults(model_cfg=model_cfg, best_params=best_model.get_params(),
                                           best_tune_score=search_alg.best_score_, bs_lower=lower, bs_upper=upper)
        pipeline_results.save(model_output_dir / "best_model.pth")
        overall_results[model_cfg.name] = pipeline_results

    ftest_result=None
    if len(best_models)>1:
        ftest_result = multi_combined_ftest_5x2cv(best_models, X, Y,
                                                  scoring=config.monitor, random_seed=config.seed)
        logger.info("/n" + tabulate([(k,) + v for k, v in ftest_result.items()], headers=["name", "f_stat", "p_value"]))

    return overall_results, ftest_result

# TODO update libraries
