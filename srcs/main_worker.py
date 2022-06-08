from pathlib import Path

import ray.tune as tune
from omegaconf import OmegaConf

from srcs.logger import Reporter
from srcs.utils import instantiate
from srcs.utils.tune import trial_name


def main_worker(config, logger):
    OmegaConf.resolve(config)

    met_funcs = [instantiate(met, is_func=True) for met in config.metrics]
    met_names = ["val_" + met.__name__ for met in met_funcs]

    reporter = Reporter(
        logger,
        metric_columns=["training_iteration", "val_loss"] + met_names
    )

    analysis = tune.run(
        tune.with_parameters(instantiate(config.trainer.train_func, is_func=True), arch_cfg=config),
        **OmegaConf.to_container(instantiate(config.trainer.run)),
        progress_reporter=reporter,
        local_dir=Path.cwd().parent,
        name=Path.cwd().name if config.resume is None else config.resume,
        trial_dirname_creator=trial_name,
        resume = config.resume is not None
    )

    return analysis
