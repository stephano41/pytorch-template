from omegaconf import OmegaConf

from srcs.logger import Reporter
from srcs.utils import instantiate, trial_name
import ray.tune as tune
from pathlib import Path


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
        name=Path.cwd().name,
        trial_dirname_creator=trial_name
    )

    return analysis
