import logging

import hydra
from ray.tune import CLIReporter

from srcs.utils import instantiate, set_seed

import ray.tune as tune

import ray
from omegaconf import OmegaConf

from pathlib import Path

# fix random seeds for reproducibility
set_seed(123)

@hydra.main(config_path='../conf/', config_name='train')
def main(config):
    OmegaConf.resolve(config)

    met_funcs = [instantiate(met, is_func=True) for met in config.metrics]
    met_names = ["val_" + met.__name__ for met in met_funcs]

    # TODO reporter to incorporate logger?
    reporter = CLIReporter(
        metric_columns=["training_iteration", "val_loss"] + met_names
    )
    scheduler = instantiate(config.tune_scheduler)
    logger = logging.getLogger('train')

    conf = OmegaConf.to_container(config, resolve=True)

    analysis = tune.run(
            tune.with_parameters(instantiate(config.trainer.train_func, is_func=True), arch_cfg=config),
            **config.trainer.run,
            progress_reporter=reporter,
            scheduler= scheduler,
            local_dir=Path.cwd().parent,
            name=Path.cwd().name
                           )

    logger.info(analysis.best_config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    ray.init()

    main()
