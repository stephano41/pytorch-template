import logging
import os

import hydra
from ray.tune import CLIReporter
from ray.train import Trainer
# from srcs.trainer import Trainer
from srcs.utils import instantiate, set_seed
from ray.util.sgd.torch import TorchTrainer
from srcs.trainer.tune_trainer import train_func
import ray.tune as tune
from srcs.trainer.base import prepare_devices
import ray
from omegaconf import OmegaConf, open_dict


from srcs.trainer.sgd_trainer import TrainableOperator
import srcs.metrics as module_metric
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
        train_func,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=Path.cwd().parent,
        name=Path.cwd().name,
        metric="val_accuracy",
        mode="max",
        config=conf,
        resources_per_trial={"cpu": conf["n_cpu"], "gpu": conf["n_gpu"]}


    )

    logger.info(analysis.best_config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    ray.init()

    main()
