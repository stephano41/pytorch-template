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

import ray
from omegaconf import OmegaConf


from srcs.trainer.sgd_trainer import TrainableOperator
import srcs.metrics as module_metric
from pathlib import Path

# fix random seeds for reproducibility
set_seed(123)
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]="Gloo"

@hydra.main(config_path='../conf/', config_name='train')
def old_main(config):
    # have to resolve before multiprocess otherwise will bug out
    OmegaConf.resolve(config)

    # better incorporate met_funcs
    met_funcs = [instantiate(met, is_func=True) for met in config.metrics]
    met_names = ["val_" + met.__name__ for met in met_funcs]

    # TODO reporter to incorporate logger?
    reporter = CLIReporter(
        metric_columns=["training_iteration", "val_loss"] + met_names
    )
    scheduler = instantiate(config.tune_scheduler)

    trainer = TorchTrainer.as_trainable(
        config=config,
        training_operator_cls= TrainableOperator,
        **config.trainer
    )
    logger = logging.getLogger('train')

    analysis = tune.run(
        trainer,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=Path.cwd().parent,
        name=Path.cwd().name,
        metric="val_accuracy",
        mode="max",

    )

    logger.info(analysis.best_config)


@hydra.main(config_path='../conf/', config_name='train')
def main(config):
    OmegaConf.resolve(config)
    trainer = Trainer(backend='torch', num_workers=1, logdir=Path.cwd(), use_gpu=True)
    trainer.start()
    trainer.run(
        train_func=train_func,
        config=config,
    )
    trainer.shutdown()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    ray.init()

    main()
