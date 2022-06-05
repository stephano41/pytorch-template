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
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]="gloo"

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

    trainer = Trainer(backend='torch', num_workers=1, logdir=os.getcwd(), use_gpu=True)
    trainable = trainer.to_tune_trainable(train_func)
    # trainer.start()
    # trainer.run(
    #     train_func=train_func,
    #     config=config,
    # )
    # trainer.shutdown()
    # better incorporate met_funcs
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
        trainable,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=Path.cwd().parent,
        name=Path.cwd().name,
        metric="val_accuracy",
        mode="max",
        config=conf,


    )

    logger.info(analysis.best_config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    ray.init()

    main()
