import logging

import hydra

from srcs.trainer import Trainer
from srcs.utils import instantiate, set_seed
from ray.util.sgd.torch import TorchTrainer
from ray.tune.logger import pretty_print
import ray
from omegaconf import OmegaConf

from trainer.sgd_trainer import GRFTrainingOperator

# fix random seeds for reproducibility
set_seed(123)


@hydra.main(config_path='../conf/', config_name='train')
def main(config):
    # have to resolve before multiprocess otherwise will bug out
    OmegaConf.resolve(config)

    trainer = TorchTrainer(
        training_operator_cls=GRFTrainingOperator,
        scheduler_step_freq="epoch",
        config=config,
        use_gpu=True,
        use_tqdm=True
    )
    logger = logging.getLogger('train')
    for i in range(10):
        logger.info(pretty_print(trainer.train()))
        logger.info(pretty_print(trainer.validate()))

    # print(metrics, val_metrics)

    trainer.shutdown()

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    ray.init()

    main()
