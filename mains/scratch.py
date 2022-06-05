import hydra
import os
import logging

from omegaconf import OmegaConf

from ray.util.sgd.torch import TorchTrainer
os.environ["HYDRA_FULL_ERROR"]="1"

# from hydra.utils import instantiate


@hydra.main(config_path='../conf/', config_name='train')
def main(config):
    OmegaConf.resolve(config)
    logger = logging.getLogger('train')

    logger.info(config.trainer)
    # print(trainer)

if __name__ == '__main__':
    main()