import hydra
import os
import logging

from omegaconf import OmegaConf
from utils import instantiate

from ray.util.sgd.torch import TorchTrainer
os.environ["HYDRA_FULL_ERROR"]="1"

# from hydra.utils import instantiate
from ray.tune.logger import pretty_print

search={
    "learning_rate": 0.005,
    "batch_size":128
}

@hydra.main(config_path='../conf/', config_name='tune')
def main(config):
    # OmegaConf.resolve(config)
    # dummytrain(search, arch_cfg=config)
    print(config)


def dummytrain(config, arch_cfg):
    config = OmegaConf.create(config)
    arch_cfg = OmegaConf.merge(arch_cfg, config)
    model = instantiate(arch_cfg.arch)

    optimizer = instantiate(arch_cfg.optimizer, model.parameters())
    print(optimizer)


if __name__ == '__main__':
    main()