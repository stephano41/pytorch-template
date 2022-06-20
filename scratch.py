import os

import hydra
from omegaconf import OmegaConf

from utils import instantiate

os.environ["HYDRA_FULL_ERROR"] = "1"
from hydra import compose
from evaluate import main as m


@hydra.main(config_path='conf/', config_name='tune', version_base='1.2')
def main(config):
    print(config)


def dummytrain(config, arch_cfg):
    config = OmegaConf.create(config)
    arch_cfg = OmegaConf.merge(arch_cfg, config)
    model = instantiate(arch_cfg.arch)

    optimizer = instantiate(arch_cfg.optimizer, model.parameters())
    print(optimizer)


if __name__ == '__main__':
    main()
    # m.__wrapped__(loaded_config)
    print("done")
