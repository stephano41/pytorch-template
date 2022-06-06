import hydra
import os
import logging

from omegaconf import OmegaConf
from utils import instantiate

from ray.util.sgd.torch import TorchTrainer
os.environ["HYDRA_FULL_ERROR"]="1"

# from hydra.utils import instantiate
from ray.tune.logger import pretty_print
import ray.tune as tune

search= OmegaConf.create(
    [{"learning_rate": {"_target_": "ray.tune.loguniform", "_args_":[1e-4, 1e-2]}},
    {"batch_size": {"_target_": "ray.tune.choice", "_args_":[[1e-4, 1e-2]]}},
     {"metric": "val_accuracy"}]
)

cfg = OmegaConf.create([
   {"_target_": "torch.nn.Linear", "in_features": 3, "out_features": 4},
   {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 5},
])

@hydra.main(config_path='../conf/', config_name='tune')
def main(config):
    # OmegaConf.resolve(config)
    # dummytrain(search, arch_cfg=config)
    print(config)
    OmegaConf.resolve(config)
    print(config)


def dummytrain(config, arch_cfg):
    config = OmegaConf.create(config)
    arch_cfg = OmegaConf.merge(arch_cfg, config)
    model = instantiate(arch_cfg.arch)

    optimizer = instantiate(arch_cfg.optimizer, model.parameters())
    print(optimizer)


if __name__ == '__main__':
    main()
    # print(search)
    # print(instantiate(search))