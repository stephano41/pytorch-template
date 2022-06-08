import hydra
import os
import logging

from omegaconf import OmegaConf
from utils import instantiate

from ray.util.sgd.torch import TorchTrainer
os.environ["HYDRA_FULL_ERROR"]="1"
from hydra import initialize, compose
from evaluate import main as m


@hydra.main(config_path='../conf/', config_name='tune')
def main(config):
    # OmegaConf.resolve(config)
    # dummytrain(search, arch_cfg=config)
    checkpoint_dir = os.path.join(hydra.utils.get_original_cwd(), "outputs/tune-MnistLeNet/2022-06-07-13-45-54/train_func_874109d9optimizerlr-0003378905971343519/checkpoint_000005/model_checkpoint.pth")

    # with initialize(config_path='../conf'):
    cfg = compose(config_name='evaluate', overrides=[f"checkpoint={checkpoint_dir}"], return_hydra_config=True)
    OmegaConf.resolve(cfg)
    print(cfg)
    m.__wrapped__(cfg)


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