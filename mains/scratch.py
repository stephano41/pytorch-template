import hydra
import os

from omegaconf import OmegaConf

from ray.util.sgd.torch import TorchTrainer
os.environ["HYDRA_FULL_ERROR"]="1"

# from hydra.utils import instantiate


@hydra.main(config_path='../conf/', config_name='train')
def main(config):
    OmegaConf.resolve(config)

    trainer = TorchTrainer.as_trainable(
        config=config,
        **config.trainer
    )
    print(trainer)
    # trainer = trainer(config=config)
    # print(trainer)

if __name__ == '__main__':
    main()