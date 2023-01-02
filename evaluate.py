import logging
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

import src.evaluation.bootstrap
from src.evaluation.ftest_5x2cv import multi_combined_ftest_5x2cv
from src.utils import set_seed

logger = logging.getLogger(__name__)


@hydra.main(config_path='conf', config_name='evaluate', version_base='1.3')
def main(config):
    set_seed(config.seed)

    if isinstance(config.folder, str):
        model_cfgs = [torch.load(path) for path in Path(config.folder).rglob("best_model.pth")]
    else:
        model_cfgs=[]
        for folder in config.folder:
            model_cfgs.extend([torch.load(path) for path in Path(folder).rglob("best_model.pth")])

    best_models=[]
    for model_cfg in model_cfgs:
        cfg = model_cfg['config']

        best_model = instantiate(cfg.arch)
        best_model.set_params(model_cfg['best_params'])
        best_models.append(best_model)

        dataset = instantiate(cfg.dataset)
        X = dataset.data_x
        Y = dataset.data_y

        logger.info(f"{repr(best_model)}")
        lower, upper = instantiate(src.evaluation.bootstrap.bootstrap, best_model, X, Y, **src.evaluation.bootstrap.bootstrap)
        logger.info(f"{best_model} confidence intervals are {lower}, {upper}")

    # logger.info(
    #     multi_combined_ftest_5x2cv(best_models, X, Y, scoring=config.monitor,
    #                                random_seed=config.seed))

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
