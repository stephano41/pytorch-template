import logging
from pathlib import Path

import hydra
import ray
from hydra import compose
from omegaconf import OmegaConf
print(Path.cwd())
from evaluate import main as evaluate_main
from srcs.main_worker import main_worker
from srcs.utils import set_seed

# fix random seeds for reproducibility
set_seed(123)

logger = logging.getLogger('train')


@hydra.main(config_path='conf/', config_name='train')
def main(config):
    analysis = main_worker(config, logger)

    logger.info("\n".join("{}\t{}".format(k, v) for k, v in analysis.best_result.items()))

    best_checkpoint_dir = hydra.utils.get_original_cwd() / Path(
        analysis.best_trial.checkpoint.value) / "model_checkpoint.pth"

    evaluate_cfg = compose(config_name='evaluate', overrides=[f"checkpoint={best_checkpoint_dir}"],
                           return_hydra_config=True)
    OmegaConf.resolve(evaluate_cfg)

    evaluate_main.__wrapped__(evaluate_cfg)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    ray.init()

    main()
