import logging
from pathlib import Path

import hydra
import ray

from srcs.main_worker import main_worker
from srcs.utils import set_seed
from hydra import compose
from omegaconf import OmegaConf
from evaluate import main as evaluate_main

# fix random seeds for reproducibility
set_seed(123)

logger = logging.getLogger("tune")


@hydra.main(config_path='../conf/', config_name='tune')
def main(config):
    analysis = main_worker(config, logger)

    # TODO make resume work
    # TODO clean up config structure
    # TODO add evaluation at the end
    # TODO add confusion matrices
    logger.info(analysis.best_config)
    best_trial = analysis.best_trial
    best_checkpoint_dir = Path(best_trial.checkpoint.value)

    logger.info("Best trial location: {}".format(best_checkpoint_dir))
    logger.info("Best trial params: {}".format(best_trial.evaluated_params))
    logger.info("\n".join("{}\t{}".format(k, v) for k, v in analysis.best_result.items()))

    # start test
    best_checkpoint_dir = hydra.utils.get_original_cwd() / best_checkpoint_dir / "model_checkpoint.pth"

    evaluate_cfg = compose(config_name='evaluate', overrides=[f"checkpoint={best_checkpoint_dir}"], return_hydra_config=True)
    OmegaConf.resolve(evaluate_cfg)

    evaluate_main.__wrapped__(evaluate_cfg)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    ray.init()

    main()
