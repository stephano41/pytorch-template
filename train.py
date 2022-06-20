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


@hydra.main(config_path='conf/', config_name='train', version_base='1.2')
def main(config):
    output_dir = Path(hydra.utils.HydraConfig.get().run.dir)

    analysis = main_worker(config, output_dir, logger)

    logger.info("\n".join(f"{k}\t{v}" for k, v in analysis.best_result.items()))

    best_checkpoint_dir = analysis.best_trial.checkpoint.value

    overrides = [f"checkpoint_dir={best_checkpoint_dir}",
                 "checkpoint_name=model_checkpoint.pth",
                 f"hydra.run.dir={str(output_dir)}/test-{analysis.best_trial.trial_id}"
                 ]

    evaluate_cfg = compose(config_name='evaluate', overrides=overrides, return_hydra_config=True)
    OmegaConf.resolve(evaluate_cfg)

    evaluate_main.__wrapped__(evaluate_cfg, cm_title=f'{analysis.best_trial.trial_id} Confusion Matrix')


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    ray.init()

    main()
