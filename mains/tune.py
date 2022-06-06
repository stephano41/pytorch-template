import logging

import hydra
# from ray.tune.function_runner import StatusReporter

from srcs.utils import instantiate, set_seed, trial_name

import ray.tune as tune

import ray
from omegaconf import OmegaConf

from pathlib import Path
from srcs.logger import Reporter

# fix random seeds for reproducibility
set_seed(123)

logger = logging.getLogger("tune")

@hydra.main(config_path='../conf/', config_name='tune')
def main(config):
    OmegaConf.resolve(config)

    met_funcs = [instantiate(met, is_func=True) for met in config.metrics]
    met_names = ["val_" + met.__name__ for met in met_funcs]

    reporter = Reporter(
        logger,
        metric_columns=["training_iteration", "val_loss"] + met_names
    )

    analysis = tune.run(
            tune.with_parameters(instantiate(config.trainer.train_func, is_func=True), arch_cfg=config),
            **OmegaConf.to_container(instantiate(config.trainer.run)),
            progress_reporter=reporter,
            local_dir=Path.cwd().parent,
            name=Path.cwd().name,
            trial_dirname_creator=trial_name
                           )
    # TODO make resume work
    # TODO add evaluation at the end
    # TODO add confusion matrices
    logger.info(analysis.best_config)
    best_trial = analysis.best_trial
    best_checkpoint_dir = Path(best_trial.checkpoint.value)

    logger.info("Best trial location: {}".format(best_checkpoint_dir))
    logger.info("Best trial params: {}".format(best_trial.evaluated_params))
    logger.info("\n".join("{}\t{}".format(k, v) for k, v in analysis.best_result.items()))


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    ray.init()

    main()
