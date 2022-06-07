import logging

import hydra
import ray

from srcs.main_worker import main_worker
from srcs.utils import set_seed

# fix random seeds for reproducibility
set_seed(123)

logger=logging.getLogger('train')


@hydra.main(config_path='../conf/', config_name='train')
def main(config):
    analysis = main_worker(config, logger)

    logger.info("\n".join("{}\t{}".format(k, v) for k, v in analysis.best_result.items()))


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    ray.init()

    main()
