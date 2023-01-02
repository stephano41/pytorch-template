import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from ray._private.dict import flatten_dict
from tabulate import tabulate

from ..utils import set_seed
from .pytorch import pt_tune

logger = logging.getLogger(__name__)


def get_sample_size(config, _test_mod_func: Any = None):
    set_seed(config.seed)

    dataset = instantiate(config.dataset)
    X = dataset.data_x
    Y = dataset.data_y

    sample_sizes = config.sample_sizes

    base_output_dir = Path(hydra.utils.HydraConfig.get().run.dir)

    all_results = {}
    for sample_size in sample_sizes:
        run_output_dir = base_output_dir / str(sample_size)
        run_output_dir.mkdir(parents=True, exist_ok=True)

        sub_sample_idx = np.random.randint(0, len(X), int(len(X) * sample_size))
        sub_sample_x = X[sub_sample_idx]
        sub_sample_y = Y[sub_sample_idx]

        results, _ = pt_tune(config, _test_mod_func, X=sub_sample_x, Y=sub_sample_y, output_dir=run_output_dir)

        # wandb.log(results, step=len(sub_sample_idx))
        logger.info(f"sample_size {len(sub_sample_idx)}:\n" + tabulate([[model_name, f"{result.bs_lower}~{result.bs_upper}", result.best_tune_score] for model_name, result in results.items()],
                                                                       headers=["model","bootstrap", "tune score"])
        )
        all_results[len(sub_sample_idx)] = results

    # START LOGGING

    for sample_size, results in all_results.items():
        for model_name, result in results.items():
            logged_config = OmegaConf.to_container(result.model_cfg, resolve=True)
            logged_config.pop("wandb_config")
            logged_config = flatten_dict(logged_config, delimiter='-')
            wandb.init(**{**config.wandb_config, "config": logged_config, "allow_val_change": True, "name": model_name,
                          "group": str(sample_size)})
            wandb.log({"boostrap_lower": result.bs_lower, "bootstrap_upper": result.bs_upper})
            wandb.finish()
