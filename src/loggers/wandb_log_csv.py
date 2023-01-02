import logging
import os
from ast import literal_eval

import ray.util.iter
import wandb
from omegaconf import OmegaConf, open_dict
from pandas import DataFrame
from ray._private.dict import flatten_dict

SUMMARY_COLS = ["mean_test_score", "std_test_score", "mean_fit_time"]

logger = logging.getLogger(__name__)


def log_df(config, df: DataFrame, num_workers=4):
    OmegaConf.resolve(config)
    if not config.get("wandb_config"):
        return
    _config = OmegaConf.to_container(config.wandb_config).copy()

    if os.name == "nt":
        os.environ["WANDB_START_METHOD"] = "thread"
    else:
        os.environ["WANDB_START_METHOD"] = "fork"

    @ray.remote
    def log(shard):
        _wandb_ids = []

        for i, row in shard:
            run_name = f"trial_{i}"

            # config = flatten_dict(literal_eval(row["params"]), delimiter='-')
            params = row["params"]
            if type(params) is str:
                params = literal_eval(params)
            with open_dict(config):
                logged_config = OmegaConf.merge(config, {"arch": params})
                logged_config = OmegaConf.to_container(logged_config)
                logged_config.pop("wandb_config")
                logged_config = flatten_dict(logged_config, delimiter='-')

            summaries = {}
            for summary_col in SUMMARY_COLS:
                if row.get(summary_col, False):
                    summaries[summary_col] = row[summary_col]
                else:
                    logger.warning(f"{summary_col} not found in result dataframe")

            summaries.update(dict(row[row.keys().str.contains("split")]))

            wandb_id = wandb.util.generate_id(10)

            wandb_kwargs = dict(
                config=logged_config,
                name=run_name,
                reinit=True,
                id=wandb_id
            )

            wandb_kwargs.update(_config)
            run = wandb.init(**wandb_kwargs)

            run.summary.update(summaries)

            run.finish()

            _wandb_ids.append(wandb_id)
        return _wandb_ids

    num_workers = min(num_workers, 10)

    it = ray.util.iter.from_items(list(df.iterrows()), num_shards=num_workers)

    wandb_ids = ray.get([log.remote(shard) for shard in it.shards()])

    return sum(wandb_ids, [])
    # for i, row in df.iterrows():
