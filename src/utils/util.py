import logging
import os
from collections.abc import Mapping

import random
from typing import Dict

import numpy as np
import torch
from hydra import compose
from omegaconf import DictConfig, open_dict, OmegaConf

logger = logging.getLogger('util')


def update_nested_dict(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def loop_cfg(main_cfg, cfg_list):
    for yaml_path in cfg_list:
        if isinstance(yaml_path, (DictConfig, Dict)):
            sub_cfg = yaml_path
        else:
            sub_cfg = compose(yaml_path)
        with open_dict(main_cfg), open_dict(sub_cfg):
            merged_cfg = OmegaConf.merge(main_cfg, sub_cfg)
        OmegaConf.set_struct(merged_cfg, True)
        yield merged_cfg

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def prepare_devices(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine,"
                       "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                       "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def config_test_mod(config):
    with open_dict(config):
        if config["hp_search"].get("num_samples"):
            config["hp_search"]["num_samples"] = 3
        else:
            d = config["hp_search"]["param_grid"]
            config["hp_search"]["param_grid"] = {next(iter(d)): d[next(iter(d))]}
        if config["arch"].get("epochs"):
            config["arch"]["epochs"] = 1
        if config.get("bootstrap"):
            config["bootstrap"]["iters"] = 3
    return config