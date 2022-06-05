from functools import partial, update_wrapper
from importlib import import_module
from itertools import repeat
from pathlib import Path
import torch
import numpy as np

import hydra
import yaml
from omegaconf import OmegaConf


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def instantiate(cfg, *args, is_func=False, **kwargs):
    """
    wrapper function for hydra.utils.instantiate.
    1. return None if config.class is None
    2. return function handle if is_func is True
    """
    assert '_target_' in cfg, f'Config should have \'_target_\' for class instantiation.'
    target = cfg['_target_']
    if target is None:
        return None
    if is_func:
        # get function handle
        modulename, funcname = target.rsplit('.', 1)
        mod = import_module(modulename)
        func = getattr(mod, funcname)

        # make partial function with arguments given in config, code
        kwargs.update({k: v for k, v in cfg.items() if k != '_target_'})
        partial_func = partial(func, *args, **kwargs)

        # update original function's __name__ and __doc__ to partial function
        update_wrapper(partial_func, func)
        return partial_func
    return hydra.utils.instantiate(cfg, *args, **kwargs)


def init_func(config,*args, **kwargs):
    target = config['_target_']
    # get function handle
    modulename, funcname = target.rsplit('.', 1)
    mod = import_module(modulename)
    func = getattr(mod, funcname)

    # make partial function with arguments given in config, code
    kwargs.update({k: v for k, v in config.items() if k != '_target_'})
    partial_func = partial(func, *args, **kwargs)

    # update original function's __name__ and __doc__ to partial function
    update_wrapper(partial_func, func)
    return partial_func


def write_yaml(content, fname):
    with fname.open('wt') as handle:
        yaml.dump(content, handle, indent=2, sort_keys=False)


def write_conf(config, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = OmegaConf.to_container(config, resolve=True)
    write_yaml(config_dict, save_path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

