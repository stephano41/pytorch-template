from functools import partial, update_wrapper
from importlib import import_module
from itertools import repeat

import torch
import numpy as np

import hydra


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
    if is_func:
        assert '_target_' in cfg, f'Config should have \'_target_\' for class instantiation.'
        target = cfg['_target_']
        if target is None:
            return None
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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

