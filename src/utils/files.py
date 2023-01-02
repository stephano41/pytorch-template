import contextlib
import os
from pathlib import Path
import re

import yaml
from omegaconf import OmegaConf


def write_yaml(content, fname):
    with fname.open('wt') as handle:
        yaml.dump(content, handle, indent=2, sort_keys=False)


def write_conf(config, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = OmegaConf.to_container(config, resolve=True)
    write_yaml(config_dict, save_path)


def valid_file_name(path: str) -> str:
    # path = os.path.splitext(path)[0]
    name = re.sub(r'[^\w\s-]', '', path.lower())
    name = re.sub(r'[-\s]+', '-', name).strip('-_')
    return name

@contextlib.contextmanager
def change_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    if path is None:
        path = prev_cwd

    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
