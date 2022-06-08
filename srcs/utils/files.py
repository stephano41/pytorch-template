from pathlib import Path

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