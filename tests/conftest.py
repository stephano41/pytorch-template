import pytest
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from src.utils import config_test_mod, loop_cfg

CONFIG_NAME = "tune"


@pytest.fixture(scope="function")
def nn_configs(cfg_tune, tmp_path):
    _cfg_tune = cfg_tune.copy()

    HydraConfig().set_config(_cfg_tune)

    return_list = []
    for cfg in loop_cfg(_cfg_tune, _cfg_tune.algorithms):
        return_list.append(config_test_mod(cfg))

    return return_list


@pytest.fixture(scope="function")
def cfg_tune(tmp_path):
    with initialize(version_base='1.3', config_path="../conf"):
        config = compose(config_name=CONFIG_NAME, return_hydra_config=True, overrides=[f"output_root={tmp_path}"])

        yield config

    GlobalHydra.instance().clear()

        # compose(config_name=CONFIG_NAME, overrides=["+model=cnn"])
