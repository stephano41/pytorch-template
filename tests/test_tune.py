import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.pipelines import pt_tune
from src.utils import config_test_mod

from .conftest import cfg_tune


@pytest.mark.slow
def test_dryrun(cfg_tune):
    HydraConfig().set_config(cfg_tune)
    with open_dict(cfg_tune):
        cfg_tune.pop("hydra")
    pt_tune(cfg_tune, config_test_mod)