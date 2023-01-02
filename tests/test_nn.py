import copy
from datetime import datetime

from hydra.utils import instantiate
from sklearn import clone
from sklearn.model_selection import train_test_split
from src.utils import set_seed

from .conftest import nn_configs

def _get_XY_nn(cfg):
    dataset = instantiate(cfg.dataset)
    X = dataset.data_x
    Y = dataset.data_y

    neural_net = instantiate(cfg.arch)

    return X, Y, neural_net


def test_basic_routine(nn_configs):
    for nn_config in nn_configs:
        X, Y, neural_net = _get_XY_nn(nn_config)

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

        neural_net.fit(train_x, train_y)

        neural_net.score(test_x, test_y)


def test_nn_not_change_x(nn_configs):
    """
    test to see if neural_net fit function changes the input
    :return:
    """

    def run(cfg):
        X, Y, neural_net = _get_XY_nn(cfg)

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
        old_train_x = copy.deepcopy(train_x)

        neural_net.fit(train_x, train_y)

        assert (old_train_x == train_x).all()

        old_test_x = copy.deepcopy(test_x)

        neural_net.score(test_x, test_y)

        assert (old_test_x == test_x).all()

    for nn_config in nn_configs:
        run(nn_config)


def test_repeated_score(nn_configs):
    for nn_config in nn_configs:
        X, Y, neural_net = _get_XY_nn(nn_config)

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

        neural_net.fit(train_x, train_y)

        score1 = neural_net.score(test_x, test_y)
        score2 = neural_net.score(test_x, test_y)
        score3 = neural_net.score(test_x, test_y)

        assert str(score1) == str(score2)
        assert str(score2) == str(score3)


def test_clone_repeatability(nn_configs):
    """
    test to see if performance is the same with the same seed after cloning
    check performance
    :param nn_config:
    :return:
    """
    for nn_config in nn_configs:
        dataset = instantiate(nn_config.dataset)
        X = dataset.data_x
        Y = dataset.data_y

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

        SEED=datetime.now().minute

        set_seed(SEED)

        neural_net = instantiate(nn_config.arch)
        neural_net.fit(train_x, train_y)
        score1 = neural_net.score(test_x, test_y)

        set_seed(SEED)

        new_model = clone(neural_net)
        new_model.fit(train_x, train_y)
        score2 = new_model.score(test_x, test_y)

        assert str(score1)==str(score2)
