import logging

import hydra
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

from metrics.confusion_matrix import createConfusionMatrix
from utils import instantiate

import pandas as pd

logger = logging.getLogger("sk_trainer")


def getX(s):
    s = s.dataset.tensors[0][s.indices]
    return s.reshape(s.shape[0], -1)


def getY(s):
    return s.dataset.tensors[1][s.indices]


def analyse(model, x, y, title, categories, logger, save_dir=None):
    y_pred = model.predict(x)
    cm = createConfusionMatrix(y, y_pred, classes=categories, fig_title=title)
    cm.savefig(str(save_dir/ title)+".png")
    plt.show()
    logger.info(classification_report(y, y_pred, target_names=categories))


@hydra.main(config_path='conf/', config_name='sk_train')
def main(config):
    data_loader, valid_loader = instantiate(config.train_data)
    test_loader = instantiate(config.test_data)
    print(test_loader)

    # have train and validation data together as sklearn will do cross validation
    X_train = data_loader.dataset.dataset.data
    Y_train = data_loader.dataset.dataset.targets

    X_test = test_loader.dataset.dataset.data
    Y_test = test_loader.dataset.targets

    assert config.status in ['tune', 'train', 'debug']
    if config.status == "tune" or config.status == "debug":
        optimal_params = instantiate(config.search_alg)

        optimal_params.fit(X_train, Y_train)

        logger.info(optimal_params.best_params_)

        model = optimal_params.best_estimator_

        result_csv = pd.DataFrame(optimal_params.cv_results_)
        result_csv.to_csv(config.log_dir / "result.csv")

    if config.status == "train" or config.status == "debug":
        model = instantiate(config.model)

        logger.info(model.get_params())

        model.fit(X_train, Y_train)

    analyse(model, X_train, Y_train, "Train", data_loader.categories, logger, save_dir=config.log_dir)
    analyse(model, X_test, Y_test, "Test", data_loader.categories, logger, save_dir=config.log_dir)

if __name__ == '__main__':

    main()