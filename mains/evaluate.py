import logging

import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from logger import BatchMetrics
from srcs.metrics.confusion_matrix import createConfusionMatrix
from srcs.utils import instantiate

logger = logging.getLogger('evaluate')


@hydra.main(config_path='../conf', config_name='evaluate')
def main(config):
    logger.info("Test start")
    logger.info('Loading checkpoint: {} ...'.format(config.checkpoint))
    checkpoint = torch.load(config.checkpoint)

    loaded_config = OmegaConf.create(checkpoint['config'])

    # setup data_loader instances
    data_loader = instantiate(config.data_loader)

    # restore network architecture
    model = instantiate(loaded_config.arch)
    logger.info(model)

    # load trained weights
    state_dict = checkpoint['state_dict']
    if loaded_config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # instantiate loss and metrics
    criterion = instantiate(loaded_config.loss, is_func=True)
    metric_ftns = [instantiate(met, is_func=True) for met in loaded_config.metrics]
    test_metrics = BatchMetrics('loss', *[m.__name__ for m in metric_ftns])

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_metrics.reset()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for i, data in enumerate(tqdm(data_loader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.data.cpu().numpy())
            all_targets.extend(targets.data.cpu().numpy())

            test_metrics.update('loss', loss.item())
            for met in metric_ftns:
                test_metrics.update(met.__name__, met(outputs, targets))

    logger.info(test_metrics.result())

    cm = createConfusionMatrix(all_targets, all_preds, fig_title="Test confusion matrix")
    cm.savefig("Test confusion matrix.png")
    cm.show()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
