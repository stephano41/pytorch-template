import logging

import hydra

from srcs.trainer import Trainer
from srcs.utils import instantiate, set_seed
from ray.util.sgd.torch import TrainingOperator, TorchTrainer
from ray.tune.logger import pretty_print
import ray
from omegaconf import OmegaConf

# fix random seeds for reproducibility
set_seed(123)


class MyTrainingOperator(TrainingOperator):
    def setup(self, config):
        logger = logging.getLogger('train')
        # setup data_loader instances
        data_loader, valid_data_loader = instantiate(config.data_loader)

        # build model. print it's structure and # trainable params.
        model = instantiate(config.arch)
        logger.info(model)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')

        # get function handles of loss and metrics
        criterion = instantiate(config.loss, is_func=True)
        metrics = [instantiate(met, is_func=True) for met in config['metrics']]

        # build optimizer, learning rate scheduler.
        optimizer = instantiate(config.optimizer, model.parameters())
        lr_scheduler = instantiate(config.lr_scheduler, optimizer)
        self.model, self.optimizer, self.criterion, self.scheduler=\
            self.register(models=model, optimizers=optimizer, criterion=criterion, schedulers=lr_scheduler)
        self.register_data(train_loader=data_loader, validation_loader=valid_data_loader)


@hydra.main(config_path='../conf/', config_name='train')
def main(config):
    # have to resolve before multiprocess otherwise will bug out
    OmegaConf.resolve(config)

    trainer = TorchTrainer(
        training_operator_cls=MyTrainingOperator,
        scheduler_step_freq="epoch",
        config=config,
        use_gpu=True
    )
    logger = logging.getLogger('train')
    for i in range(10):
        logger.info(pretty_print(trainer.train(profile=True)))
        logger.info(pretty_print(trainer.validate(profile=True)))

    # print(metrics, val_metrics)

    trainer.shutdown()

logger = logging.getLogger('train')
@hydra.main(config_path='../conf/', config_name='train')
def old_main(config):
    # setup data_loader instances
    data_loader, valid_data_loader = instantiate(config.data_loader)

    # build model. print it's structure and # trainable params.
    model = instantiate(config.arch)
    logger.info(model)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')

    # get function handles of loss and metrics
    criterion = instantiate(config.loss, is_func=True)
    metrics = [instantiate(met, is_func=True) for met in config['metrics']]

    # build optimizer, learning rate scheduler.
    optimizer = instantiate(config.optimizer, model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    old_main()
    # ray.init()
    # main()
