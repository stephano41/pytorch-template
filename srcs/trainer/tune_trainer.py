import os
import logging
from pathlib import Path

from utils import instantiate
from ray import train
# from ray import tune
import torch
from srcs.logger import BatchMetrics
from srcs.trainer.base import save_checkpoint, prepare_devices
from hydra.utils import get_original_cwd


def train_func(config):
    # os.chdir(get_original_cwd())

    logger = logging.getLogger("train")

    device, device_ids = prepare_devices(config['n_gpu'])

    # setup dataloaders
    data_loader, valid_data_loader = instantiate(config['data_loader'])

    # setup model
    model = instantiate(config['arch'])
    logger.info(model)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')

    model = model.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = instantiate(config['loss'], is_func=True)

    optimizer = instantiate(config['optimizer'], model.parameters())

    lr_scheduler = None
    if "lr_scheduler" in config:
        lr_scheduler = instantiate(config['lr_scheduler'], optimizer)

    # later changed if checkpoint
    start_epoch=0

    checkpoint = train.load_checkpoint()
    if checkpoint:
        model.load_state_dict(checkpoint.get("state_dict"))
        optimizer.load_state_dict(checkpoint.get("optimizer"))
        start_epoch=checkpoint.get("epoch",-1)+1

    metric_ftns = [instantiate(met, is_func=True) for met in config['metrics']]
    train_metrics = BatchMetrics('loss', *[m.__name__ for m in metric_ftns])
    valid_metrics = BatchMetrics('loss', *[m.__name__ for m in metric_ftns])

    for epoch in range(start_epoch, config['trainer']['epochs']):  # loop over the dataset multiple times
        train_metrics.reset()
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_metrics.update('loss', loss.item())
            for met in metric_ftns:
                train_metrics.update(met.__name__, met(outputs, targets))

        valid_metrics.reset()
        for i, data in enumerate(valid_data_loader, 0):
            with torch.no_grad():
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, targets)
                valid_metrics.update('loss', loss.item())
                for met in metric_ftns:
                    valid_metrics.update(met.__name__, met(outputs, targets))

        if lr_scheduler is not None:
            lr_scheduler.step()

        state = {
            'arch': type(model).__name__,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        train.save_checkpoint(**state)

        log = train_metrics.result()
        val_log = valid_metrics.result()
        log.update(**{'val_'+k : v for k, v in val_log.items()})
        train.report(**log)