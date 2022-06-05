import os
import logging
from pathlib import Path

from utils import instantiate
from ray import tune
import torch
from srcs.logger import BatchMetrics
from srcs.trainer.base import save_checkpoint, prepare_devices
from hydra.utils import get_original_cwd


def train_func(config, arch_cfg, checkpoint_dir=None):
    # os.chdir(get_original_cwd())

    logger = logging.getLogger("train")

    device, device_ids = prepare_devices(arch_cfg.n_gpu)

    # setup dataloaders
    data_loader, valid_data_loader = instantiate(arch_cfg.data_loader)

    # setup model
    model = instantiate(arch_cfg.arch)
    logger.info(model)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')

    model = model.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = instantiate(arch_cfg.loss, is_func=True)

    optimizer = instantiate(arch_cfg.optimizer, model.parameters())

    lr_scheduler = None
    if "lr_scheduler" in arch_cfg:
        lr_scheduler = instantiate(arch_cfg.lr_scheduler, optimizer)

    # later changed if checkpoint
    start_epoch=0

    if checkpoint_dir:
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch=checkpoint["epoch"]+1

    metric_ftns = [instantiate(met, is_func=True) for met in arch_cfg.metrics]
    train_metrics = BatchMetrics('loss', *[m.__name__ for m in metric_ftns])
    valid_metrics = BatchMetrics('loss', *[m.__name__ for m in metric_ftns])

    for epoch in range(start_epoch, arch_cfg.trainer.epochs):  # loop over the dataset multiple times
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

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            filename = str(Path(checkpoint_dir) / 'model_checkpoint.pth')
            torch.save(state, filename)

        log = train_metrics.result()
        val_log = valid_metrics.result()
        log.update(**{'val_'+k : v for k, v in val_log.items()})
        tune.report(**log)