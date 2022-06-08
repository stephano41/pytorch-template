from pathlib import Path

import torch
from omegaconf import OmegaConf
from ray import tune

from srcs.logger import BatchMetrics
from srcs.utils import instantiate, prepare_devices
from srcs.utils.files import write_conf


def train_func(config, arch_cfg, checkpoint_dir=None):
    # os.chdir(get_original_cwd())
    # cwd is changed to the trial folder

    config = OmegaConf.create(config)
    arch_cfg = OmegaConf.merge(arch_cfg, config)
    write_conf(arch_cfg, "config.yaml")

    device, device_ids = prepare_devices(arch_cfg.n_gpu)

    # setup dataloaders
    data_loader, valid_data_loader = instantiate(arch_cfg.data_loader)

    # logger=logging.getLogger('tune')
    # setup model
    model = instantiate(arch_cfg.arch)

    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')

    model = model.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = instantiate(arch_cfg.loss, is_func=True)

    optimizer = instantiate(arch_cfg.optimizer, model.parameters())

    lr_scheduler = None
    if "lr_scheduler" in arch_cfg:
        lr_scheduler = instantiate(arch_cfg.lr_scheduler, optimizer)

    # later changed if checkpoint
    start_epoch = 0

    if checkpoint_dir:
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint["epoch"] + 1

    metric_ftns = [instantiate(met, is_func=True) for met in arch_cfg.metrics]
    train_metrics = BatchMetrics('loss', *[m.__name__ for m in metric_ftns])
    valid_metrics = BatchMetrics('loss', *[m.__name__ for m in metric_ftns])

    for epoch in range(start_epoch, arch_cfg.trainer.epochs):  # loop over the dataset multiple times
        train_metrics.reset()
        train_log = one_epoch(data_loader, criterion, model, device, metric_ftns, train_metrics, optimizer)

        valid_metrics.reset()
        with torch.no_grad():
            val_log = one_epoch(valid_data_loader, criterion, model, device, metric_ftns, valid_metrics)

        if lr_scheduler is not None:
            lr_scheduler.step()

        state = {
            'arch': type(model).__name__,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": arch_cfg
        }

        # create checkpoint
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            filename = str(Path(checkpoint_dir) / 'model_checkpoint.pth')
            torch.save(state, filename)

            # log metrics, log in checkpoint in case actor dies half way
            train_log.update(**{'val_' + k: v for k, v in val_log.items()})
            tune.report(**train_log)


def one_epoch(data_loader, criterion, model, device, metric_ftns, metric_tracker: BatchMetrics,
              optimizer=None) -> dict:
    for i, data in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        if optimizer is not None:
            # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if optimizer is not None:
            loss.backward()
            optimizer.step()
        metric_tracker.update('loss', loss.item())
        for met in metric_ftns:
            metric_tracker.update(met.__name__, met(outputs, targets))

    return metric_tracker.result()
