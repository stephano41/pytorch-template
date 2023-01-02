from __future__ import annotations

import inspect
import os
import tempfile
from typing import Any, List, Dict

import numpy as np
import torch
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch import nn as nn
from torch.utils.data import TensorDataset, SubsetRandomSampler, DataLoader, WeightedRandomSampler

from ._stopper import EarlyStopper
from ..loggers.result_trackers import MetricCollection, FitResults
from ..preprocessors.transforms import TorchStandardScaler
from ..utils import prepare_devices, update_nested_dict


class NeuralNet:
    __slots__ = ["model", "loss_fn", "optimizer", "data_dims", "metric_names", "n_gpu", "num_workers", "epochs",
                 "batch_size", "standardise", "weighted", "n_iter_no_change", "stop_grace_period",
                 "validation_fraction", "monitor", "mode", "_device", "_device_ids", "_mode_modifier", "_scaler",
                 "_optimizer", "_model", "_loss_fn"]

    def __init__(self, model: Dict,
                 loss_fn: Dict,
                 optimizer: Dict,
                 data_dims: Any,
                 metric_names: List[str] | str = 'accuracy',
                 n_gpu: int = 0,
                 num_workers: int = 0,
                 epochs: int = 100,
                 batch_size: int = 128,
                 standardise: bool = False,
                 weighted: bool = False,
                 n_iter_no_change: int = 10,
                 stop_grace_period: int = 10,
                 validation_fraction: float = 0.1,
                 monitor: str = 'val/accuracy',
                 mode: str = 'max'):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.data_dims = data_dims
        self.metric_names = metric_names
        self.n_gpu = n_gpu
        self.num_workers = num_workers
        self.epochs = epochs
        self.batch_size = batch_size
        self.standardise = standardise
        self.weighted = weighted
        self.n_iter_no_change = n_iter_no_change
        self.stop_grace_period = stop_grace_period
        self.validation_fraction = validation_fraction
        self.monitor = monitor
        self.mode = mode

        self._device, self._device_ids = prepare_devices(self.n_gpu)
        self._mode_modifier = 1. if self.mode == 'max' else -1.
        self._scaler, self._optimizer, self._model = None, None, None
        self._loss_fn = instantiate(self.loss_fn)

    def __init_modules(self, X, Y):
        size = self.data_dims(X, Y)
        self._scaler = TorchStandardScaler(dim=size.std_dims)
        self._model: nn.Module = instantiate(self.model, sizes=size)
        self._optimizer = instantiate(self.optimizer, self._model.parameters())

    def __str__(self):
        model_params = ", ".join(f"{k}={v}" for k, v in self.model.items() if k != "_target_")
        if self._model is None:
            return f"{get_class(self.model['_target_']).__name__}({model_params})"
        return f"{self._model._get_name()}({model_params})"

    def __repr__(self):
        if self._model is None or self._optimizer is None:
            return str(self) + ", ".join(f"{k}={v}" for k, v in self.optimizer.items() if k != "_target_")
        return repr(self._model) + '\n' + repr(self._optimizer)

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        self.__init_modules(X, Y)
        train_metrics = MetricCollection(self.metric_names, prefix="train/", include_loss=True)
        val_metrics = MetricCollection(self.metric_names, prefix="val/", include_loss=True)
        fit_results = FitResults(list(train_metrics.metric_tracker.keys) + list(val_metrics.metric_tracker.keys))

        train_idx, val_idx = train_test_split(np.arange((len(X))), test_size=self.validation_fraction)

        if self.standardise:
            self._scaler.fit(X[train_idx])
            dataset = TensorDataset(self._scaler.transform(X), Y)
        else:
            dataset = TensorDataset(X, Y)

        val_sampler = SubsetRandomSampler(val_idx)
        val_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=val_sampler, num_workers=self.num_workers)
        if self.weighted:
            train_loader = get_weighted_dl(train_idx, dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        else:
            train_sampler = SubsetRandomSampler(train_idx)
            train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler,
                                      num_workers=self.num_workers)

        stopper = EarlyStopper(self.mode, n_iter_no_change=self.n_iter_no_change, grace_period=self.stop_grace_period)

        self._model = self._model.to(self._device)

        if len(self._device_ids) > 1:
            self._model = nn.DataParallel(self._model, device_ids=self._device_ids)

        best_score = -1
        # create temp directory to save best state
        with tempfile.TemporaryDirectory() as tmp:
            temp_file_name = os.path.join(tmp, 'best_state.pth')
            for epoch in range(self.epochs):
                train_metrics.reset()
                train_log = one_epoch(self._model, self._device, train_loader, train_metrics, loss_fn=self._loss_fn,
                                      optimizer=self._optimizer)

                val_metrics.reset()
                with torch.no_grad():
                    val_log = one_epoch(self._model, self._device, val_loader, val_metrics, loss_fn=self._loss_fn)

                fit_results.update(**train_log, **val_log)

                monitor_score = val_log[self.monitor]
                if monitor_score * self._mode_modifier > best_score * self._mode_modifier:
                    best_score = monitor_score
                    torch.save({"model": self._model.state_dict(), "optimizer": self._optimizer.state_dict()},
                               temp_file_name)

                if stopper(monitor_score):
                    break
            best_states = torch.load(temp_file_name)
            self._model.load_state_dict(best_states["model"])
            self._optimizer.load_state_dict(best_states["optimizer"])
        return fit_results

    def score(self, X, Y, prefix="test/"):
        test_metrics = MetricCollection(self.metric_names, prefix=prefix, include_loss=True)
        outputs = self.predict_proba(X)

        loss = self._loss_fn(outputs, Y)
        test_metrics.update(Y, torch.argmax(outputs, dim=1), loss=loss)
        return test_metrics.result()

    def predict_proba(self, X):
        """
        returns logits
        """
        self._model.eval()
        if self.standardise:
            X = self._scaler.transform(X)
        if len(X) > 2 * self.batch_size:
            test_dataset = TensorDataset(X)
            dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
            self._model.to(self._device)
            outputs = []
            for inputs in dataloader:
                inputs = inputs[0].to(self._device)
                outputs.append(self._model(inputs))

            outputs = torch.row_stack(outputs)

        else:
            self._model.to(torch.device('cpu'))
            outputs = self._model(X)

        return outputs.detach().cpu()

    def predict(self, X):
        """
        returns argmaxed
        """
        return torch.argmax(self.predict_proba(X), dim=1)

    def get_params(self, deep=None):
        return {key: getattr(self, key) for key in inspect.signature(NeuralNet.__init__).parameters.keys() if
                key != "self"}

    def set_params(self, **params):
        for key, value in params.items():
            if not hasattr(self, key):
                raise AttributeError("Unknown param in set_params")
            if isinstance(value, (Dict, DictConfig)) and isinstance(getattr(self, key), (Dict, DictConfig)):
                setattr(self, key, update_nested_dict(getattr(self, key), value))
            else:
                setattr(self, key, value)

        return self


def get_weighted_dl(train_idx, dataset: TensorDataset, batch_size=128, num_workers=2):
    target = dataset.tensors[1]
    class_sample_count = torch.tensor([(target == t).sum() for t in torch.unique(target, sorted=True)])
    class_weights = 1 / class_sample_count.float()
    sample_weights = torch.tensor([class_weights[t] for t in target])
    final_weights = torch.zeros(len(target))
    final_weights[train_idx] = sample_weights[train_idx]
    train_sampler = WeightedRandomSampler(final_weights, len(final_weights))

    return DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)


def one_epoch(model, device, dataloader, metric_tracker: MetricCollection, loss_fn, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        if optimizer is not None:
            # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        if optimizer is not None:
            loss.backward()
            optimizer.step()
        metric_tracker.update(targets, torch.argmax(outputs, dim=1), loss)

    return metric_tracker.result()


class TimeSeriesNN(NeuralNet):
    def __init__(self, *args, **kwargs):
        kwargs["data_dims"] = TimeSeriesDD
        super().__init__(*args, **kwargs)

    def get_params(self, deep=None):
        params = super().get_params()
        params.pop("data_dims")
        return params


class TimeSeriesDD:
    __slots__ = ["interp_len", "channel_n", "output_size", "std_dims"]

    def __init__(self, X, Y):
        x_shape = X.shape
        self.interp_len = x_shape[1]
        self.channel_n = x_shape[2]
        self.output_size = len(torch.unique(Y))
        self.std_dims = (0, 1)


class ImageNN(NeuralNet):
    def __init__(self, *args, **kwargs):
        kwargs["data_dims"] = ImageDD
        super().__init__(*args, **kwargs)

    def get_params(self, deep=None):
        params = super().get_params()
        params.pop("data_dims")
        return params


class ImageDD:
    __slots__ = ["channel_n", "scale", "interp_len", "output_size", "std_dims"]

    def __init__(self, X, Y):
        x_shape = X.shape
        self.channel_n = x_shape[1]
        self.scale = x_shape[2]
        self.interp_len = x_shape[3]
        self.output_size = len(torch.unique(Y))
        self.std_dims = (0, 2, 3)
