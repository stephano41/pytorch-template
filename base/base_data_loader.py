import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import torch


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


class WeightedDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, targets,
                 collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, train_idx, valid_idx = self._split_sampler(self.validation_split, targets)

        train_set = Subset(dataset, train_idx)
        self.valid_set = Subset(dataset, valid_idx)

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, pin_memory=True, dataset=train_set, **self.init_kwargs)

    # make it not random
    def _split_sampler(self, split, targets):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = self._weighted_sampler(targets, train_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, train_idx, valid_idx

    @staticmethod
    def _weighted_sampler(targets, subset_idx):
        class_sample_count = torch.tensor(
            [(targets[subset_idx] == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in targets[subset_idx]])

        return WeightedRandomSampler(samples_weight, len(samples_weight))

    def split_validation(self):
        return DataLoader(pin_memory=True, dataset=self.valid_set, **self.init_kwargs)
