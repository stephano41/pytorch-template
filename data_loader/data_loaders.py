from __future__ import unicode_literals, print_function, division

from base import BaseDataLoader
from utils.NLP import *

import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

from transformers import BertTokenizer


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class W_ClassDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=2, training=True):
        self.data_dir = data_dir

        all_categories = []
        data_x = []
        data_y = []

        for filename in findFiles(self.data_dir+'*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            all_categories.append(category)
            lines = readLines(filename)
            for line in lines:
                line = lineToTensor2(line)
                data_x.append(line)
                data_y.append(all_categories.index(category))

        self.n_categories = len(all_categories)
        data_x = pad_sequence(data_x, batch_first=True)
        data_y = torch.tensor(data_y)

        self.dataset = TensorDataset(data_x, data_y)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class BertWDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=2, training=True):
        self.data_dir = data_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        input_ids=[]
        labels = []
        all_categories = []

        for filename in findFiles(self.data_dir+'*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            all_categories.append(category)
            lines = readLines(filename)
            for line in lines:
                encoded_name = self.tokenizer.encode(line)
                input_ids.append(torch.tensor(encoded_name))
                labels.append(all_categories.index(category))

        input_ids = pad_sequence(input_ids, batch_first=True)

        attention_masks = self._createAttentionMask(input_ids)

        labels = torch.tensor(labels)

        self.dataset = TensorDataset(input_ids, attention_masks, labels)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    @staticmethod
    def _createAttentionMask(input_ids):
        attention_masks = []

        for sent in input_ids:
            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)
        return torch.tensor(attention_masks)