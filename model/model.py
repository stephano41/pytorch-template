import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import utils.NLP as NLP


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class RNN(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size, input_size=NLP.n_letters()):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, self.num_layers, batch_first=True)
        # x needs to be size (batch_size, sequence size, input_size)

        self.fc = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out, _ = self.rnn(x)

        out = out[:, -1, :]

        out = self.fc(out)

        out = self.softmax(out)
        return out


class LSTM(nn.Module):
    def __init__(self,  hidden_size, num_layers, output_size, input_size=NLP.n_letters()):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.num_layers = int(num_layers)
        self.output_size = int(output_size)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        out = self.softmax(out)
        return out

    # TODO transformer model