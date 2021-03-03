from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from contextlib import contextmanager


@contextmanager
def timer():
    st_time = time.time()
    yield
    print("Total time %.3f seconds" % (time.time() - st_time))


class StaticMnistNet(nn.Module):
    def __init__(self):
        super(StaticMnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.logsoftmax = nn.functional.log_softmax
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2d(x, )
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dequant(x)
        output = self.logsoftmax(x, dim=1)
        return output
