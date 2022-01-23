import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Module, Softmax, BatchNorm1d, Dropout


class Adjacency(nn.Module):
    def __init__(self):
        super(Adjacency, self).__init__()
        self.fc = nn.Linear(400, 400)

    def forward(self, x, identity):

        x = torch.squeeze(x)
        # print('x: ', x)

        mask = self.fc(identity)
        mask = torch.sigmoid(mask)
        mask = torch.triu(mask)
        mask += mask.T - torch.diag(torch.diag(mask, 0), 0)

        # print('weight: ', weight)
        adjacency = torch.round(mask)
        # print('adjacency: ', adjacency)

        output = x.mul(adjacency)
        # print('output: ', output)

        output = torch.unsqueeze(output, 1)

        return output, adjacency

