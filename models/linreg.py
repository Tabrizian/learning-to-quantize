import torch.nn as nn
# import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, dim, num_class):
        super(Linear, self).__init__()
        self.linear = nn.Linear(dim, num_class)

    def forward(self, x):
        x = self.linear(x)
        return x


class TwoLinear(nn.Module):
    def __init__(self, dim, num_class):
        super(TwoLinear, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, num_class)

    def forward(self, x):
        # x = F.relu(self.linear1(x))
        x = self.linear1(x)
        x = self.linear2(x)
        return x
