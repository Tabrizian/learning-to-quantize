import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    def __init__(self, dropout=True):
        """30 epochs no lr update
        """
        super(MNISTNet, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        if self.dropout:
            x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=-1)
        return x


class Convnet(nn.Module):
    def __init__(self, dropout=True):
        """
        2conv + 2fc + dropout, something to get ~.5% error.
        something close to what maxout paper uses?
        30 epochs no lr update
        """
        super(Convnet, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*4*4, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        if self.dropout:
            x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 128*4*4)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=-1)
        return x


class BigConvnet(nn.Module):
    def __init__(self, dropout=True):
        """
        Bigger than Convnet, 1000 hidden dims
        """
        super(BigConvnet, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(1, 1000, kernel_size=5)
        self.conv2 = nn.Conv2d(1000, 1000, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1000*4*4, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        if self.dropout:
            x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 1000*4*4)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=-1)
        return x


class MLP(nn.Module):
    def __init__(self, dropout=True):
        """
        Dropout paper, table 2, row 4, 1.25% error.
        http://www.cs.toronto.edu/~nitish/dropout/mnist.pbtxt
        50 epochs, lr update 30
        """
        super(MLP, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        # self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        if self.dropout:
            x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        # if self.dropout:
        #     x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc3(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        return F.log_softmax(x, dim=-1)


class SmallMLP(nn.Module):
    def __init__(self, dropout=True):
        """
        Like MLP but smaller hidden dims
        """
        super(SmallMLP, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(28*28, 50)
        self.fc2 = nn.Linear(50, 50)
        # self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        if self.dropout:
            x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        # if self.dropout:
        #     x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc3(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        # return F.log_softmax(x, dim=-1)
        return x


class SuperSmallMLP(nn.Module):
    def __init__(self, dropout=True):
        """
        Like MLP but smaller hidden dims
        """
        super(SuperSmallMLP, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(28*28, 20)
        self.fc2 = nn.Linear(20, 20)
        # self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        if self.dropout:
            x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        # if self.dropout:
        #     x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc3(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        # return F.log_softmax(x, dim=-1)
        return x
