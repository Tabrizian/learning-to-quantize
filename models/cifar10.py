# https://github.com/akamaster/pytorch_resnet_cifar10
'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ['ResNet', 'resnet8', 'resnet20', 'resnet32',
           'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        if hasattr(init, 'kaiming_normal_'):
            init.kaiming_normal_(m.weight)
        else:
            init.kaiming_normal(m.weight)


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A',
                 nobatchnorm=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.nobatchnorm = nobatchnorm

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, planes // 4, planes // 4),
                                    "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        if self.nobatchnorm:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_class=10, nobatchnorm=False):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.nobatchnorm = nobatchnorm

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_class)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                nobatchnorm=self.nobatchnorm))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.nobatchnorm:
            out = F.relu(self.conv1(x))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # return F.log_softmax(out, dim=-1)
        return out


def resnet8(num_class=10, nobatchnorm=False):
    return ResNet(BasicBlock, [1, 1, 1], num_class=num_class,
                  nobatchnorm=nobatchnorm)


def resnet20(num_class=10):
    return ResNet(BasicBlock, [3, 3, 3], num_class=num_class)


def resnet32(num_class=10):
    return ResNet(BasicBlock, [5, 5, 5], num_class=num_class)


def resnet44(num_class=10):
    return ResNet(BasicBlock, [7, 7, 7], num_class=num_class)


def resnet56(num_class=10):
    return ResNet(BasicBlock, [9, 9, 9], num_class=num_class)


def resnet110(num_class=10):
    return ResNet(BasicBlock, [18, 18, 18], num_class=num_class)


def resnet1202(num_class=10):
    return ResNet(BasicBlock, [200, 200, 200], num_class=num_class)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(
        lambda p: p.requires_grad and len(p.data.size()) > 1,
        net.parameters()))))


class Convnet(nn.Module):
    def __init__(self, dropout=True, num_class=10):
        """
        2conv + 2fc + dropout, from adam's paper
        similar to mnist's convnet
        100 epochs lr update at 50
        """
        super(Convnet, self).__init__()
        self.dropout = dropout
        # self.input_drop = nn.Dropout2d(p=0.2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(128*5*5, 1000)
        self.fc2 = nn.Linear(1000, num_class)

    def forward(self, x):
        if self.dropout:
            x = F.dropout2d(x, training=self.training, p=0.2)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.relu(F.max_pool2d(self.conv3(x), 3))
        x = x.view(-1, 128*5*5)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=-1)
        return x


class MLP(nn.Module):
    def __init__(self, dropout=True, num_class=10):
        """
        mnist MLP
        """
        super(MLP, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(3*32*32, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        # self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, num_class)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
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


class SmallMLP(nn.Module):
    def __init__(self, dropout=True, num_class=10):
        """
        mnist MLP
        """
        super(SmallMLP, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(3*32*32, 512)
        self.fc2 = nn.Linear(512, num_class)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        if self.dropout:
            x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return F.log_softmax(x, dim=-1)
        return x


class MoreSmallMLP(nn.Module):
    def __init__(self, dropout=True, num_class=10):
        """
        mnist MLP
        """
        super(MoreSmallMLP, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(3*32*32, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        if self.dropout:
            x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return F.log_softmax(x, dim=-1)
        return x


class SuperSmallMLP(nn.Module):
    def __init__(self, dropout=True, num_class=10):
        """
        mnist MLP
        """
        super(SuperSmallMLP, self).__init__()
        self.fc1 = nn.Linear(3*32*32, num_class)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = self.fc1(x)
        # return F.log_softmax(x, dim=-1)
        return x


class SmallCNN(nn.Module):
    def __init__(self, dropout=False, num_class=10):
        """
        async kfac's cnn
        """
        super(SmallCNN, self).__init__()
        self.dropout = dropout
        # self.input_drop = nn.Dropout2d(p=0.2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64*4*4, num_class)  # 1000)
        # self.fc2 = nn.Linear(1000, num_class)

    def forward(self, x):
        if self.dropout:
            x = F.dropout2d(x, training=self.training, p=0.2)
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=2, padding=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=2, padding=1))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, stride=2, padding=1))
        x = x.view(-1, 64*4*4)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        # if self.dropout:
        #     x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=-1)
        return x


class SuperSmallCNN(nn.Module):
    def __init__(self, dropout=False, num_class=10):
        """
        smaller than small cnn
        """
        super(SuperSmallCNN, self).__init__()
        self.dropout = dropout
        # self.input_drop = nn.Dropout2d(p=0.2)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(16*4*4, num_class)  # 1000)
        # self.fc2 = nn.Linear(1000, num_class)

    def forward(self, x):
        if self.dropout:
            x = F.dropout2d(x, training=self.training, p=0.2)
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=2, padding=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=2, padding=1))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, stride=2, padding=1))
        x = x.view(-1, 16*4*4)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        # if self.dropout:
        #     x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=-1)
        return x


class LP(nn.Module):
    def __init__(self, dropout=False, num_class=10):
        """
        mnist MLP
        """
        super(LP, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(3*32*32, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = F.relu(self.fc1(x))
        # return F.log_softmax(x, dim=-1)
        return x


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
