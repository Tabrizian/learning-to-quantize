import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np
import os


def get_loaders(opt):
    if opt.dataset == 'mnist':
        return get_mnist_loaders(opt)
    elif opt.dataset == 'cifar10':
        return get_cifar10_loaders(opt)
    elif opt.dataset == 'cifar100':
        return get_cifar100_loaders(opt)
    elif opt.dataset == 'svhn':
        return get_svhn_loaders(opt)
    elif opt.dataset.startswith('imagenet'):
        return get_imagenet_loaders(opt)
    elif opt.dataset == 'logreg':
        return get_logreg_loaders(opt)
    elif 'class' in opt.dataset:
        return get_logreg_loaders(opt)


def dataset_to_loaders(train_dataset, test_dataset, opt):
    kwargs = {'num_workers': opt.workers,
              'pin_memory': True} if opt.cuda else {}
    idxdataset = IndexedDataset(train_dataset, opt, train=True)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        idxdataset,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        IndexedDataset(test_dataset, opt),
        batch_size=opt.test_batch_size, shuffle=False,
        **kwargs)

    train_test_loader = torch.utils.data.DataLoader(
        IndexedDataset(train_dataset, opt, train=True),
        batch_size=opt.test_batch_size, shuffle=False,
        **kwargs)
    return train_loader, test_loader, train_test_loader


def get_minvar_loader(train_loader, opt):
    kwargs = {'num_workers': opt.workers,
              'pin_memory': True} if opt.cuda else {}
    idxdataset = train_loader.dataset
    train_loader = torch.utils.data.DataLoader(
        idxdataset,
        batch_size=opt.g_batch_size,
        shuffle=True,
        drop_last=False, **kwargs)
    return train_loader


class IndexedDataset(data.Dataset):
    def __init__(self, dataset, opt, train=False):
        np.random.seed(2222)
        self.ds = dataset
        self.opt = opt

    def __getitem__(self, index):
        subindex = index
        img, target = self.ds[subindex]
        return img, target, index

    def __len__(self):
        return len(self.ds)


def get_mnist_loaders(opt, **kwargs):
    transform = transforms.ToTensor()
    if not opt.no_transform:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST(
        opt.data, train=True, download=True, transform=transform)

    test_dataset = datasets.MNIST(opt.data, train=False, transform=transform)
    return dataset_to_loaders(train_dataset, test_dataset, opt, **kwargs)


def get_cifar10_100_transform(opt):
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.2023, 0.1994, 0.2010))

    if opt.data_aug:
        transform = [
            transforms.RandomAffine(10, (.1, .1), (0.7, 1.2), 10),
            transforms.ColorJitter(.2, .2, .2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    return normalize, transform


def get_cifar10_loaders(opt):
    normalize, transform = get_cifar10_100_transform(opt)

    train_dataset = datasets.CIFAR10(root=opt.data, train=True,
                                     transform=transforms.Compose(transform),
                                     download=True)
    test_dataset = datasets.CIFAR10(
        root=opt.data, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    return dataset_to_loaders(train_dataset, test_dataset, opt)


def get_cifar100_loaders(opt):
    normalize, transform = get_cifar10_100_transform(opt)

    train_dataset = datasets.CIFAR100(root=opt.data, train=True,
                                      transform=transforms.Compose(transform),
                                      download=True)
    test_dataset = datasets.CIFAR100(
        root=opt.data, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    return dataset_to_loaders(train_dataset, test_dataset, opt)


def get_svhn_loaders(opt, **kwargs):
    normalize = transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))
    if opt.data_aug:
        transform = [
            transforms.RandomAffine(10, (.1, .1), (0.7, 1.), 10),
            transforms.ColorJitter(.2, .2, .2),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ]

    train_dataset = torch.utils.data.ConcatDataset(
        (datasets.SVHN(
            opt.data, split='train', download=True,
            transform=transforms.Compose(transform)),
         datasets.SVHN(
             opt.data, split='extra', download=True,
             transform=transforms.Compose(transform))))
    test_dataset = datasets.SVHN(opt.data, split='test', download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5),
                                                          (0.5, 0.5, 0.5))
                                 ]))
    return dataset_to_loaders(train_dataset, test_dataset, opt)


def get_imagenet_loaders(opt):
    # Data loading code
    traindir = os.path.join(opt.data, 'train')
    valdir = os.path.join(opt.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    test_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    return dataset_to_loaders(train_dataset, test_dataset, opt)


class InfiniteLoader(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.data_iter = iter([])
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            if isinstance(self.data_loader, list):
                II = self.data_loader
                self.data_iter = (II[i] for i in torch.randperm(len(II)))
            else:
                self.data_iter = iter(self.data_loader)
            data = next(self.data_iter)
        return data

    def next(self):
        # for python2
        return self.__next__()

    def __len__(self):
        return len(self.data_loader)


def random_orthogonal_matrix(gain, shape):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are "
                           "supported.")

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return np.asarray(gain * q, dtype=np.float)


class LinearDataset(data.Dataset):

    def __init__(self, C, D, num, dim, num_class, train=True):
        X = np.zeros((C.shape[0], num))
        Y = np.zeros((num,))
        for i in range(num_class):
            n = num // num_class
            e = np.random.normal(0.0, 1.0, (dim, n))
            X[:, i * n:(i + 1) * n] = np.dot(D[:, :, i], e) + C[:, i:i + 1]
            Y[i * n:(i + 1) * n] = i
        self.X = X
        self.Y = Y
        self.classes = range(num_class)

    def __getitem__(self, index):
        X = torch.Tensor(self.X[:, index]).float()
        Y = int(self.Y[index])
        return X, Y

    def __len__(self):
        return self.X.shape[1]


def get_logreg_loaders(opt, **kwargs):
    # np.random.seed(1234)
    np.random.seed(2222)
    # print("Create W")
    C = opt.c_const * random_orthogonal_matrix(1.0, (opt.dim, opt.num_class))
    D = opt.d_const * random_orthogonal_matrix(
        1.0, (opt.dim, opt.dim, opt.num_class))
    # print("Create train")
    train_dataset = LinearDataset(C, D, opt.num_train_data, opt.dim,
                                  opt.num_class, train=True)
    # print("Create test")
    test_dataset = LinearDataset(C, D,
                                 opt.num_test_data, opt.dim, opt.num_class,
                                 train=False)
    torch.save((train_dataset.X, train_dataset.Y,
                test_dataset.X, test_dataset.Y,
                C), opt.logger_name + '/data.pth.tar')

    return dataset_to_loaders(train_dataset, test_dataset, opt)
