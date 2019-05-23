import argparse
import yaml
import os

import torch
import utils


def add_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch NUQSGD')

    # options overwritting yaml options
    parser.add_argument('--path_opt', default='default.yaml',
                        type=str, help='path to a yaml options file')
    parser.add_argument('--data', default=argparse.SUPPRESS,
                        type=str, help='path to data')
    parser.add_argument('--logger_name', default='runs/runX')
    parser.add_argument('--dataset', default='mnist', help='mnist|cifar10')

    # options that can be changed from default
    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size',
                        type=int, default=argparse.SUPPRESS, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=argparse.SUPPRESS,
                        metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=argparse.SUPPRESS,
                        metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true',
                        default=argparse.SUPPRESS,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=argparse.SUPPRESS,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='how many batches to wait before logging training'
                        ' status')
    parser.add_argument('--tblog_interval',
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('--optim', default=argparse.SUPPRESS, help='sgd|dmom')
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default=argparse.SUPPRESS,
                        help='model architecture: (default: resnet32)')
    parser.add_argument('-j', '--workers', default=argparse.SUPPRESS,
                        type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--weight_decay', '--wd', default=argparse.SUPPRESS,
                        type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--train_accuracy', action='store_true',
                        default=argparse.SUPPRESS)
    parser.add_argument('--log_profiler', action='store_true')
    parser.add_argument('--lr_decay_epoch',
                        default=argparse.SUPPRESS)
    parser.add_argument('--log_keys', default='')
    parser.add_argument('--exp_lr',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--nodropout',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--data_aug',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--num_class',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--lr_decay_rate',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--nesterov',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--run_dir', default='runs/runX')
    parser.add_argument('--ckpt_name', default='model_best.pth.tar')
    parser.add_argument('--g_estim', default=argparse.SUPPRESS, type=str)
    parser.add_argument('--epoch_iters',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--gvar_log_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--gvar_estim_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--gvar_start',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_optim',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--g_optim_start',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_epoch',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--niters',
                        default=argparse.SUPPRESS, type=int)
    # NQU
    parser.add_argument('--nuq_method', default='q', help='q|nuq|qinf')
    parser.add_argument('--nuq_bits', default=4, type=int)
    parser.add_argument('--nuq_bucket_size', default=1024, type=int)
    parser.add_argument('--nuq_ngpu', default=1, type=int)
    parser.add_argument('--nuq_mul', default=0.5, type=float)
    parser.add_argument('--untrain_steps', default=0, type=int)
    parser.add_argument('--untrain_lr', default=0.001, type=float)
    parser.add_argument('--untrain_std', default=0.001, type=float)
    args = parser.parse_args()
    return args


def opt_to_nuq_kwargs(opt):
    return {'ngpu': opt.nuq_ngpu, 'bits': opt.nuq_bits,
            'bucket_size': opt.nuq_bucket_size, 'method': opt.nuq_method,
            'multiplier': opt.nuq_mul}


def yaml_opt(yaml_path):
    opt = {}
    with open(yaml_path, 'r') as handle:
        opt = yaml.load(handle, Loader=yaml.FullLoader)
    return opt


def get_opt():
    args = add_args()
    opt = yaml_opt('options/default.yaml')
    opt_s = yaml_opt(os.path.join('options/{}/{}'.format(args.dataset,
                                                         args.path_opt)))
    opt.update(opt_s)
    opt.update(vars(args).items())
    opt = utils.DictWrapper(opt)

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()

    if opt.g_batch_size == -1:
        opt.g_batch_size = opt.batch_size
    return opt
