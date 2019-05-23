from collections import OrderedDict


def mnist(args):
    dataset = 'mnist'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_nuq' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch']
    shared_args = [('dataset', dataset),
                   ('lr', .1),  # [.1, .05, .01]),
                   ('weight_decay', 0),
                   ('momentum', 0),  # [0, 0.9]),
                   ('epochs', [
                       (30, OrderedDict([('lr_decay_epoch', 30)])),
                   ]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    gvar_args = [
                 # ('gvar_estim_iter', 10),
                 ('gvar_log_iter', 1000),  # 100
                 ('gvar_start', 0),
                 ('g_optim', ''),
                 ('g_optim_start', 0),
                 # ('g_epoch', ''),
                 ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_nuq = [
        ('g_estim', ['nuq']),
        ('nuq_bits', 4),
        ('nuq_bucket_size', [1024, 4096, 8192]),  # 8192),
        ('nuq_ngpu', [2, 4]),
        ('nuq_method', ['q', 'qinf',
                        ('nuq', OrderedDict([('nuq_mul', 0.5)]))
                        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_nuq)]

    return args, log_dir, module_name, exclude


def imagenet_last_epoch(args):
    dataset = 'imagenet'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_last_epoch' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch',
               'pretrained', 'niters', 'epoch_iters',
               'gvar_log_iter', 'gvar_start', 'g_optim', 'g_optim_start']
    shared_args = [('dataset', dataset),
                   # ('optim', 'sgd'),  # 'sgd', 'adam'
                   # ('arch', 'resnet18'),
                   # ('arch', 'resnet34'),
                   ('arch', 'resnet34'),
                   ('batch_size', 128),  # 256),
                   # ('test_batch_size', 64),
                   #  ### pretrained
                   ('pretrained', ['']),
                   # ('epochs', [1]),
                   ('niters', 10000),
                   ('epoch_iters', 500),
                   ('lr', [.001]),
                   ('lr_decay_epoch', 10000),
                   # ('exp_lr', [None]),
                   ('momentum', 0),
                   ('weight_decay', 0),
                   ('untrain_steps', 5),  # 10),
                   ('untrain_lr', 0.001),
                   ('untrain_std', 5e-3),
                   ]
    gvar_args = [
                 # ('gvar_estim_iter', 10),
                 ('gvar_log_iter', 1000),  # 100
                 ('gvar_start', 0),
                 ('g_optim', ''),
                 ('g_optim_start', 0),
                 # ('g_epoch', ''),
                 ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_nuq = [
        ('g_estim', ['nuq']),
        ('nuq_bits', 4),
        ('nuq_bucket_size', 8192),
        ('nuq_ngpu', 2),  # 4
        ('nuq_method', ['q', 'qinf',
                        ('nuq', OrderedDict([('nuq_mul', 0.5)]))
                        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_nuq)]

    return args, log_dir, module_name, exclude


def imagenet_first_epoch(args):
    dataset = 'imagenet'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_first_epoch' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch',
               'pretrained', 'niters', 'epoch_iters',
               'gvar_log_iter', 'gvar_start', 'g_optim',
               'g_optim_start']
    shared_args = [('dataset', dataset),
                   # ('optim', 'sgd'),  # 'sgd', 'adam'
                   # ('arch', 'resnet18'),
                   # ('arch', 'resnet34'),
                   ('arch', 'resnet34'),
                   ('batch_size', 128),  # 256),
                   # ('test_batch_size', 64),
                   ('niters', 10000),
                   ('epoch_iters', 500),
                   ('lr', [0.1]),
                   ('lr_decay_epoch', 10000),
                   ('momentum', 0),
                   ('weight_decay', 0),
                   ]
    gvar_args = [
                 # ('gvar_estim_iter', 10),
                 ('gvar_log_iter', 500),  # 100
                 ('gvar_start', 0),
                 ('g_optim', ''),
                 ('g_optim_start', 0),
                 # ('g_epoch', ''),
                 ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_nuq = [
        ('g_estim', ['nuq']),
        ('nuq_bits', 4),
        ('nuq_bucket_size', 8192),
        ('nuq_ngpu', 2),  # 4
        ('nuq_method', ['q', 'qinf',
                        ('nuq', OrderedDict([('nuq_mul', 0.5)]))
                        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_nuq)]

    return args, log_dir, module_name, exclude


def cifar10_first_epoch(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_first_epoch' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch',
               'pretrained', 'niters', 'epoch_iters',
               'gvar_log_iter', 'gvar_start', 'g_optim',
               'g_optim_start']
    shared_args = [('dataset', dataset),
                   # ('optim', 'sgd'),  # 'sgd', 'adam'
                   ('arch', 'resnet110'),
                   ('batch_size', 128),
                   # ('test_batch_size', 64),
                   ('niters', 10000),
                   ('epoch_iters', 500),
                   ('lr_decay_epoch', 10000),
                   ('lr', [0.1]),
                   ('momentum', 0),
                   ('weight_decay', 0),
                   ]
    gvar_args = [
                 # ('gvar_estim_iter', 10),
                 ('gvar_log_iter', 500),  # 100
                 ('gvar_start', 0),
                 ('g_optim', ''),
                 ('g_optim_start', 0),
                 # ('g_epoch', ''),
                 ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_nuq = [
        ('g_estim', ['nuq']),
        ('nuq_bits', 4),
        ('nuq_bucket_size', 8192),
        ('nuq_ngpu', 2),  # 4
        ('nuq_method', ['q', 'qinf',
                        ('nuq', OrderedDict([('nuq_mul', 0.5)]))
                        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_nuq)]

    return args, log_dir, module_name, exclude
