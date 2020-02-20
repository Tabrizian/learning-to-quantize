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


def cifar10_full(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_full' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch',
               'pretrained', 'niters', 'epoch_iters',
               'gvar_log_iter', 'gvar_start', 'g_bsnap_iter',
               'g_optim_start', 'nuq_truncated_interval']
    shared_args = [('dataset', dataset),
                   ('optim', ['sgd']),  # 'sgd', 'adam'
                   # ('arch', 'resnet32'),
                   ('arch', ['resnet32', 'resnet56']),
                   ('batch_size', 128),
                   ('lr', [0.1]),
                   ('momentum', 0.9),
                   ('weight_decay', 1e-4),
                   ('niters', 80000),
                   ('lr_decay_epoch', '40000,60000'),
                   ]
    gvar_args = [
                 # ('gvar_estim_iter', 10),
                 ('gvar_log_iter', 100),  # 100
                 ('gvar_start', 0),
                 ('g_osnap_iter', 100),
                 ('g_bsnap_iter', 10000),
                 ('g_optim', ''),
                 ('g_optim_start', 0),
                 ('g_epoch', ''),
                 ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_super_sgd = [
        ('g_estim', ['nuq']),
        ('nuq_bits', [4]),
        ('nuq_bucket_size', [65536]),
        ('nuq_ngpu', 4),  # 2
        ('nuq_method', [
                        ('nuq3', OrderedDict([('nuq_truncated_interval', 1), ('nuq_number_of_samples', 100), ('nuq_learning_rate', 0.7), ('nuq_layer',  1)])),
                        ('nuq5', OrderedDict([('nuq_truncated_interval', 1), ('nuq_number_of_samples', 100), ('nuq_cd_epochs', 30), ('nuq_layer',  1)])),
                        ('qinf', OrderedDict([('nuq_truncated_interval', 1), ('nuq_number_of_samples', 100), ('nuq_learning_rate', 0.7), ('nuq_layer', 1)])),
                        ('nuq', OrderedDict([('nuq_mul', 0.5),('nuq_truncated_interval', 1), ('nuq_number_of_samples', 100),('nuq_layer',  1)])),
                        ('none', OrderedDict([('nuq_truncated_interval', 1), ('nuq_number_of_samples', 100), ('nuq_learning_rate', 0.7), ('nuq_layer', 1)])),
                        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_super_sgd)]

    return args, log_dir, module_name, exclude


def imagenet_full(args):
    dataset = 'imagenet'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_full' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch',
               'pretrained', 'niters', 'epoch_iters',
               'gvar_log_iter', 'gvar_start', 'g_bsnap_iter', 
               'g_optim_start']
    shared_args = [('dataset', dataset),
                   # ('optim', 'sgd'),  # 'sgd', 'adam'
                   # ('arch', 'resnet18'),
                   # ('arch', 'resnet34'),
                   ('arch', 'resnet34'),
                   ('batch_size', 64),  # 256),
                   ('niters', 60*10000),
                   ('epoch_iters', 1000),
                   ('lr', 0.02),
                   ('lr_decay_epoch', '300000,450000'),
                   ('momentum', 0.9),
                   ('weight_decay', 1e-4),
                   # ('train_accuracy', ''),
                   ]
    gvar_args = [
                 # ('gvar_estim_iter', 10),
                 ('gvar_log_iter', 1000),  # 100
                 ('gvar_start', 0),
                 ('g_bsnap_iter', 100*10000),
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
        ('nuq_method', ['q',
                        'qinf',
                        ('nuq', OrderedDict([('nuq_mul', 0.5),
                                             ]))
                        ]),
        ('nuq_parallel', 'ngpu')
    ]
    args += [OrderedDict(shared_args+gvar_args+args_nuq)]

    args_nuq = [
        ('g_estim', ['nuq']),
        ('nuq_ngpu', 2),  # [4, 8]),  # 2
        ('nuq_method', 'none'),
        ('nuq_parallel', 'ngpu')
    ]
    args += [OrderedDict(shared_args+gvar_args+args_nuq)]

    return args, log_dir, module_name, exclude


def cifar10_gvar(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch',
               'pretrained', 'niters', 'epoch_iters',
               'gvar_log_iter', 'gvar_start', 'g_bsnap_iter', 'g_optim',
               'g_optim_start']
    shared_args = [('dataset', dataset),
                   # ('optim', 'sgd'),  # 'sgd', 'adam'
                   # ('arch', 'resnet32'),
                   ('arch', 'resnet110'),
                   ('batch_size', 128),
                   ('lr', [0.1]),
                   ('momentum', 0.9),
                   ('weight_decay', 1e-4),  # 0
                   ('niters', 80000),
                   ('lr_decay_epoch', '40000,60000'),
                   ]
    gvar_args = [
                 # ('gvar_estim_iter', 10),
                 ('gvar_log_iter', 500),  # 100
                 ('gvar_start', 0),
                 ('g_bsnap_iter', 10000),
                 ('g_osnap_iter', 10000),
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
        ('nuq_ngpu', 8),
        ('nuq_method', ['q', 'qinf',
                        ('nuq', OrderedDict([('nuq_mul', 0.5)])),
                        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_nuq)]

    args_nuq = [
        ('g_estim', ['nuq']),
        ('nuq_ngpu', 8),
        ('nuq_method', 'none')
    ]
    args += [OrderedDict(shared_args+gvar_args+args_nuq)]

    return args, log_dir, module_name, exclude
