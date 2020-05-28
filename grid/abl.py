from collections import OrderedDict


def bucket_size(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_full' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch',
               'pretrained', 'niters', 'epoch_iters',
               'gvar_log_iter', 'gvar_start', 'g_bsnap_iter',
               'g_optim_start', 'nuq_truncated_interval', 'train_accuracy',
               'nuq_number_of_samples', 'chkpt_iter', 'g_osnap_iter']
    shared_args = [('dataset', dataset),
                   ('optim', ['sgd']),  # 'sgd', 'adam'
                   # ('arch', 'resnet32'),
                   ('arch', ['resnet8']),
                   ('batch_size', 128),
                   ('lr', [0.1]),
                   ('chkpt_iter', 2000),
                   ('momentum', 0.9),
                   ('weight_decay', 1e-4),
                   ('niters', 80000),
                   ('lr_decay_epoch', '40000,60000'),
                   ('train_accuracy', ''),
                   ]
    gvar_args = [
        # ('gvar_estim_iter', 10),
        ('gvar_log_iter', 100),  # 100
        ('gvar_start', 0),
        ('g_osnap_iter', '100,2000,10000'),
        ('g_bsnap_iter', 10000),
        # ('g_optim', ''),
        # ('g_optim_start', 0),
        # ('g_epoch', ''),
    ]

    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_nuq_sgd = [
        ('g_estim', ['nuq']),
        ('nuq_bits', [3]),
        ('nuq_bucket_size', [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 8192*2, 8192*4]),
        ('nuq_ngpu', 4),  # 2
        ('dist_num', [350]),
        ('nuq_layer', ''),
        ('nuq_ig_sm_bkts', ''),
        ('nuq_truncated_interval', 1),
        ('nuq_number_of_samples', 10),
        ('nuq_method', [
            ('amq', OrderedDict([('nuq_amq_lr', 0.7), ('nuq_amq_epochs', 40)])),
            ('amq_nb', OrderedDict([('nuq_amq_lr', 0.7), ('nuq_amq_epochs', 40)])),
            ('alq', OrderedDict([('nuq_cd_epochs', 30)])),
            'qinf',
            'trn',
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30), ('nuq_sym', ''), ('nuq_inv', '')])),
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30), ('nuq_inv', '')])),
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30), ('nuq_sym', '')])),
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30)])),
            ('nuq', OrderedDict([('nuq_mul', 0.5)])),
        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_nuq_sgd)]
    args_super_sgd = [
        ('g_estim', ['nuq']),
        ('nuq_ngpu', 4),  # 2
        ('nuq_truncated_interval', 1),
        ('nuq_number_of_samples', 10),
        ('nuq_method', [
            'none'
        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_super_sgd)]

    return args, log_dir, module_name, exclude

def bits(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_full' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch',
               'pretrained', 'niters', 'epoch_iters',
               'gvar_log_iter', 'gvar_start', 'g_bsnap_iter',
               'g_optim_start', 'nuq_truncated_interval', 'train_accuracy',
               'nuq_number_of_samples', 'chkpt_iter', 'g_osnap_iter']
    shared_args = [('dataset', dataset),
                   ('optim', ['sgd']),  # 'sgd', 'adam'
                   # ('arch', 'resnet32'),
                   ('arch', ['resnet8']),
                   ('batch_size', 128),
                   ('lr', [0.1]),
                   ('chkpt_iter', 2000),
                   ('momentum', 0.9),
                   ('weight_decay', 1e-4),
                   ('niters', 80000),
                   ('lr_decay_epoch', '40000,60000'),
                   ('train_accuracy', ''),
                   ]
    gvar_args = [
        # ('gvar_estim_iter', 10),
        ('gvar_log_iter', 100),  # 100
        ('gvar_start', 0),
        ('g_osnap_iter', '100,2000,10000'),
        ('g_bsnap_iter', 10000),
        ('g_optim', ''),
        ('g_optim_start', 0),
        # ('g_epoch', ''),
    ]

    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_nuq_sgd = [
        ('g_estim', ['nuq']),
        ('nuq_bits', [2, 3, 4, 5, 6, 7]),
        ('nuq_bucket_size', [8192*2]),
        ('nuq_ngpu', 4),  # 2
        ('dist_num', [350]),
        ('nuq_layer', ''),
        ('nuq_ig_sm_bkts', ''),
        ('nuq_truncated_interval', 1),
        ('nuq_number_of_samples', 10),
        ('nuq_method', [
            ('amq', OrderedDict([('nuq_amq_lr', 0.7), ('nuq_amq_epochs', 40)])),
            ('amq_nb', OrderedDict([('nuq_amq_lr', 0.7), ('nuq_amq_epochs', 40)])),
            ('alq', OrderedDict([('nuq_cd_epochs', 30)])),
            'qinf',
            'trn',
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30), ('nuq_sym', ''), ('nuq_inv', '')])),
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30), ('nuq_inv', '')])),
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30), ('nuq_sym', '')])),
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30)])),
            ('nuq', OrderedDict([('nuq_mul', 0.5)])),
        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_nuq_sgd)]
    args_super_sgd = [
        ('g_estim', ['nuq']),
        ('nuq_ngpu', 4),  # 2
        ('nuq_truncated_interval', 1),
        ('nuq_number_of_samples', 10),
        ('nuq_method', [
            'none'
        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_super_sgd)]

    return args, log_dir, module_name, exclude

def bits(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_full' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch',
               'pretrained', 'niters', 'epoch_iters',
               'gvar_log_iter', 'gvar_start', 'g_bsnap_iter',
               'g_optim_start', 'nuq_truncated_interval', 'train_accuracy',
               'nuq_number_of_samples', 'chkpt_iter', 'g_osnap_iter']
    shared_args = [('dataset', dataset),
                   ('optim', ['sgd']),  # 'sgd', 'adam'
                   # ('arch', 'resnet32'),
                   ('arch', ['resnet8']),
                   ('batch_size', 128),
                   ('lr', [0.1]),
                   ('chkpt_iter', 2000),
                   ('momentum', 0.9),
                   ('weight_decay', 1e-4),
                   ('niters', 80000),
                   ('lr_decay_epoch', '40000,60000'),
                   ('train_accuracy', ''),
                   ]
    gvar_args = [
        # ('gvar_estim_iter', 10),
        ('gvar_log_iter', 100),  # 100
        ('gvar_start', 0),
        ('g_osnap_iter', '100,2000,10000'),
        ('g_bsnap_iter', 10000),
        ('g_optim', ''),
        ('g_optim_start', 0),
        # ('g_epoch', ''),
    ]

    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_nuq_sgd = [
        ('g_estim', ['nuq']),
        ('nuq_bits', [2, 3, 4, 5, 6, 7]),
        ('nuq_bucket_size', [8192*2]),
        ('nuq_ngpu', 4),  # 2
        ('dist_num', [350]),
        ('nuq_layer', ''),
        ('nuq_ig_sm_bkts', ''),
        ('nuq_truncated_interval', 1),
        ('nuq_number_of_samples', 10),
        ('nuq_method', [
            ('amq', OrderedDict([('nuq_amq_lr', 0.7), ('nuq_amq_epochs', 40)])),
            ('amq_nb', OrderedDict([('nuq_amq_lr', 0.7), ('nuq_amq_epochs', 40)])),
            ('alq', OrderedDict([('nuq_cd_epochs', 30)])),
            'qinf',
            'trn',
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30), ('nuq_sym', ''), ('nuq_inv', '')])),
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30), ('nuq_inv', '')])),
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30), ('nuq_sym', '')])),
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30)])),
            ('nuq', OrderedDict([('nuq_mul', 0.5)])),
        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_nuq_sgd)]
    args_super_sgd = [
        ('g_estim', ['nuq']),
        ('nuq_ngpu', 4),  # 2
        ('nuq_truncated_interval', 1),
        ('nuq_number_of_samples', 10),
        ('nuq_method', [
            'none'
        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_super_sgd)]

    return args, log_dir, module_name, exclude

def interval(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_full' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch',
               'pretrained', 'niters', 'epoch_iters',
               'gvar_log_iter', 'gvar_start', 'g_bsnap_iter',
               'g_optim_start', 'nuq_truncated_interval', 'train_accuracy',
               'nuq_number_of_samples', 'chkpt_iter', 'g_osnap_iter']
    shared_args = [('dataset', dataset),
                   ('optim', ['sgd']),  # 'sgd', 'adam'
                   # ('arch', 'resnet32'),
                   ('arch', ['resnet8']),
                   ('batch_size', 128),
                   ('lr', [0.1]),
                   ('chkpt_iter', 2000),
                   ('momentum', 0.9),
                   ('weight_decay', 1e-4),
                   ('niters', 80000),
                   ('lr_decay_epoch', '40000,60000'),
                   ('train_accuracy', ''),
                   ]
    gvar_args = [
        # ('gvar_estim_iter', 10),
        ('gvar_log_iter', 100),  # 100
        ('gvar_start', 0),
        ('g_osnap_iter', ['0,80000,80000', '40000,60000,80000', '100,1000,2000,40000,60000,80000']),
        ('g_bsnap_iter', 10000),
        ('g_optim', ''),
        ('g_optim_start', 0),
        # ('g_epoch', ''),
    ]

    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_nuq_sgd = [
        ('g_estim', ['nuq']),
        ('nuq_bits', [3]),
        ('nuq_bucket_size', [8192*2]),
        ('nuq_ngpu', 4),  # 2
        ('dist_num', [350]),
        ('nuq_layer', ''),
        ('nuq_ig_sm_bkts', ''),
        ('nuq_truncated_interval', 1),
        ('nuq_number_of_samples', 10),
        ('nuq_method', [
            ('amq', OrderedDict([('nuq_amq_lr', 0.7), ('nuq_amq_epochs', 40)])),
            ('amq_nb', OrderedDict([('nuq_amq_lr', 0.7), ('nuq_amq_epochs', 40)])),
            ('alq', OrderedDict([('nuq_cd_epochs', 30)])),
            'qinf',
            'trn',
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30), ('nuq_sym', ''), ('nuq_inv', '')])),
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30), ('nuq_inv', '')])),
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30), ('nuq_sym', '')])),
            ('alq_nb', OrderedDict([('nuq_cd_epochs', 30)])),
            ('nuq', OrderedDict([('nuq_mul', 0.5)])),
        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_nuq_sgd)]
    args_super_sgd = [
        ('g_estim', ['nuq']),
        ('nuq_ngpu', 4),  # 2
        ('nuq_truncated_interval', 1),
        ('nuq_number_of_samples', 10),
        ('nuq_method', [
            'none'
        ])
    ]
    args += [OrderedDict(shared_args+gvar_args+args_super_sgd)]

    return args, log_dir, module_name, exclude
