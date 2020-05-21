import logging
import torch

import utils
from data import get_minvar_loader
from log_utils import LogCollector
from estim.gvar import MinVarianceGradient


class OptimizerFactory(object):

    def __init__(self, model, train_loader, tb_logger, opt):
        self.model = model
        self.opt = opt
        self.niters = 0
        self.optimizer = None
        self.epoch = 0
        self.logger = LogCollector(opt)
        self.param_groups = None
        self.gest_used = False
        minvar_loader = get_minvar_loader(train_loader, opt)
        self.gvar = MinVarianceGradient(
            model, minvar_loader, opt, tb_logger)
        self.reset()

    def reset(self):
        model = self.model
        opt = self.opt
        if opt.optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=opt.lr, momentum=opt.momentum,
                                        weight_decay=opt.weight_decay,
                                        nesterov=opt.nesterov)
        elif opt.optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=opt.lr,
                                         weight_decay=opt.weight_decay)
        self.optimizer = optimizer
        if self.param_groups is not None:
            self.optimizer.param_groups = self.param_groups
        else:
            self.param_groups = self.optimizer.param_groups

    def step(self, profiler):
        gvar = self.gvar
        opt = self.opt
        model = self.model

        self.optimizer.zero_grad()

        # Frequent snaps
        inits = list(map(int, opt.g_osnap_iter.split(',')[:-1]))
        every = int(opt.g_osnap_iter.split(',')[-1])
        oitercond = (self.niters - opt.gvar_start) % every == 0

        if (oitercond or self.niters in inits) and \
           self.niters >= opt.gvar_start:
            print(self.niters)

            if opt.g_estim == 'nuq' and opt.nuq_method != 'none':
                stats = gvar.gest.snap_online_mean(model)
                if opt.nuq_parallel == 'ngpu':
                    for qdq in gvar.gest.qdq:
                        qdq.set_mean_variance(stats)
                else:
                    gvar.gest.qdq.set_mean_variance(stats)

            isamq = opt.nuq_method == 'amq' or opt.nuq_method == 'amq_nb'
            isalq = opt.nuq_method == 'alq' or opt.nuq_method == 'alq_nb'
            if isamq or isalq:
                if opt.nuq_parallel == 'ngpu':
                    for qdq in gvar.gest.qdq:
                        qdq.update_levels()
                else:
                    gvar.gest.qdq.update_levels()

        pg_used = gvar.gest_used
        loss = gvar.grad(self.niters)
        if gvar.gest_used != pg_used:
            logging.info('Optimizer reset.')
            self.gest_used = gvar.gest_used
            utils.adjust_lr(self, opt)
            self.reset()
        self.optimizer.step()
        profiler.toc('optim')

        profiler.end()
        return loss
