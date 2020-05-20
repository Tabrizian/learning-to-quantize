import torch
import torch.nn
import torch.multiprocessing
import numpy as np


from estim.sgd import SGDEstimator
from estim.nuq import NUQEstimator


class MinVarianceGradient(object):
    def __init__(self, model, data_loader, opt, tb_logger):
        self.model = model
        sgd = SGDEstimator(data_loader, opt, tb_logger)
        if opt.g_estim == 'sgd':
            gest = SGDEstimator(data_loader, opt, tb_logger)
        elif opt.g_estim == 'nuq':
            if opt.nuq_parallel == 'no':
                gest = NUQEstimator(data_loader, opt, tb_logger)
        self.sgd = sgd
        self.gest = gest
        self.opt = opt
        self.tb_logger = tb_logger
        self.gest_used = False
        self.Esgd = 0
        self.last_log_iter = 0
        self.opt = opt

    def is_log_iter(self, niters):
        opt = self.opt
        if (niters-self.last_log_iter >= opt.gvar_log_iter
                and niters >= opt.gvar_start):
            self.last_log_iter = niters
            return True
        return False

    def log_var(self, model, niters):
        tb_logger = self.tb_logger
        gviter = self.opt.gvar_estim_iter
        Ege, var_e, snr_e, nv_e = self.gest.get_Ege_var(model, gviter)
        Esgd, var_s, snr_s, nv_s = self.sgd.get_Ege_var(model, gviter)
        bias = torch.mean(torch.cat(
            [(ee-gg).abs().flatten() for ee, gg in zip(Ege, Esgd)]))
        if self.opt.g_estim == 'nuq':
            if self.opt.nuq_method != 'none':
                tb_logger.log_value('bits', float(
                    self.gest.qdq.bits), step=niters)
                tb_logger.log_value('levels', float(
                    len(self.gest.qdq.levels)), step=niters)
                for index, level in enumerate(self.gest.qdq.levels):
                    tb_logger.log_value(
                        'levels/' + str(index), float(level), step=niters)
                tb_logger.log_value('includes_zero', float(
                    1 if 0 in self.gest.qdq.levels else 0), step=niters)
                number_of_positive_levels = 0
                number_of_negative_levels = 0
                for level in self.gest.qdq.levels:
                    if level > 0:
                        number_of_positive_levels += 1
                    elif level < 0:
                        number_of_negative_levels += 1
                tb_logger.log_value('positive_levels', float(
                    number_of_positive_levels), step=niters)
                tb_logger.log_value('negative_levels', float(
                    number_of_negative_levels), step=niters)
                if self.gest.qdq.error is not None:
                    tb_logger.log_value(
                        'nb_error', self.gest.qdq.error, step=niters)
                if self.gest.qdq.grad_dist_nl is not None:
                    tb_logger.log_value(
                        'stats/mean', self.gest.qdq.grad_dist_nl.mean, step=niters)
                    tb_logger.log_value(
                        'stats/sigma', self.gest.qdq.grad_dist_nl.sigma, step=niters)

            if self.opt.nuq_method == 'amq' or self.opt.nuq_method == 'amq_nb':
                tb_logger.log_value('multiplier', float(
                    self.gest.qdq.multiplier), step=niters)

        print('est_var is', var_e)
        tb_logger.log_value('grad_bias', float(bias), step=niters)
        tb_logger.log_value('est_var', float(var_e), step=niters)
        tb_logger.log_value('sgd_var', float(var_s), step=niters)
        tb_logger.log_value('est_snr', float(snr_e), step=niters)
        tb_logger.log_value('sgd_snr', float(snr_s), step=niters)
        tb_logger.log_value('est_nvar', float(nv_e), step=niters)
        tb_logger.log_value('sgd_nvar', float(nv_s), step=niters)
        sgd_x, est_x = ('', '[X]') if self.gest_used else ('[X]', '')
        return ('G Bias: %.8f\t'
                '%sSGD Var: %.8f\t %sEst Var: %.8f\t'
                'SGD N-Var: %.8f\t Est N-Var: %.8f\t'
                % (43, sgd_x, var_s, est_x, var_e, nv_s, nv_e))

    def grad(self, niters):
        model = self.model
        model.train()
        use_sgd = self.use_sgd(niters)
        if use_sgd:
            self.gest_used = False
            return self.sgd.grad(model, in_place=True)
        self.gest_used = True
        return self.gest.grad(model, in_place=True)

    def use_sgd(self, niters):
        return not self.opt.g_optim or niters < self.opt.g_optim_start

    def state_dict(self):
        return self.gest.state_dict()

    def load_state_dict(self, state):
        self.gest.load_state_dict(state)
