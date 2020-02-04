import torch
import torch.nn
import torch.multiprocessing



from estim.sgd import SGDEstimator
from estim.nuq import NUQEstimator
from estim.nuq import NUQEstimatorSingleGPUParallel
from estim.nuq import NUQEstimatorMultiGPUParallel


class MinVarianceGradient(object):
    def __init__(self, model, data_loader, opt, tb_logger, simple_logger):
        self.model = model
        sgd = SGDEstimator(data_loader, opt, tb_logger)
        if opt.g_estim == 'sgd':
            gest = SGDEstimator(data_loader, opt, tb_logger)
        elif opt.g_estim == 'nuq':
            if opt.nuq_parallel == 'no':
                gest = NUQEstimator(data_loader, opt, tb_logger)
            elif opt.nuq_parallel == 'gpu1':
                gest = NUQEstimatorSingleGPUParallel(
                    data_loader, opt, tb_logger)
            else:
                gest = NUQEstimatorMultiGPUParallel(
                    data_loader, opt, tb_logger)
        self.sgd = sgd
        self.gest = gest
        self.opt = opt
        self.tb_logger = tb_logger
        self.gest_used = False
        self.Esgd = 0
        self.last_log_iter = 0
        self.opt = opt
        self.writer = simple_logger


    def is_log_iter(self, niters):
        opt = self.opt
        if (niters-self.last_log_iter >= opt.gvar_log_iter
                and niters >= opt.gvar_start):
            self.last_log_iter = niters
            return True
        return False

    def log_var(self, model, niters):
        tb_logger = self.tb_logger
        writer = self.writer
        gviter = self.opt.gvar_estim_iter
        Ege, var_e, snr_e, nv_e = self.gest.get_Ege_var(model, gviter)
        Esgd, var_s, snr_s, nv_s = self.sgd.get_Ege_var(model, gviter)
        if self.opt.g_estim == 'sgd':
            parameters = torch.cat([layer.view(-1) for layer in self.sgd.grad(model)])
            tb_logger.log_histogram('sgd_dist', parameters, step=niters)
        variances, means, total_mean, total_variance, total_variance_normalized, total_mean_normalized, total_mean_unconcatenated, total_variance_unconcatenated = self.gest.get_gradient_distribution(model, gviter)
        bias = torch.mean(torch.cat(
            [(ee-gg).abs().flatten() for ee, gg in zip(Ege, Esgd)]))
        if self.opt.g_estim == 'nuq':
            if self.opt.nuq_method == 'nuq3' or self.opt.nuq_method == 'nuq4':
                if self.opt.nuq_method == 'nuq3':
                    tb_logger.log_value('co_error', float(self.gest.qdq.error), step=niters)
                tb_logger.log_value('multiplier', float(self.gest.qdq.multiplier), step=niters)
                tb_logger.log_value('bits', float(self.gest.qdq.bits), step=niters)
                

        
        
        tb_logger.log_value('grad_bias', float(bias), step=niters)
        tb_logger.log_value('est_var', float(var_e), step=niters)
        tb_logger.log_value('sgd_var', float(var_s), step=niters)
        tb_logger.log_value('est_snr', float(snr_e), step=niters)
        tb_logger.log_value('sgd_snr', float(snr_s), step=niters)
        tb_logger.log_value('est_nvar', float(nv_e), step=niters)
        tb_logger.log_value('sgd_nvar', float(nv_s), step=niters)
        tb_logger.log_value('tot_var_norm', float(total_variance_normalized), step=niters)
        tb_logger.log_value('tot_var', float(total_variance), step=niters)
        tb_logger.log_value('tot_mean_norm', float(total_mean_normalized), step=niters)
        tb_logger.log_value('tot_mean', float(total_mean), step=niters)
        tb_logger.log_value
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
