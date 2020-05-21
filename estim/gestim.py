import torch
import torch.nn
import torch.multiprocessing
import numpy as np

import copy
import logging

from data import InfiniteLoader


class GradientEstimator(object):
    def __init__(self, data_loader, opt, tb_logger=None, *args, **kwargs):
        self.opt = opt
        self.model = None
        self.data_loader = data_loader
        self.tb_logger = tb_logger
        self.niters = 0
        self.random_indices = None

    def update_niters(self, niters):
        self.niters = niters

    def init_data_iter(self):
        self.data_iter = iter(InfiniteLoader(self.data_loader))
        self.estim_iter = iter(InfiniteLoader(self.data_loader))

    def snap_batch(self, model):
        pass

    def update_sampler(self):
        pass

    def _get_raw_grad(self, model):
        dt = self.data_iter
        self.data_iter = self.estim_iter
        model.zero_grad()
        data = next(self.data_iter)
        loss = model.criterion(model, data)
        grad = torch.autograd.grad(loss, model.parameters())
        self.data_iter = dt
        return grad

    def _bucketize(self, grad, bs, stats_nb):
        """Calculate the stats for a single bucket
        Parameters:
            grad (torch.Tensor): gradient vector
            bs (int): bucket size
            stats_nb (dict): dictionary containing norm-based statistics
        """
        ig_sm_bkts = self.opt.nuq_ig_sm_bkts
        variance = 0
        num_params = 0
        tot_sum = 0
        num_buckets = int(np.ceil(len(grad) / bs))
        for bucket in range(num_buckets):
            start = bucket * bs
            end = min((bucket + 1) * bs, len(grad))
            current_bk = grad[start:end]
            norm = current_bk.norm()
            current_bk = current_bk / norm
            b_len = len(current_bk)
            if b_len != bs and ig_sm_bkts:
                continue
            num_params += b_len
            var = torch.var(current_bk)

            # update norm-less variance
            variance += var * (b_len - 1)
            tot_sum += torch.sum(current_bk)

            stats_nb['norms'].append(norm)
            stats_nb['sigmas'].append(torch.sqrt(var))
            stats_nb['means'].append(torch.mean(current_bk))

        return tot_sum, variance, num_params

    def snap_online_mean(self, model):
        """Sample the gradient and calculate the stats
        """

        stats_nb = {
            'means': [],
            'sigmas': [],
            'norms': []
        }

        total_variance = 0.0
        tot_sum = 0.0

        num_of_samples = self.opt.nuq_number_of_samples
        total_params = 0
        bs = self.opt.nuq_bucket_size

        lb = not self.opt.nuq_layer

        for i in range(num_of_samples):
            grad = self._get_raw_grad(model)
            if lb:
                flattened = self._flatten_lb(grad)
                for i, layer in enumerate(flattened):
                    b_sum, b_var, b_params = self._bucketize(
                        layer, bs, stats_nb)
                    tot_sum += b_sum
                    total_variance += b_var
                    total_params += b_params
            else:
                flattened = self._flatten(grad)
                b_sum, b_var, b_params = self._bucketize(
                    flattened, bs, stats_nb)
                tot_sum += b_sum
                total_variance += b_var
                total_params += b_params

        stats_nb['means'] = torch.stack(stats_nb['means']).cpu().tolist()
        stats_nb['sigmas'] = torch.stack(stats_nb['sigmas']).cpu().tolist()
        stats_nb['norms'] = torch.stack(stats_nb['norms']).cpu().tolist()

        # Select the most significant norms
        if len(stats_nb['means']) > self.opt.dist_num:
            indexes = np.argsort(-np.asarray(stats_nb['norms']))[
                :self.opt.dist_num]
            stats_nb['means'] = np.array(stats_nb['means'])[indexes].tolist()
            stats_nb['sigmas'] = np.array(stats_nb['sigmas'])[
                indexes].tolist()
            stats_nb['norms'] = np.array(stats_nb['norms'])[indexes].tolist()

        stats = {
            'nb': stats_nb,
            'nl': {
                'mean': (tot_sum / total_params).cpu().item(),
                'sigma':
                torch.sqrt(total_variance / total_params).cpu().item(),
            }
        }
        return stats

    def grad(self, model_new, in_place=False, data=None):
        raise NotImplementedError('grad not implemented')

    def _normalize(self, layer, bucket_size, nocat=False):
        """normalize gradients of a single layer
        """
        normalized = []
        num_bucket = int(np.ceil(len(layer) / bucket_size))
        for bucket_i in range(num_bucket):
            start = bucket_i * bucket_size
            end = min((bucket_i + 1) * bucket_size, len(layer))
            x_bucket = layer[start:end].clone()
            norm = x_bucket.norm()
            normalized.append(x_bucket / (norm + 1e-7))
        if not nocat:
            return torch.cat(normalized)
        else:
            return normalized

    def grad_estim(self, model):
        # ensuring continuity of data seen in training
        dt = self.data_iter
        self.data_iter = self.estim_iter
        ret = self.grad(model)
        self.data_iter = dt
        return ret

    def get_Ege_var(self, model, gviter):
        # estimate grad mean and variance
        Ege = [torch.zeros_like(g) for g in model.parameters()]
        for i in range(gviter):
            ge = self.grad_estim(model)
            for e, g in zip(Ege, ge):
                e += g

        for e in Ege:
            e /= gviter

        nw = sum([w.numel() for w in model.parameters()])
        var_e = 0
        Es = [torch.zeros_like(g) for g in model.parameters()]
        En = [torch.zeros_like(g) for g in model.parameters()]
        for i in range(gviter):
            ge = self.grad_estim(model)
            v = sum([(gg-ee).pow(2).sum() for ee, gg in zip(Ege, ge)])
            for s, e, g, n in zip(Es, Ege, ge, En):
                s += g.pow(2)
                n += (e-g).pow(2)
            var_e += v/nw

        var_e /= gviter
        # Division by gviter cancels out in ss/nn
        snr_e = sum(
            [((ss+1e-10).log()-(nn+1e-10).log()).sum()
             for ss, nn in zip(Es, En)])/nw
        nv_e = sum([(nn/(ss+1e-7)).sum() for ss, nn in zip(Es, En)])/nw
        return Ege, var_e, snr_e, nv_e

    def _flatten_lb(self, gradient):
        """flatten the gradient in every layer
        """
        flatt_params = []

        for layer_parameters in gradient:
            flatt_params.append(torch.flatten(layer_parameters))

        return flatt_params

    def _flatten(self, gradient):
        flatt_params = []
        for layer_parameters in gradient:
            flatt_params.append(torch.flatten(layer_parameters))

        return torch.cat(flatt_params)

    def unflatten(self, gradient, parameters, tensor=False):
        """Change the shape of the gradient to the shape of the parameters

        Parameters:
            gradient: flattened gradient
            parameters: convert the flattened gradient to the unflattened
                        version
            tensor: convert to tonsor otherwise it will be an array
        """
        shaped_gradient = []
        begin = 0
        for layer in parameters:
            size = layer.view(-1).shape[0]
            shaped_gradient.append(
                gradient[begin:begin+size].view(layer.shape))
            begin += size
        if tensor:
            return torch.stack(shaped_gradient)
        else:
            return shaped_gradient

    def _flatt_and_normalize_lb(self, gradient, bucket_size=1024, nocat=False):
        flatt_params_lb = self._flatten_lb(gradient)

        normalized_buckets_lb = []
        for layer in flatt_params_lb:
            normalized_buckets_lb.append(
                self._normalize(layer, bucket_size, nocat))
        return normalized_buckets_lb

    def _flatt_and_normalize(self, gradient, bucket_size=1024, nocat=False):
        flatt_params = self._flatten(gradient)

        return self._normalize(flatt_params, bucket_size, nocat)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass
