import torch
import torch.nn
import torch.multiprocessing
import numpy as np
import math
import random

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

    def _calc_stats_buckets(self, buckets):
        stats = {
            'sigma': [],
            'mean': []
        }
        i = 0
        for bucket in buckets:
            current_bk = torch.stack(buckets[bucket])
            stats['mean'].append(torch.mean(current_bk).cpu().item())
            stats['sigma'].append(torch.sqrt(torch.mean(
                torch.var(current_bk, dim=0, unbiased=False))).cpu().item())
            i += 1

        return stats

    def _get_grad_samples(self, model, num_of_samples):
        grads = []
        for i in range(num_of_samples):
            grad = self.grad_estim(model)
            copy_array = []
            for layer in grad:
                copy_array.append(layer.clone())

            grads.append(copy_array)
        return grads

    def _get_stats_lb(self, grads):
        # get stats layer based
        bs = self.opt.nuq_bucket_size
        nuq_layer = self.opt.nuq_layer
        sep_bias_grad = self.opt.sep_bias_grad

        # total number of weights
        nw = sum([w.numel() for w in grads[0]])

        # total sum of gradients
        tsum = torch.zeros(nw).cuda()

        buckets = None
        total_norm = None
        for i, grad in enumerate(grads):
            fl_norm_lb = self._flatt_and_normalize_lb(grad, bs, nocat=True)
            if buckets is None:
                buckets = [[] for j in range(len(fl_norm_lb))]
                total_norm = [0.0 for j in range(len(fl_norm_lb))]

            fl_norm = self._flatten_lb(grad, nocat=True)
            tsum += self._flatten_lb(fl_norm_lb, nocat=False)
            for j in range(len(fl_norm_lb)):
                buckets[j].append(fl_norm_lb[j])
                total_norm[j] += fl_norm[j].norm()

        stats = self._calc_stats_buckets(buckets)
        stats['norm'] = torch.tensor(total_norm)

        return stats

    def _get_stats_lb_sep(self, grads):
        # get stats layer based
        bs = self.opt.nuq_bucket_size
        nuq_layer = self.opt.nuq_layer
        sep_bias_grad = self.opt.sep_bias_grad

        buckets_bias = {}
        total_norm_bias = {}

        buckets_weights = {}
        total_norm_weights = {}

        samples = len(grads)

        fl_norm_bias, fl_norm_weights = self._flatten_sep(grads[0])
        fl_norm_lb_bias, fl_norm_lb_weights = \
            self._flatt_and_normalize_lb_sep(grads[0], bs, nocat=True)

        j = 0
        for layer in fl_norm_lb_bias:
            for bias in layer:
                buckets_bias[j] = []
                total_norm_bias[j] = 0.0
                j += 1

        j = 0
        for layer in fl_norm_lb_weights:
            for weights in layer:
                buckets_weights[j] = []
                total_norm_weights[j] = 0.0
                j += 1

        for i, grad in enumerate(grads):
            fl_norm_lb_bias, fl_norm_lb_weights = \
                self._flatt_and_normalize_lb_sep(grad, bs, nocat=True)

            fl_norm_bias, fl_norm_weights = self._flatten_lb_sep(grad, bs)

            j = 0
            for layer in fl_norm_lb_bias:
                for bias in layer:
                    buckets_bias[j].append(bias)
                    j += 1

            j = 0
            for layer in fl_norm_lb_weights:
                for weight in layer:
                    buckets_weights[j].append(weight)
                    j += 1

            j = 0
            for layer in fl_norm_bias:
                for bias in layer:
                    total_norm_bias[j] += bias.norm() / samples
                    j += 1

            j = 0
            for layer in fl_norm_weights:
                for weight in layer:
                    total_norm_weights[j] += weight.norm() / samples
                    j += 1

        stats_bias = self._calc_stats_buckets(buckets_bias)
        stats_bias['norm'] = torch.tensor(list(total_norm_bias.values()))
        stats_bias['norm'] = stats_bias['norm'].cpu().tolist()

        stats_weights = self._calc_stats_buckets(buckets_weights)
        stats_weights['norm'] = torch.tensor(list(total_norm_weights.values()))
        stats_weights['norm'] = stats_weights['norm'].cpu().tolist()

        stats = {
            'bias': stats_bias,
            'weights': stats_weights
        }

        return stats

    def _get_stats_sep(self, grads):
        # get stats for weights and bias separately
        pass

    def _get_stats_nl_lb(self, grads):
        # get stats normless

        bs = self.opt.nuq_bucket_size
        nuq_layer = self.opt.nuq_layer
        samples = len(grads)

        tsum = 0.0
        tot_var = 0.0

        num_params = len(self._flatt_and_normalize_lb(grads[0], bs))

        for grad in grads:
            params = self._flatt_and_normalize_lb(grad, bs)
            tsum += self._flatten([torch.cat(layer)
                                   for layer in params])

        mean = tsum / samples

        for grad in grads:
            params = self._flatt_and_normalize_lb_sep(grad, bs)
            tot_var += torch.sum((mean - self._flatten(
                [torch.cat(layer) for layer in params])) ** 2)

        tot_mean = tsum / num_params
        tot_var /= (num_params * samples)

        return {
            'mean': tot_mean,
            'var': tot_var
        }

    def _get_stats_nl_lb_sep(self, grads):
        # get normless stats, bias and weights separated

        bs = self.opt.nuq_bucket_size
        nuq_layer = self.opt.nuq_layer
        sep_bias_grad = self.opt.sep_bias_grad
        samples = len(grads)

        tsum_bias = 0.0
        tot_var_bias = 0.0

        tot_var_weights = 0.0
        tsum_weights = 0.0

        bias, weights = self._flatt_and_normalize_lb_sep(grads[0], bs)
        num_bias = len(torch.cat(bias))
        num_weights = len(torch.cat(weights))

        for grad in grads:
            bias, weights = self._flatt_and_normalize_lb_sep(grad, bs)
            tsum_bias += torch.cat(bias)
            tsum_weights += torch.cat(weights)

        mean_bias = tsum_bias / samples
        mean_weights = tsum_weights / samples

        for grad in grads:
            bias, weights = self._flatt_and_normalize_lb_sep(grad, bs)
            tot_var_bias += torch.sum((mean_bias - torch.cat(bias)) ** 2)
            tot_var_weights += torch.sum((mean_weights -
                                          torch.cat(weights)) ** 2)

        tot_mean_bias = torch.sum(mean_bias) / num_bias
        tot_mean_weights = torch.sum(mean_weights) / num_weights

        tot_var_weights /= (num_weights * samples)
        tot_var_bias /= (num_bias * samples)

        stats = {
            'bias': {
                'sigma': torch.sqrt(tot_var_bias).cpu().item(),
                'mean': tot_mean_bias.cpu().item()
            },
            'weights': {
                'sigma': torch.sqrt(tot_var_weights).cpu().item(),
                'mean': tot_mean_weights.cpu().item()
            }
        }
        return stats

    def _get_stats(self, grads):
        # get stats
        pass

    def snap_online(self, model):
        num_of_samples = self.opt.nuq_number_of_samples
        grads = self._get_grad_samples(model, num_of_samples)

        lb = True if self.opt.nuq_layer == 0 else False
        sep = True if self.opt.sep_bias_grad == 1 else False

        # TODO implement variations of lb and sep

        stats = {
            'nb': self._get_stats_lb_sep(grads),
            'nl': self._get_stats_nl_lb_sep(grads)
        }

        return stats

    def snap_online_mean(self, model):

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

        params = list(model.parameters())

        for i in range(num_of_samples):
            grad = self.grad_estim(model)
            flattened = self._flatten_lb(grad)
            for i, layer in enumerate(flattened):
                num_buckets = int(np.ceil(len(layer) / bs))
                for bucket in range(num_buckets):
                    start = bucket * bs
                    end = min((bucket + 1) * bs, len(layer))
                    current_bk = layer[start:end]
                    norm = current_bk.norm()
                    current_bk = current_bk / norm
                    b_len = len(current_bk)
                    # TODO: REMOVE THIS LINE
                    if b_len != bs:
                        continue
                    total_params += b_len
                    var = torch.var(current_bk)

                    # update norm-less variance
                    total_variance += var * (b_len - 1)
                    tot_sum += torch.sum(current_bk)

                    # update norm-based stats
                    stats_nb['norms'].append(norm)
                    stats_nb['sigmas'].append(
                        torch.sqrt(var))
                    stats_nb['means'].append(torch.mean(current_bk))

        nw = sum([w.numel() for w in model.parameters()])
        stats_nb['means'] = torch.stack(stats_nb['means']).cpu().tolist()
        stats_nb['sigmas'] = torch.stack(stats_nb['sigmas']).cpu().tolist()
        stats_nb['norms'] = torch.stack(stats_nb['norms']).cpu().tolist()

        stats = {
            'nb': stats_nb,
            'nl': {
                'mean': (tot_sum / total_params).cpu().item(),
                'sigma': torch.sqrt(total_variance / total_params).cpu().item(),
            }
        }
        return stats

    def grad(self, model_new, in_place=False, data=None):
        raise NotImplementedError('grad not implemented')

    def _normalize(self, layer, bucket_size, nocat=False):
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
        # TODO: make sure sub-classes never use any other data_iter, e.g. raw
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

    def _flatten_lb_sep(self, gradient, bs=None):
        # flatten layer based and handle weights and bias separately
        flatt_params = [], []

        for layer in gradient:
            if len(layer.size()) == 1:
                if bs is None:
                    flatt_params[0].append(
                        torch.flatten(layer))
                else:
                    buckets = []
                    flatt = torch.flatten(layer)
                    num_bucket = int(np.ceil(len(flatt) / bs))
                    for bucket_i in range(num_bucket):
                        start = bucket_i * bs
                        end = min((bucket_i + 1) * bs, len(flatt))
                        x_bucket = flatt[start:end].clone()
                        buckets.append(x_bucket)
                    flatt_params[0].append(
                        buckets)
            else:
                if bs is None:
                    flatt_params[1].append(
                        torch.flatten(layer))
                else:
                    buckets = []
                    flatt = torch.flatten(layer)
                    num_bucket = int(np.ceil(len(flatt) / bs))
                    for bucket_i in range(num_bucket):
                        start = bucket_i * bs
                        end = min((bucket_i + 1) * bs, len(flatt))
                        x_bucket = flatt[start:end].clone()
                        buckets.append(x_bucket)
                    flatt_params[1].append(
                        buckets)
        return flatt_params

    def _flatten_lb(self, gradient):
        # flatten layer based
        flatt_params = []

        for layer_parameters in gradient:
            flatt_params.append(torch.flatten(layer_parameters))

        return flatt_params

    def _flatten_sep(self, gradient, bs=None):
        # flatten weights and bias separately
        flatt_params = [], []

        for layer_parameters in gradient:
            if len(layer_parameters.size()) == 1:
                flatt_params[0].append(
                    torch.flatten(layer_parameters))
            else:
                flatt_params[1].append(torch.flatten(layer_parameters))
        return torch.cat(flatt_params[0]), torch.cat(flatt_params[1])

    def _flatten(self, gradient):
        flatt_params = []
        for layer_parameters in gradient:
            flatt_params.append(torch.flatten(layer_parameters))

        return torch.cat(flatt_params)

    def unflatten(self, gradient, parameters, tensor=False):
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

    def _flatt_and_normalize_lb_sep(self, gradient, bucket_size=1024,
                                    nocat=False):
        # flatten and normalize weight and bias separately

        bs = bucket_size
        # totally flat and layer-based layers
        flatt_params_lb = self._flatten_lb_sep(gradient)

        normalized_buckets_lb = [], []

        for bias in flatt_params_lb[0]:
            normalized_buckets_lb[0].append(
                self._normalize(bias, bucket_size, nocat))
        for weight in flatt_params_lb[1]:
            normalized_buckets_lb[1].append(
                self._normalize(weight, bucket_size, nocat))
        return normalized_buckets_lb

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

    def _flatt_and_normalize_sep(self, gradient,
                                 bucket_size=1024, nocat=False):
        flatt_params = self._flatten_sep(gradient)

        return [self._normalize(flatt_params[0], bucket_size, nocat),
                self._normalize(flatt_params[1], bucket_size, nocat)]

    def get_gradient_distribution(self, model, gviter, bucket_size):
        """
        gviter: Number of minibatches to apply on the model
        model: Model to be evaluated
        """
        bucket_size = self.opt.nuq_bucket_size
        mean_estimates_normalized = self._flatt_and_normalize(
            model.parameters(), bucket_size)
        mean_estimates_unconcatenated = self._flatt_and_normalize_lb(
            model.parameters(), bucket_size)
        # estimate grad mean and variance
        mean_estimates = [torch.zeros_like(g) for g in model.parameters()]
        mean_estimates_unconcatenated = [torch.zeros_like(
            g) for g in mean_estimates_unconcatenated]
        mean_estimates_normalized = torch.zeros_like(mean_estimates_normalized)

        for i in range(gviter):
            minibatch_gradient = self.grad_estim(model)
            minibatch_gradient_normalized = self._flatt_and_normalize(
                minibatch_gradient, bucket_size)
            minibatch_gradient_unconcatenated = self._flatt_and_normalize_lb(
                minibatch_gradient, bucket_size)

            for e, g in zip(mean_estimates, minibatch_gradient):
                e += g

            for e, g in zip(mean_estimates_unconcatenated, minibatch_gradient_unconcatenated):
                e += g

            mean_estimates_normalized += minibatch_gradient_normalized

        # Calculate the mean
        for e in mean_estimates:
            e /= gviter

        for e in mean_estimates_unconcatenated:
            e /= gviter

        mean_estimates_normalized /= gviter

        # Number of Weights
        number_of_weights = sum([layer.numel()
                                 for layer in model.parameters()])

        variance_estimates = [torch.zeros_like(g) for g in model.parameters()]
        variance_estimates_unconcatenated = [
            torch.zeros_like(g) for g in mean_estimates_unconcatenated]

        variance_estimates_normalized = torch.zeros_like(
            mean_estimates_normalized)

        for i in range(gviter):
            minibatch_gradient = self.grad_estim(model)
            minibatch_gradient_normalized = self._flatt_and_normalize(
                minibatch_gradient, bucket_size)
            minibatch_gradient_unconcatenated = self._flatt_and_normalize_lb(
                minibatch_gradient, bucket_size)

            v = [(gg - ee).pow(2)
                 for ee, gg in zip(mean_estimates, minibatch_gradient)]
            v_normalized = (mean_estimates_normalized -
                            minibatch_gradient_normalized).pow(2)
            v_normalized_unconcatenated = [(gg - ee).pow(2) for ee, gg in zip(
                mean_estimates_unconcatenated, minibatch_gradient_unconcatenated)]
            for e, g in zip(variance_estimates, v):
                e += g

            for e, g in zip(variance_estimates_unconcatenated, v_normalized_unconcatenated):
                e += g

            variance_estimates_normalized += v_normalized

        variance_estimates_normalized = variance_estimates_normalized / gviter
        for e in variance_estimates_unconcatenated:
            e /= gviter

        variances = []
        means = []
        # random_indices = self.get_random_index(model, 4)
        # for index in random_indices:
        #     variance_estimate_layer = variance_estimates[index[0]]
        #     mean_estimate_layer = mean_estimates[index[0]]

        #     for weight in index[1:]:
        #         variance_estimate_layer = variance_estimate_layer[weight]
        #         variance_estimate_layer.squeeze_()

        #         mean_estimate_layer = mean_estimate_layer[weight]
        #         mean_estimate_layer.squeeze_()
        #     variance = variance_estimate_layer / (gviter)

        #     variances.append(variance)
        #     means.append(mean_estimate_layer)

        total_mean = torch.tensor(0, dtype=float)
        for mean_estimate in mean_estimates:
            total_mean += torch.sum(mean_estimate)

        total_variance = torch.tensor(0, dtype=float)
        for variance_estimate in variance_estimates:
            total_variance += torch.sum(variance_estimate)

        total_variance = total_variance / number_of_weights
        total_mean = total_mean / number_of_weights

        total_variance_normalized = torch.tensor(0, dtype=float)
        total_variance_normalized = torch.sum(
            variance_estimates_normalized) / number_of_weights
        total_mean_normalized = torch.tensor(0, dtype=float)
        total_mean_normalized = torch.sum(
            mean_estimates_normalized) / number_of_weights
        total_mean_unconcatenated = sum([torch.sum(
            mean) / mean.numel() for mean in mean_estimates_unconcatenated]) / len(mean_estimates)
        total_variance_unconcatenated = sum([torch.sum(variance) / variance.numel(
        ) for variance in variance_estimates_unconcatenated]) / len(mean_estimates)

        return variances, means, total_mean, total_variance, total_variance_normalized, total_mean_normalized, total_mean_unconcatenated, total_variance_unconcatenated

    def get_norm_distribution(self, model, gviter, bucket_size=1024):
        norms = {}
        for i in range(gviter):
            minibatch_gradient = self.grad_estim(model)
            flattened_parameters = self._flatten(
                minibatch_gradient)
            num_bucket = int(np.ceil(len(flattened_parameters) / bucket_size))
            for bucket_i in range(num_bucket):
                start = bucket_i * bucket_size
                end = min((bucket_i + 1) * bucket_size,
                          len(flattened_parameters))
                if (end == len(flattened_parameters)):
                    continue
                x_bucket = flattened_parameters[start:end].clone()
                norm = x_bucket.norm()
                if norm.cpu() in norms.keys():
                    print('An error occured')
                norms[norm.cpu()] = x_bucket
        return norms

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def snap_model(self, model):
        logging.info('Snap Model')
        if self.model is None:
            self.model = copy.deepcopy(model)
            return
        # update sum
        for m, s in zip(model.parameters(), self.model.parameters()):
            s.data.copy_(m.data)
