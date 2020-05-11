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

    def snap_online(self, model):
        total_variance = 0
        iterations = self.opt.nuq_number_of_samples 
        nw = sum([w.numel() for w in model.parameters()])
        tsum = torch.zeros(nw).cuda()

        grads = []
        for i in range(iterations):
            grad = self.grad_estim(model)
            copy_array = []
            for layer in grad:
                copy_array.append(layer.clone())

            grads.append(copy_array)

        bucket_size = self.opt.nuq_bucket_size
        nuq_layer = self.opt.nuq_layer
        buckets_normalized = {}
        buckets = {}
        for i in range(iterations):
            grad = grads[i]
            flattened, flattened_lb = self.flatten_and_normalize(grad, bucket_size)
            flattened_lb_flt, _ = self.flatten(flattened_lb) 
            flattened_unnormalized, flattened_unnormalized_lb = self.flatten(grad)
            with torch.no_grad():
                if self.opt.nuq_layer == 0:
                    tsum += flattened_lb_flt
                else:
                    tsum += flattened
            if self.opt.nuq_layer == 1:
                flattened_unnormalized, _ = self.flatten(grad)
                num_bucket = int(np.ceil(len(flattened) / bucket_size))
                for bucket_i in range(num_bucket):
                    start = bucket_i * bucket_size
                    end = min((bucket_i + 1) * bucket_size, len(flattened))
                    x_bucket = flattened[start:end].clone()
                    x_bucket_unnormalized = flattened_unnormalized[start:end].clone()
                    if bucket_i not in buckets_normalized.keys():
                        buckets_normalized[bucket_i] = []
                        buckets[bucket_i] = 0

                    buckets_normalized[bucket_i].append(x_bucket)
                    buckets[bucket_i] += x_bucket_unnormalized.norm()
            else:
                bucket_index = 0
                for layer, flattened in enumerate(flattened_lb):
                    num_bucket = int(np.ceil(len(flattened) / bucket_size))
                    for bucket_i in range(num_bucket):
                        start = bucket_i * bucket_size
                        end = min((bucket_i + 1) * bucket_size, len(flattened))
                        x_bucket = flattened[start:end].clone()
                        x_bucket_unnormalized = flattened_unnormalized_lb[layer][start:end].clone()
                        if i == 0:
                            buckets_normalized[bucket_index] = []
                            buckets[bucket_index] = 0

                        buckets_normalized[bucket_index].append(x_bucket)
                        buckets[bucket_index] += x_bucket_unnormalized.norm()
                        bucket_index += 1


        
        mean = tsum / iterations
        total_mean = torch.sum(mean) / (nw)

        for i in range(iterations):
            grad = grads[i]
            flattened, flattened_lb = self.flatten_and_normalize(grad, bucket_size)

            flattened_lb_flt, _ = self.flatten(flattened_lb)

            with torch.no_grad():
                if self.opt.nuq_layer == 0:
                    total_variance += (flattened_lb_flt - mean).pow(2).sum()
                else:
                    total_variance += (flattened - mean).pow(2).sum()
        
        norm_results = {
            'norm': [],
            'sigma': [],
            'mean': []
        }

        i = 0
        for bucket in buckets_normalized:
            current_bk = torch.stack(buckets_normalized[bucket])
            norm_results['mean'].append(torch.mean(current_bk).cpu().item())
            norm_results['sigma'].append(torch.sqrt(torch.mean(torch.var(current_bk, dim=0, unbiased=False))).cpu().item())
            norm_results['norm'].append((buckets[i] / iterations).cpu().item())
            i += 1

        # if len(norm_results['mean']) > self.opt.dist_num:
        #     indexes = np.argsort(-np.asarray(norm_results['norm']))[:self.opt.dist_num]
        #     norm_results['mean'] = np.array(norm_results['mean'])[indexes].tolist()
        #     norm_results['sigma'] = np.array(norm_results['sigma'])[indexes].tolist()
        #     norm_results['norm'] = np.array(norm_results['norm'])[indexes].tolist()
        total_variance /= (iterations * nw)
        print(norm_results)

        return total_mean, total_variance, norm_results

        #return torch.tensor(0), torch.tensor(0.01), norm_results

    def grad(self, model_new, in_place=False, data=None):
        raise NotImplementedError('grad not implemented')

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

    def flatten(self, gradient):
        flattened_parameters = []
        for layer_parameters in gradient:
            flattened_parameters.append(torch.flatten(layer_parameters))
        return torch.cat(flattened_parameters), flattened_parameters
    
    def unflatten(self, gradient, parameters):
        shaped_gradient = []
        begin = 0
        for layer in parameters:
            size = layer.view(-1).shape[0]
            shaped_gradient.append(gradient[begin:begin+size].view(layer.shape))
            begin += size
        return shaped_gradient
        
    
    def flatten_and_normalize(self, gradient, bucket_size=1024):
        flattened_parameters, less_flattened = self.flatten(gradient)
        num_bucket = int(np.ceil(len(flattened_parameters) / bucket_size))
        normalized_buckets = []
        for bucket_i in range(num_bucket):
            start = bucket_i * bucket_size
            end = min((bucket_i + 1) * bucket_size, len(flattened_parameters))
            x_bucket = flattened_parameters[start:end].clone()
            norm = x_bucket.norm()
            normalized_buckets.append(x_bucket / (norm + 1e-7))

        unconcatenated_buckets = []
        for layer in less_flattened:
            num_bucket = int(np.ceil(len(layer) / bucket_size))
            normalized_unconcatenated_buckets = []
            for bucket_i in range(num_bucket):
                start = bucket_i * bucket_size
                end = min((bucket_i + 1) * bucket_size, len(layer))
                x_bucket = layer[start:end].clone()
                norm = x_bucket.norm()
                normalized_unconcatenated_buckets.append(x_bucket / (norm + 1e-7))
            unconcatenated_buckets.append(torch.cat(normalized_unconcatenated_buckets))
        return torch.cat(normalized_buckets), unconcatenated_buckets

    def get_random_index(self, model, number):
        if self.random_indices == None:
            parameters = list(model.parameters())
            random_indices = []
            # Fix the randomization seed
            begin = 0
            end = int(len(parameters) / number)
            for i in range(number): 
                random_layer = torch.randint(begin, end, (1,))
                random_weight_layer_size = parameters[random_layer].shape
                random_weight_array = [random_layer]
                for weight in random_weight_layer_size:
                    random_weight_array.append(torch.randint(0, weight, (1,)))
                random_indices.append(random_weight_array)
                begin = end
                end =  int((i + 2) * len(parameters) / number)
            self.random_indices = random_indices 
        return self.random_indices

    def get_gradient_distribution(self, model, gviter, bucket_size):
        """
        gviter: Number of minibatches to apply on the model
        model: Model to be evaluated
        """
        bucket_size = self.opt.nuq_bucket_size
        mean_estimates_normalized, mean_estimates_unconcatenated = self.flatten_and_normalize(model.parameters(), bucket_size)
        # estimate grad mean and variance
        mean_estimates = [torch.zeros_like(g) for g in model.parameters()]
        mean_estimates_unconcatenated = [torch.zeros_like(g) for g in mean_estimates_unconcatenated]
        mean_estimates_normalized = torch.zeros_like(mean_estimates_normalized)

        for i in range(gviter):
            minibatch_gradient = self.grad_estim(model)
            minibatch_gradient_normalized, minibatch_gradient_unconcatenated = self.flatten_and_normalize(minibatch_gradient, bucket_size)


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
        number_of_weights = sum([layer.numel() for layer in model.parameters()])

        variance_estimates = [torch.zeros_like(g) for g in model.parameters()]
        variance_estimates_unconcatenated = [torch.zeros_like(g) for g in mean_estimates_unconcatenated]

        variance_estimates_normalized = torch.zeros_like(mean_estimates_normalized)

        for i in range(gviter):
            minibatch_gradient = self.grad_estim(model)
            minibatch_gradient_normalized, minibatch_gradient_unconcatenated = self.flatten_and_normalize(minibatch_gradient, bucket_size)

            v = [(gg - ee).pow(2) for ee, gg in zip(mean_estimates, minibatch_gradient)]
            v_normalized = (mean_estimates_normalized - minibatch_gradient_normalized).pow(2)
            v_normalized_unconcatenated = [(gg - ee).pow(2) for ee, gg in zip(mean_estimates_unconcatenated, minibatch_gradient_unconcatenated)]
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
        total_variance_normalized = torch.sum(variance_estimates_normalized) / number_of_weights
        total_mean_normalized = torch.tensor(0, dtype=float)
        total_mean_normalized = torch.sum(mean_estimates_normalized) / number_of_weights
        total_mean_unconcatenated = sum([torch.sum(mean) / mean.numel() for mean in mean_estimates_unconcatenated]) / len(mean_estimates)
        total_variance_unconcatenated = sum([torch.sum(variance) / variance.numel() for variance in variance_estimates_unconcatenated]) / len(mean_estimates)

        return variances, means, total_mean, total_variance, total_variance_normalized, total_mean_normalized, total_mean_unconcatenated, total_variance_unconcatenated

    def get_norm_distribution(self, model, gviter, bucket_size=1024):
        norms = {}
        for i in range(gviter):
            minibatch_gradient = self.grad_estim(model)
            flattened_parameters, less_flattened = self.flatten(minibatch_gradient)
            num_bucket = int(np.ceil(len(flattened_parameters) / bucket_size))
            for bucket_i in range(num_bucket):
                start = bucket_i * bucket_size
                end = min((bucket_i + 1) * bucket_size, len(flattened_parameters))
                if (end == len(flattened_parameters)):
                    continue
                x_bucket = flattened_parameters[start:end].clone()
                norm = x_bucket.norm()
                if norm.cpu() in norms.keys():
                    print('It shouldn\'t happend!!!!!!!!!!!')
                norms[norm.cpu()] =  x_bucket
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