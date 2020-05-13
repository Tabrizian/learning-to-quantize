import torch
import torch.nn
import torch.multiprocessing
import numpy as np
import copy

from args import opt_to_nuq_kwargs
from .gestim import GradientEstimator
from nuq.quantize import QuantizeMultiBucket


class NUQEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(NUQEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.qdq = QuantizeMultiBucket(**opt_to_nuq_kwargs(self.opt))
        self.ngpu = self.opt.nuq_ngpu
        self.acc_grad = None

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
                x_bucket = flattened_parameters[start:end].clone()
                if bucket_i not in norms.keys():
                    norms[bucket_i] = []
                norms[bucket_i].append(x_bucket)
        return norms

    def grad(self, model_new, in_place=False):
        model = model_new

        if self.acc_grad is None:
            self.acc_grad = []
            with torch.no_grad():
                for p in model.parameters():
                    self.acc_grad += [torch.zeros_like(p)]
        else:
            for a in self.acc_grad:
                a.zero_()

        for i in range(self.ngpu):
            model.zero_grad()
            data = next(self.data_iter)
            loss = model.criterion(model, data)
            grad = torch.autograd.grad(loss, model.parameters())
            layers = len(list(model.parameters()))

            per_layer = False
            with torch.no_grad():
                if self.opt.nuq_layer == 1:
                    flattened_array = self._flatten(grad)
                    gradient_quantized = self.qdq.quantize(
                        flattened_array, layers) / self.ngpu
                    unflattened_array = self.unflatten(
                        gradient_quantized, grad)
                    for g, a in zip(unflattened_array, self.acc_grad):
                        a += g
                else:
                    for g, a in zip(grad, self.acc_grad):
                        a += self.qdq.quantize(g, layers) / self.ngpu

        if in_place:
            for p, a in zip(model.parameters(), self.acc_grad):
                if p.grad is None:
                    p.grad = a.clone()
                else:
                    p.grad.copy_(a)
            return loss
        return self.acc_grad


class NUQEstimatorMultiGPUParallel(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(NUQEstimatorMultiGPUParallel, self).__init__(*args, **kwargs)
        self.init_data_iter()
        nuq_kwargs = opt_to_nuq_kwargs(self.opt)
        self.ngpu = self.opt.nuq_ngpu
        self.acc_grad = None
        self.models = None
        self.qdq = []
        for i in range(self.ngpu):
            with torch.cuda.device(i):
                self.qdq += [QuantizeMultiBucket(**nuq_kwargs)]

    def grad(self, model_new, in_place=False):
        if self.models is None:
            self.models = [model_new]
            for i in range(1, self.ngpu):
                with torch.cuda.device(i):
                    self.models += [copy.deepcopy(model_new)]
                    self.models[-1] = self.models[-1].cuda()

        else:
            # sync weights
            for i in range(1, self.ngpu):
                for p0, pi in zip(self.models[0].parameters(),
                                  self.models[i].parameters()):
                    with torch.no_grad():
                        pi.copy_(p0)

        models = self.models

        # forward-backward prop
        loss = []
        for i in range(self.ngpu):
            models[i].zero_grad()  # criterion does it
            data = next(self.data_iter)
            with torch.cuda.device(i):
                loss += [models[i].criterion(models[i], data)]
                loss[i].backward()

        loss = loss[-1]

        layers = len(list(models[0].parameters()))
        # quantize all grads
        for i in range(self.ngpu):
            with torch.no_grad():
                with torch.cuda.device(i):
                    torch.cuda.synchronize()
                    if self.opt.nuq_layer == 1:
                        flattened_array = self._flatten(
                            models[i].parameters())
                        gradient_quantized = self.qdq[i].quantize(
                            flattened_array, layers) / self.ngpu
                        unflattened_array = self.unflatten(
                            gradient_quantized, models[i].parameters())
                        for p, q in zip(models[i].parameters(), unflattened_array):
                            p.grad.copy_(q)
                    else:
                        for p in models[i].parameters():
                            p.grad.copy_(self.qdq[i].quantize(
                                p.grad, layers) / self.ngpu)

        # aggregate grads into gpu0
        for i in range(1, self.ngpu):
            for p0, pi in zip(models[0].parameters(), models[i].parameters()):
                p0.grad.add_(pi.grad.to('cuda:0'))

        if in_place:
            return loss

        acc_grad = []
        with torch.no_grad():
            for p in models[0].parameters():
                acc_grad += [p.grad.clone()]
        return acc_grad
