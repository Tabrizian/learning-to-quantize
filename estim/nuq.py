import torch
import torch.nn
import torch.multiprocessing

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

    def snap_online(self, model):
        total_variance = 0
        iterations = self.opt.nuq_number_of_samples 
        mean = []
        with torch.no_grad():
            for p in model.parameters():
                mean += [torch.zeros_like(p)]

        for i in range(iterations):
            acc_grad = None
            if acc_grad is None:
                acc_grad = []
                with torch.no_grad():
                    for p in model.parameters():
                        acc_grad += [torch.zeros_like(p)]
            else:
                for a in acc_grad:
                    a.zero_()

            model.zero_grad()
            data = next(self.data_iter)
            loss = model.criterion(model, data)
            grad = torch.autograd.grad(loss, model.parameters())
            layers = len(list(model.parameters()))

            with torch.no_grad():
                for g, a in zip(grad, mean):
                    # if len(a.shape) != 1:
                    a += g
        
        nw = sum([w.numel() for w in model.parameters()])
        for i, a in enumerate(mean):
            mean[i] /= iterations
        
        for i in range(iterations):
            variance = 0.0
            acc_grad = None
            if acc_grad is None:
                acc_grad = []
                with torch.no_grad():
                    for p in model.parameters():
                        acc_grad += [torch.zeros_like(p)]
            else:
                for a in acc_grad:
                    a.zero_()

            model.zero_grad()
            data = next(self.data_iter)
            loss = model.criterion(model, data)
            grad = torch.autograd.grad(loss, model.parameters())
            layers = len(list(model.parameters()))

            with torch.no_grad():
                for ee, gg in zip(mean, grad):
                    # if len(ee.shape) != 1:
                    variance += (gg-ee).pow(2).sum()
                total_variance += variance / nw
        total_variance /= iterations
        total_mean = sum([item.sum() for item in mean])
        
        return total_mean, total_variance
        

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

            with torch.no_grad():
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


class NUQEstimatorSingleGPUParallel(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(NUQEstimatorSingleGPUParallel, self).__init__(*args, **kwargs)
        self.init_data_iter()
        nuq_kwargs = opt_to_nuq_kwargs(self.opt)
        self.ngpu = self.opt.nuq_ngpu
        self.acc_grad = None
        self.models = None
        self.qdq = QuantizeMultiBucket(**nuq_kwargs)

    def grad(self, model_new, in_place=False):
        if self.models is None:
            self.models = [model_new]
            for i in range(1, self.ngpu):
                self.models += [copy.deepcopy(model_new)]

        models = self.models

        # forward prop
        loss = []
        for i in range(self.ngpu):
            models[i].zero_grad()
            data = next(self.data_iter)

            loss += [models[i].criterion(models[i], data)]

        # backward prop
        for i in range(self.ngpu):
            loss[i].backward()

        loss = loss[-1]

        layers = len(list(model.parameters()))
        # quantize all grads
        for i in range(self.ngpu):
            with torch.no_grad():
                for p in models[i].parameters():
                    p.grad.copy_(self.qdq.quantize(p.grad, layers)/self.ngpu)

        # aggregate grads into gpu0
        for i in range(1, self.ngpu):
            for p0, pi in zip(models[0].parameters(),
                              models[i].parameters()):
                p0.grad.add_(pi.grad)

        if in_place:
            return loss

        acc_grad = []
        with torch.no_grad():
            for p in models[0].parameters():
                acc_grad += [p.grad.clone()]
        return acc_grad


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

        layers = len(list(model.parameters()))
        # quantize all grads
        for i in range(self.ngpu):
            with torch.no_grad():
                with torch.cuda.device(i):
                    torch.cuda.synchronize()
                    for p in models[i].parameters():
                        p.grad.copy_(self.qdq[i].quantize(p.grad, layers) / self.ngpu)

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
