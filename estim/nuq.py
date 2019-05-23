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

            with torch.no_grad():
                for g, a in zip(grad, self.acc_grad):
                    a += self.qdq.quantize(g)/self.ngpu

        if in_place:
            for p, a in zip(model.parameters(), self.acc_grad):
                if p.grad is None:
                    p.grad = a.clone()
                else:
                    p.grad.copy_(a)
            return loss
        return self.acc_grad
