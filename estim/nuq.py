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

    def state_dict(self):
        return {
            'qdq': self.qdq.state_dict()
        }

    def load_state_dict(self, state):
        print(state)
        self.qdq.load_state_dict(state['qdq'])

    def grad(self, model_new, in_place=False):
        model = model_new
        ig_sm_bkts = self.opt.nuq_ig_sm_bkts

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

            per_layer = not self.opt.nuq_layer
            with torch.no_grad():
                if not per_layer:
                    flatt_grad = self._flatten(grad)
                    flatt_grad_q = self.qdq.quantize(flatt_grad, ig_sm_bkts)
                    grad_like_q = self.unflatten(flatt_grad_q, grad)
                    for g, a in zip(grad_like_q, self.acc_grad):
                        a += g / self.ngpu

                else:
                    for g, a in zip(grad, self.acc_grad):
                        a += self.qdq.quantize(g, ig_sm_bkts) / self.ngpu

        if in_place:
            for p, a in zip(model.parameters(), self.acc_grad):
                if p.grad is None:
                    p.grad = a.clone()
                else:
                    p.grad.copy_(a)
            return loss
        return self.acc_grad
