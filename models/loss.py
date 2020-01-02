import torch
import torch.nn.functional as F

from cusvd import svdj


class Loss(object):
    def __call__(self, model, data,
                 reduction='mean', weights=1, return_output=False):
        data, target = data[0].cuda(), data[1].cuda()
        model.zero_grad()
        output = model(data)
        loss = self.loss(output, target, reduction=reduction)*weights
        if return_output:
            return loss, output
        return loss

    def loss_sample(self, output, target, empirical=False):
        with torch.no_grad():
            if empirical:
                sampled_y = target
            else:
                with torch.no_grad():
                    sampled_y = self._sample_y(output)
        loss_sample = self.loss(output, sampled_y, reduction='none')
        return loss_sample

    def E_f(self, f, output, n_samples=-1):
        if n_samples == -1:
            return self.fisher_exact(output)

    def fisher(self, output, n_samples=-1):
        if n_samples == -1:
            return self.fisher_exact(output)
        F_L = 0
        output = output.detach()
        output.requires_grad = True
        for i in range(n_samples):
            with torch.no_grad():
                sampled_y = self._sample_y(output)
            loss_sample = self.loss(output, sampled_y, reduction='none')
            loss_sample = loss_sample.sum()
            ograd = torch.autograd.grad(loss_sample, [output])[0]
            F_L += torch.einsum('bo,bp->bop', ograd, ograd)
        F_L /= n_samples
        return F_L, self.QL_from_FL(F_L)

    def QL_from_FL(self, F_L):
        Q_L = []
        eps = 1e-7
        for i in range(F_L.shape[0]):
            U, S, V = svdj(F_L[i] + F_L.new_ones(F_L[i].shape[0]).diag()*eps)
            assert all(S > 0), 'S has negative elements'
            Q_L += [U @ S.sqrt().diag()]
            # Q_L += [torch.eye(s.shape[1], dtype=s.dtype, device=s.device)]
        Q_L = torch.stack(Q_L)
        return Q_L


class CELoss(Loss):
    def __init__(self):
        self.loss = F.cross_entropy
        self.do_accuracy = True

    def _sample_y(self, output):
        probs = F.softmax(output, dim=-1)
        sampled_y = torch.multinomial(probs, 1).squeeze(-1)
        return sampled_y

    def fisher_exact(self, s):
        # TODO: brute-force fisher convergence compared to SGD
        sisj = torch.einsum('bo,bp->bop', s, s)
        d = torch.einsum('bo,op->bop', s, s.new_ones(s.shape[1]).diag())
        F_L = d - sisj
        # TODO: analytical expression for Q_L
        return F_L, self.QL_from_FL(F_L)


class MSELoss(Loss):
    def __init__(self):
        # self.loss = F.mse_loss
        self.do_accuracy = False

    def loss(self, *args, **kwargs):
        reduction = 'mean'
        if 'reduction' in kwargs:
            reduction = kwargs['reduction']
        kwargs['reduction'] = 'none'

        ret = 0.5 * F.mse_loss(*args, **kwargs)
        if reduction == 'mean':
            return ret.sum(1).mean(0)  # ret.mean(1).mean(0)
        return ret.sum(1)  # ret.mean(1)

    def _sample_y(self, output):
        return torch.normal(output, torch.ones_like(output))

    def fisher_exact(self, output):
        F_L = torch.eye(output.shape[1], dtype=output.dtype,
                        device=output.device)
        F_L = F_L.unsqueeze(0).expand(output.shape + (output.shape[1],))
        Q_L = F_L
        return F_L, Q_L
