import numpy as np
import torch
from cuquant import QDQ
import math
from estim.dist import TruncNorm, CondNormalTrunc, CondNormalTruncHist

import time
from scipy.stats import truncnorm, norm
import scipy.integrate as integrate

EPS = 1e-7


def calculate_new_error(positive_levels, mean, sigma, min, max):
    sum = []
    trunc_norm = TruncNorm(mean, sigma, min, max)
    for index, level in enumerate(positive_levels[1:-1]):
        def inline_func(x):
            normal_func = trunc_norm.pdf(x)
            index_level = get_level(x, positive_levels)
            variance = (x - positive_levels[index_level]) * \
                (positive_levels[index_level + 1] - x)
            return variance * normal_func
        sum.append(integrate.quad(lambda x: inline_func(
            x), positive_levels[index], positive_levels[index + 1]))

    sum.append(integrate.quad(lambda x: (
        positive_levels[0] ** 2 - x ** 2) * trunc_norm.pdf(x), 0, positive_levels[0]))

    return 2 * np.sum(sum)


def calculate_norm_error(positive_levels, means, sigmas, norms, min, max):
    trunc_norm = CondNormalTrunc(means, sigmas, norms, min, max)
    sum = []
    for index, level in enumerate(positive_levels[:-1]):
        def inline_func(x):
            normal_func = trunc_norm.pdf(x)
            index_level = get_level(x, positive_levels)
            variance = (x - positive_levels[index_level]) * \
                (positive_levels[index_level + 1] - x)
            return variance * normal_func
        sum.append(integrate.fixed_quad(lambda x: inline_func(
            x), positive_levels[index], positive_levels[index + 1], n=40))

    return 2 * np.sum(sum)


def get_uniform_levels(bits):
    """uniform (QSGD)"""
    num_levels = 2 << bits - 1
    levels_uni = np.linspace(-1, 1, num=num_levels)
    return levels_uni


def get_quantile_levels(bits, mean, sigma, min, max):
    """quantile levels """
    trunc_norm = TruncNorm(mean, sigma, min, max)
    num_levels = 2 << bits - 1
    cdf_points = np.linspace(0, 1, num=num_levels)
    levels = [trunc_norm.ppf(level) for level in cdf_points]

    levels[0] = min
    levels[-1] = max
    return levels


def get_uniform_levels(bits):
    """uniform (QSGD)"""
    num_levels = 2 << bits - 1
    levels_uni = np.linspace(-1, 1, num=num_levels)
    return levels_uni


def get_exp_levels(bits, multiplier=0.5):
    """ exponential (NUQSGD)

    multiplier: is used to modify levels_exp based on the number of bits
    """
    num_levels = 2 << bits - 1

    levels = sum([[-multiplier**j for j in range(num_levels >> 1)],
                  [multiplier**j for j in reversed(range(num_levels >> 1))]],
                 [])
    return np.asarray(levels)


def finite_diff_gradient_descent(f, begin, end, x0=None, niters=10, lr=1):
    eps = (end-begin)/1000
    if x0 is None:
        x0 = (begin + end) / 2
    x = x0
    for i in range(niters):
        df = (f(x+eps)-f(x-eps))/(2*eps)
        x -= lr*df
    return x


def bisection(begin, end, f):
    x = (begin + end) / 2
    if (np.abs(f(x) - 0) < 1e-7):
        return x
    both_negative = f(begin) < 0 and f(end) < 0
    both_positive = f(begin) > 0 and f(end) > 0
    if both_negative or both_positive:
        print('Bisection failed')

    x_neg_end_pos = f(x) < 0 and f(end) > 0
    x_pos_end_neg = f(x) > 0 and f(end) < 0
    if x_neg_end_pos or x_pos_end_neg:
        return bisection(x, end, f)
    return bisection(begin, x, f)

def amq_norm_based(initial_point, grad_dist, bits, lr=0.1, epochs=200):
    mul = initial_point
    s = 2 ** (bits - 1) - 1
    all_mul = []
    iter = 0
    for epoch in range(epochs):
        sum = 0.0
        for norm, mean, sigma, coeff in zip(
                grad_dist.norms,
                grad_dist.means,
                grad_dist.sigmas,
                grad_dist.coeff):

            dist_comp = TruncNorm(
                mean, sigma, grad_dist.begin, grad_dist.end, grad_dist.nbins)

            # from eq G.3 in Appendix
            def arg1_1(j):
                return mean * (j * mul ** (j - 1) + (j + 1) * mul ** j) \
                    - (2 * j + 1) * mul ** (2 * j)
            arg1 = np.sum(np.asarray(
                [arg1_1(j)*(dist_comp.cdf(mul**j) - dist_comp.cdf(mul**(j+1)))
                    for j in range(0, s)]))

            def arg2_1(j):
                return j * mul ** (j - 1) + (j + 1) * mul ** j
            arg2 = np.sum(np.asarray(
                [arg2_1(j) * (dist_comp.pdf(mul ** (j + 1))
                              - dist_comp.pdf(mul ** (j)))
                    for j in range(0, s)]))
            sum += coeff * (arg1 + sigma ** 2 * arg2)

        gradient = 2 * s * (mul ** (2 * s - 1)) * \
            (grad_dist.cdf(mul ** s) - grad_dist.cdf(0)) + sum
        mul = mul - lr * gradient
        iter += 1
        all_mul.append(mul)
    return mul, all_mul


def amq_norm_less(initial_point, grad_dist, bits, lr=0.1, epochs=200):
    mul = initial_point
    s = 2 ** (bits - 1) - 1
    mean = grad_dist.mean
    sigma = grad_dist.sigma
    all_mul = []
    iter = 0
    for epoch in range(epochs):
        sum = 0.0

        def arg1_1(j):
            return mean * (j * mul ** (j - 1) + (j + 1) * mul ** j) \
                - (2 * j + 1) * mul ** (2 * j)
        arg1 = np.sum(np.asarray([arg1_1(j) * (
            grad_dist.cdf(mul ** j) -
            grad_dist.cdf(mul ** (j+1))) for j in range(0, s)]))

        def arg2_1(j):
            return j * mul ** (j - 1) + (j + 1) * mul ** j
        arg2 = np.sum(np.asarray([
            arg2_1(j) * (grad_dist.pdf(mul ** (j + 1)) -
                         grad_dist.pdf(mul ** (j))) for j in range(0, s)]))

        gradient = 2 * s * (mul ** (2 * s - 1)) * \
            (grad_dist.cdf(mul ** s) - grad_dist.cdf(0)) \
            + arg1 + sigma ** 2 * arg2

        mul = mul - lr * gradient
        iter += 1
        all_mul.append(mul)

    return mul, all_mul


def alq_sym(initial_levels, grad_dist, epochs):
    # symmetric alq norm-based
    positive_levels = initial_levels[len(initial_levels) // 2:]
    losses = []
    # Assuming last level is 1, setting first dummy level to 0
    new_levels = [0] + list(positive_levels).copy()
    all_levels = [new_levels.copy()]
    for epoch in range(epochs):

        def objective(x, left_level, right_level):
            # from equation below corollary 1
            left_var = grad_dist.est_var_adjacent_levels(left_level, x)
            right_var = grad_dist.est_var_adjacent_levels(x, right_level)
            return left_var+right_var

        for index in range(1, len(new_levels)-1):
            left_level = new_levels[index - 1]
            right_level = new_levels[index + 1]
            new_levels[index] = finite_diff_gradient_descent(
                lambda x: objective(x, left_level, right_level),
                left_level, right_level, x0=new_levels[index])
        losses.append(grad_dist.estimate_variance(new_levels))
        all_levels.append(new_levels.copy())
    # dropping dummy level at 0
    new_levels = new_levels[1:]
    negative_levels = [-level for level in new_levels]
    negative_levels.reverse()
    new_levels = negative_levels + new_levels
    return new_levels, all_levels, losses


def alq_asym(initial_levels, grad_dist, epochs):
    # asymmetric alq norm-based
    # TODO I need to test it
    losses = []
    new_levels = list(initial_levels).copy()
    all_levels = [new_levels.copy()]
    for epoch in range(epochs):
        for index in range(1, len(new_levels)-1):
            left_level = new_levels[index - 1]
            right_level = new_levels[index + 1]
            new_levels[index] = grad_dist.estimate_variance_adj_inv(
                left_level, right_level)

        losses.append(grad_dist.estimate_variance(new_levels))
        all_levels.append(new_levels.copy())
    return new_levels, all_levels, losses

def get_exp_levels(bits, multiplier):
    """ exponential (NUQSGD)

    multiplier: is used to modify levels_exp based on the number of bits
    """
    num_levels = 2 << bits - 1

    # if bits == 2:
    #     multiplier = 0.1
    # elif bits == 4:
    #     multiplier = 0.5
    # elif bits == 6:
    #     multiplier = 0.9
    # elif bits == 8:
    #     multiplier = 0.95
    levels = sum([[-multiplier**j for j in range(num_levels >> 1)],
                  [multiplier**j for j in reversed(range(num_levels >> 1))]],
                 [])
    return levels


def get_exp_levels(bits, multiplier):
    """ exponential (NUQSGD)

    multiplier: is used to modify levels_exp based on the number of bits
    """
    num_levels = 2 << bits - 1

    # if bits == 2:
    #     multiplier = 0.1
    # elif bits == 4:
    #     multiplier = 0.5
    # elif bits == 6:
    #     multiplier = 0.9
    # elif bits == 8:
    #     multiplier = 0.95
    levels = sum([[-multiplier**j for j in range(num_levels >> 1)],
                  [multiplier**j for j in reversed(range(num_levels >> 1))]],
                 [])
    return levels


class QuantizeMultiBucket(object):
    def __init__(self, method, bits, bucket_size, multiplier, **kwargs):
        """
        QSGD: qdqL2 + levels_uni
        NUQSGD: qdqL2 + levels_exp
        QSGD-inf: qdqLinf + levels_uni
        """
        self.method = method
        self.multiplier = multiplier
        if kwargs['interval'] != None:
            self.interval = kwargs['interval']
            a, b = (-self.interval - 0) / 0.1, (self.interval - 0) / 0.1
        if method == 'q':
            self.levels = get_uniform_levels(bits)
            self.norm_type = 'fro'
        elif method == 'nuq':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'qinf':
            self.levels = get_uniform_levels(bits)
            self.norm_type = float('inf')
        elif method == 'nuq2':
            self.levels = get_quantile_levels(
                bits, 0, 0.1, -self.interval, self.interval)
            self.norm_type = 'fro'
        elif method == 'nuq2inf':
            self.levels = get_quantile_levels(
                bits, 0, 0.1, -self.interval, self.interval)
            self.norm_type = float('inf')
        elif method == 'amq':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'amq_nb':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'nuq4':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'alq':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'alq_nb':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'none':
            return

        self.number_of_iterations = 0
        self.gradient_samples = []
        self.gradient_samples_overtime = []
        self.previous_best = None

        self.bucket_size = bucket_size
        self.bits = bits
        self.epochs = kwargs['cd_epochs']
        self.path = kwargs['path']
        self.symmetric = kwargs['symmetric']
        self.levels = torch.as_tensor(self.levels, dtype=torch.float32).cuda()
        self.qdq = QDQ(self.levels)
        self.mean = 0
        self.variance = 0.1
        self.error = None

    def set_mean_variance(self, mean, variance, norms):
        self.mean = mean
        self.variance = variance
        self.norms = norms
        self.number_of_iterations += 1
        interval = self.interval
        sigma = torch.sqrt(torch.tensor(variance)).cpu().item()
        self.grad_dist_nb = CondNormalTruncHist(
            norms['mean'], norms['sigma'], norms['norm'], -interval, interval, nbins=100000, bin_type='linear')
        self.grad_dist_nl = TruncNorm(
            mean, sigma, -interval, interval, nbins=100000, bin_type='linear')

        self.error = self.grad_dist_nb.estimate_variance(self.levels.cpu())
        if self.method == 'amq':
            np.savetxt(self.path + '/norms_mean' +
                       str(self.number_of_iterations), np.asarray(self.norms['mean']))
            np.savetxt(self.path + '/norms_sigma' +
                       str(self.number_of_iterations), np.asarray(self.norms['sigma']))
            np.savetxt(self.path + '/norms_norm' +
                       str(self.number_of_iterations), np.asarray(self.norms['norm']))

    def update_levels(self):
        interval = self.interval
        mean = self.mean
        variance = self.variance
        grad_dist_nl = self.grad_dist_nl
        grad_dist_nb = self.grad_dist_nb
        sigma = torch.sqrt(torch.tensor(self.variance)).cpu().item()
        half_point = int(len(self.levels) / 2)
        quantile_levels = get_quantile_levels(
            self.bits, mean, sigma, -interval, interval)
        uniform_levels = get_uniform_levels(
            self.bits)
        exp_levels = get_exp_levels(
            self.bits, 0.5)

        bits = self.bits
        if self.method == 'alq':
            epochs = self.epochs
            initial_levels = self.levels

            levels_qua, _, losses_qua = alq_sym(quantile_levels, grad_dist_nl, epochs)
            levels_uniform, _, losses_uni = alq_sym(uniform_levels, grad_dist_nl, epochs)
            levels_exp, _, losses_exp = alq_sym(exp_levels, grad_dist_nl, epochs)
            candidate_levels = np.asarray([levels_qua, levels_uniform, levels_exp])
            candidate_losses = np.asarray([losses_qua[-1], losses_uni[-1], losses_exp[-1]])
            self.levels = candidate_levels[np.argsort(candidate_losses)][0]

        elif self.method == 'alq_nb':
            epochs = self.epochs
            self.levels = get_quantile_levels(
                self.bits, mean, sigma, -interval, interval)
            initial_levels = self.levels
            if self.symmetric:
                qua_levels, all_levels, qua_losses = alq_sym(initial_levels, grad_dist_nb, epochs)
                levels_qua, _, losses_qua = alq_sym(quantile_levels, grad_dist_nb, epochs)
                levels_uniform, _, losses_uni = alq_sym(uniform_levels, grad_dist_nb, epochs)
                levels_exp, _, losses_exp = alq_sym(exp_levels, grad_dist_nb, epochs)
                candidate_levels = np.asarray([levels_qua, levels_uniform, levels_exp])
                candidate_losses = np.asarray([losses_qua[-1], losses_uni[-1], losses_exp[-1]])
                self.levels = candidate_levels[np.argsort(candidate_losses)][0]
            else: 
                qua_levels, all_levels, qua_losses = alq_asym(initial_levels, grad_dist_nb, epochs)
                levels_qua, _, losses_qua = alq_asym(quantile_levels, grad_dist_nb, epochs)
                levels_uniform, _, losses_uni = alq_asym(uniform_levels, grad_dist_nb, epochs)
                levels_exp, _, losses_exp = alq_asym(exp_levels, grad_dist_nb, epochs)
                candidate_levels = np.asarray([levels_qua, levels_uniform, levels_exp])
                candidate_losses = np.asarray([losses_qua[-1], losses_uni[-1], losses_exp[-1]])
                self.levels = candidate_levels[np.argsort(candidate_losses)][0]

        elif self.method == 'amq':
            initial_points = []

            if self.previous_best is None:
                initial_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9]
            else:
                initial_points = [0.1, 0.2, 0.3, 0.4,
                                  self.previous_best,  0.8, 0.9]
            optimal_points = []
            for point in initial_points:
                optimal_p, _ = amq_norm_less(point, grad_dist_nl, self.bits)
                optimal_points.append(optimal_p)
            optimal_points_costs = [
                grad_dist_nl.estimate_variance(get_exp_levels(self.bits, p)[
                                                        half_point:]) for p in optimal_points]
            index = np.argmin(optimal_points_costs)
            self.multiplier = optimal_points[index]
            self.previous_best = self.multiplier
            self.levels = get_exp_levels(self.bits, self.multiplier)

        elif self.method == 'amq_nb':
            initial_points = []

            if self.previous_best is None:
                initial_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9]
            else:
                initial_points = [0.1, 0.2, 0.3, 0.4,
                                  self.previous_best,  0.8, 0.9]
            optimal_points = []
            for point in initial_points:
                optimal_p, _ = amq_norm_based(point, grad_dist_nb, bits)
                optimal_points.append(optimal_p)
            optimal_points_costs = [
                grad_dist_nb.estimate_variance(get_exp_levels(self.bits, p)[
                                                        half_point:]) for p in optimal_points]
            index = np.argmin(optimal_points_costs)
            self.multiplier = optimal_points[index]
            self.previous_best = self.multiplier
            self.levels = get_exp_levels(self.bits, self.multiplier)
        
        self.levels = torch.as_tensor(self.levels, dtype=torch.float32).cuda()
        self.qdq = QDQ(self.levels)

    def quantize(self, x, number_of_layers):
        if self.method == 'none':
            return x
        assert isinstance(x, torch.cuda.FloatTensor)
        bucket_size = self.bucket_size

        num_tail = math.ceil(x.numel()/bucket_size)*bucket_size-x.numel()
        xv = torch.cat((x.view(-1),
                        torch.zeros(num_tail, dtype=x.dtype, device=x.device)))
        xv = xv.view(-1, bucket_size)
        norm = xv.norm(p=self.norm_type, dim=1, keepdim=True).expand(
            xv.shape[0], xv.shape[1]).contiguous().view(-1).contiguous()
        q = torch.zeros_like(x)
        r = torch.randint_like(x, 1000001).long()

        self.qdq.qdqGPU(x, norm, q, r)

        return q
