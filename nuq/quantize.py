import numpy as np
import torch
from cuquant import QDQ
import math
from estim.dist import Normal, TruncNorm, CondNormal, CondNormalTrunc

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
            variance = (x - positive_levels[index_level]) * (positive_levels[index_level + 1] - x)
            return variance * normal_func
        sum.append(integrate.quad(lambda x: inline_func(x), positive_levels[index], positive_levels[index + 1]))

    sum.append(integrate.quad(lambda x: (positive_levels[0] ** 2 - x ** 2) * trunc_norm.pdf(x), 0, positive_levels[0]))
    
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
    cdf_points = np.linspace(0, 1, num=num_levels - 2)
    levels = [trunc_norm.ppf(level) for level in cdf_points]
    
    levels = [min] + levels + [max]
    return levels

def get_level(x, levels):
    for index, level in enumerate(levels[0:len(levels) - 1]):
        if x >= levels[index] and x < levels[index + 1]:
            return index

def truncated_cdf(x, minimum, maximum, loc, scale):
    a = norm.cdf(x, loc=loc, scale=scale) - norm.cdf(minimum, loc=loc, scale=scale)
    b = norm.cdf(maximum, loc=loc, scale=scale) - norm.cdf(minimum, loc=loc, scale=scale)
    return a / b

def calculate_estimated_error(mean, sigma, levels, min, max):
    sum = []
    trunc_norm = TruncNorm(mean, sigma, min, max)
    for index, level in enumerate(levels[:-1]):
        def inline_func(x):
            normal_func = trunc_norm.pdf(x)
            index_level = get_level(x, levels)
            variance = (x - levels[index_level]) * (levels[index_level + 1] - x)
            return variance * normal_func

        sum.append(integrate.quad(lambda x: inline_func(x), levels[index], levels[index + 1]))

    return np.sum(sum)

def get_adaptive_levels_co(initial_levels, number_of_levels, mean, sigma, epochs, minimum, maximum):
    losses = []
    new_levels = np.zeros_like(initial_levels)
    new_levels[0] = initial_levels[0]
    new_levels[-1] = initial_levels[-1]
    all_levels = [initial_levels]
    mean = mean.item()
    sigma = sigma.item()
    minimum = minimum.item()
    maximum = maximum.item()
    trunc_norm = TruncNorm(mean, sigma, minimum, maximum)
    for epoch in range(epochs):
        indexes = list(range(len(initial_levels)))[1:-1]
        for index in indexes:
            a = (initial_levels[index - 1] - mean) / (initial_levels[index + 1] - initial_levels[index - 1])
            b = trunc_norm.cdf(initial_levels[index + 1]) - trunc_norm.cdf(initial_levels[index - 1])
            c = (trunc_norm.pdf(initial_levels[index + 1]) - trunc_norm.pdf(initial_levels[index - 1])) / (initial_levels[index + 1] - initial_levels[index - 1])
            initial_levels[index] = trunc_norm.ppf(
                trunc_norm.cdf(initial_levels[index + 1]) + a * b + sigma ** 2 * c
                )

        losses.append(calculate_estimated_error(initial_levels,mean, sigma,  minimum, maximum))
        print('Epoch', epoch, 'error', losses[-1])
        # initial_levels = new_levels
        all_levels.append(initial_levels)
    return initial_levels, all_levels, losses

def bisect_find_point(l_2, mean, sigma, minimum, maximum, begin, end):
    trunc_norm = TruncNorm(mean, sigma, minimum, maximum)
    x = (begin + end) / 2
    def objective(x):
        part_1 = (l_2 - mean) * (trunc_norm.cdf(l_2) - trunc_norm.cdf(x)) + 2 * x * (trunc_norm.cdf(0) - trunc_norm.cdf(x))
        part_2 = sigma ** 2 * (trunc_norm.pdf(l_2) - trunc_norm.pdf(x))
        return part_1 + part_2
    

    if (objective(begin) > 0 and objective(end)  > 0) or (objective(end) < 0 and objective(begin) < 0):
        print('Whhhattt', 'How can I use bisection!!!!')

    if (np.abs(objective(x) - 0) < 0.00001):
        return x
    if (objective(x) <0 and objective(end) > 0) or (objective(x) > 0 and objective(end) < 0):
        return bisect_find_point(l_2, mean, sigma, minimum, maximum, x, end)
    elif (objective(x) <0 and objective(begin) > 0) or (objective(x) > 0 and objective(begin) < 0):
        return bisect_find_point(l_2, mean, sigma, minimum, maximum, begin, x)

    
def bisect_find_point_2(level_1, level_2, norms, a, b, begin, end):
    trunc_norm = TruncNorm(mean, sigma, minimum, maximum)
    x = (begin + end) / 2
    def objective(x):
        part_1 = (l_2 - mean) * (trunc_norm.cdf(l_2) - trunc_norm.cdf(x)) + 2 * x * (trunc_norm.cdf(0) - trunc_norm.cdf(x))
        part_2 = sigma ** 2 * (trunc_norm.pdf(l_2) - trunc_norm.pdf(x))
        return part_1 + part_2
    

    if (objective(begin) > 0 and objective(end)  > 0) or (objective(end) < 0 and objective(begin) < 0):
        print('Whhhattt', 'How can I use bisection!!!!')

    if (np.abs(objective(x) - 0) < 0.00001):
        return x
    if (objective(x) <0 and objective(end) > 0) or (objective(x) > 0 and objective(end) < 0):
        return bisect_find_point(l_2, mean, sigma, minimum, maximum, x, end)
    elif (objective(x) <0 and objective(begin) > 0) or (objective(x) > 0 and objective(begin) < 0):
        return bisect_find_point(l_2, mean, sigma, minimum, maximum, begin, x)

def get_adaptive_levels_co_2(initial_levels, number_of_levels, mean, sigma, epochs, minimum, maximum):
    losses = []
    new_levels = np.zeros_like(initial_levels)
    new_levels[0] = initial_levels[0]
    new_levels[-1] = initial_levels[-1]
    all_levels = [initial_levels]
    mean = mean.item()
    sigma = sigma.item()
    minimum = minimum.item()
    maximum = maximum.item()
    trunc_norm = TruncNorm(mean, sigma, minimum, maximum)
    for epoch in range(epochs):
        indexes = list(range(len(initial_levels)))[1:-1]
        initial_levels[0] = bisect_find_point(initial_levels[1], mean, sigma, minimum, maximum, 0, initial_levels[1])
        for index in indexes:
            if index == 1:
                continue
            a = (initial_levels[index - 1] - mean) / (initial_levels[index + 1] - initial_levels[index - 1])
            b = trunc_norm.cdf(initial_levels[index + 1]) - trunc_norm.cdf(initial_levels[index - 1])
            c = (trunc_norm.pdf(initial_levels[index + 1]) - trunc_norm.pdf(initial_levels[index - 1])) / (initial_levels[index + 1] - initial_levels[index - 1])
            initial_levels[index] = trunc_norm.ppf(
                trunc_norm.cdf(initial_levels[index + 1]) + a * b + sigma ** 2 * c
                )

        losses.append(calculate_new_error(initial_levels, mean, sigma,  minimum, maximum))
        print('Epoch', epoch, 'error', losses[-1])
        # initial_levels = new_levels
        all_levels.append(initial_levels)
    negative_levels = [-level for level in initial_levels]
    negative_levels.reverse()
    initial_levels = negative_levels + initial_levels
    return initial_levels, all_levels, losses


def get_adaptive_levels_co_2(initial_levels, number_of_levels, mean, sigma, epochs, minimum, maximum):
    losses = []
    new_levels = np.zeros_like(initial_levels)
    new_levels[0] = initial_levels[0]
    new_levels[-1] = initial_levels[-1]
    all_levels = [initial_levels]
    mean = mean.item()
    sigma = sigma.item()
    minimum = minimum.item()
    maximum = maximum.item()
    trunc_norm = TruncNorm(mean, sigma, minimum, maximum)
    for epoch in range(epochs):
        indexes = list(range(len(initial_levels)))[1:-1]
        initial_levels[0] = bisect_find_point(initial_levels[1], mean, sigma, minimum, maximum, 0, initial_levels[1])
        for index in indexes:
            if index == 1:
                continue
            a = (initial_levels[index - 1] - mean) / (initial_levels[index + 1] - initial_levels[index - 1])
            b = trunc_norm.cdf(initial_levels[index + 1]) - trunc_norm.cdf(initial_levels[index - 1])
            c = (trunc_norm.pdf(initial_levels[index + 1]) - trunc_norm.pdf(initial_levels[index - 1])) / (initial_levels[index + 1] - initial_levels[index - 1])
            initial_levels[index] = trunc_norm.ppf(
                trunc_norm.cdf(initial_levels[index + 1]) + a * b + sigma ** 2 * c
                )

        losses.append(calculate_new_error(initial_levels, mean, sigma,  minimum, maximum))
        print('Epoch', epoch, 'error', losses[-1])
        # initial_levels = new_levels
        all_levels.append(initial_levels)
    negative_levels = [-level for level in initial_levels]
    negative_levels.reverse()
    initial_levels = negative_levels + initial_levels
    return initial_levels, all_levels, losses

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
            self.levels = get_quantile_levels(bits, 0, 0.1, -self.interval, self.interval)
            self.norm_type = 'fro'
        elif method == 'nuq2inf':
            self.levels = get_quantile_levels(bits, 0, 0.1, -self.interval, self.interval)
            self.norm_type = float('inf')
        elif method == 'nuq3':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'nuq4':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'nuq5':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'none':
            return

        self.number_of_iterations = 0
        self.gradient_samples = []
        self.gradient_samples_overtime = []

        self.bucket_size = bucket_size
        self.bits = bits
        self.co_epochs = kwargs['cd_epochs']
        self.levels = torch.as_tensor(self.levels, dtype=torch.float32).cuda()
        self.qdq = QDQ(self.levels)
        self.mean = 0
        self.variance = 0.1
        self.error = None
    

    def exponential_gd(self, initial_point, learning_rate=0.7, epochs=10000):
        mean = self.mean
        sigma = torch.sqrt(torch.tensor(self.variance)).cpu().item()
        p = initial_point
        s = 2 ** (self.bits - 1) - 1 
        trunc_norm = TruncNorm(mean, sigma, -self.interval, self.interval)
        i = 0
        while True:
            def arg1_1(j):
                return mean * (j * p ** (j - 1) + (j + 1) * p ** j) - (2 * j + 1) * p ** (2 * j)
            arg1 = torch.sum(torch.tensor([ arg1_1(j) * (trunc_norm.cdf(p ** j) - trunc_norm.cdf(p ** (j+1))) for j in range(0, s)]))
            def arg2_1(j):
                return j * p ** (j - 1) + (j + 1) * p ** j
            arg2 = torch.sum(torch.tensor([arg2_1(j) * (trunc_norm.pdf(p ** (j + 1)) - trunc_norm.pdf(p ** (j))) for j in range(0, s)]))
            gradient = 2 * s * (p ** (2 * s - 1)) * (trunc_norm.cdf(p ** s) - trunc_norm.cdf(0)) + arg1 + sigma ** 2 * arg2
            if i == 200:
                break
            p = p - learning_rate * gradient
            print('Multiplier value is', p, 'Epoch is ', i, 'gradient', gradient)
            i += 1
        return p

    def set_mean_variance(self, mean, variance, norms):
        self.mean = mean
        self.variance = variance
        self.norms = norms
        print('Current mean is', mean, 'current variance is', variance)
        sigma = torch.sqrt(torch.tensor(variance)).cpu().item()
        half_point = int(len(self.levels) / 2)
        half_levels = self.levels[half_point:].cpu()
        self.error = calculate_new_error(half_levels, self.mean, sigma, -self.interval, self.interval)
        print('Error for CO is ', self.error)


    def update_levels(self):
        interval = self.interval
        mean = self.mean
        variance = self.variance
        sigma = torch.sqrt(torch.tensor(self.variance)).cpu().item()
        if self.method == 'nuq2':
            self.levels = get_quantile_levels(self.bits, mean, sigma, -interval, interval)
            initial_levels = self.levels
            self.levels, all_levels, losses = get_adaptive_levels_co(initial_levels, len(self.levels), mean, sigma, self.co_epochs, -interval, interval)
        elif self.method == 'nuq5':
            self.levels = get_quantile_levels(self.bits, mean, sigma, -interval, interval)
            half_point = int(len(self.levels) / 2)
            initial_levels = self.levels
            self.levels, all_levels, losses = get_adaptive_levels_co_2(initial_levels[half_point:], len(self.levels) / 2, mean, sigma, self.co_epochs, -interval, interval)
        elif self.method =='nuq6':
            self.levels = get_quantile_levels(self.bits, mean, sigma, -self.interval, self.interval)
            half_point = int(len(self.levels) / 2)
            initial_levels = self.levels
            self.levels, all_levels, losses = get_adaptive_levels_co_3(initial_levels[half_point:], len(self.levels) / 2, mean, sigma, self.co_epochs, -interval, interval)
            print('Levels are', self.levels)
        elif self.method == 'nuq3':
            self.previous_best = None
            initial_points = []

            if self.previous_best == None:
                initial_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9]
            else:
                initial_points = [0.1, 0.2, 0.3, 0.4, self.previous_best,  0.8, 0.9]
            optimal_points = []
            for point in initial_points:
                optimal_p = self.exponential_gd(point)
                optimal_points.append(optimal_p)
            half_point = 2 ** (self.bits - 1)
            optimal_points_costs = [calculate_new_error(get_exp_levels(self.bits, p)[half_point:], mean, sigma, -interval, interval) for p in optimal_points]
            index = np.argmin(optimal_points_costs)
            self.multiplier = optimal_points[index]
            self.previous_best = self.multiplier
            self.levels = get_exp_levels(self.bits, self.multiplier)

        elif self.method == 'nuq4':
            self.previous_best = None
    
            initial_points = []

            if self.previous_best == None:
                initial_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9]
            else:
                initial_points = [0.1, 0.2, 0.3, 0.4, self.previous_best,  0.8, 0.9]
            optimal_points = []
            for point in initial_points:
                optimal_p = self.exponential_gd(point)
                optimal_points.append(optimal_p)
            half_point = 2 ** (self.bits - 1)
            optimal_points_costs = [calculate_new_error(get_exp_levels(self.bits, p)[half_point:], mean, sigma, -interval, interval) for p in optimal_points]
            index = np.argmin(optimal_points_costs)
            self.multiplier = optimal_points[index]
            self.previous_best = self.multiplier
            self.levels = get_exp_levels(self.bits, self.multiplier)
            print('Levels are', self.levels)

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
