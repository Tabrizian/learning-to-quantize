import numpy as np
import torch
from cuquant import QDQ
import math
from scipy.stats import truncnorm
import scipy.integrate as integrate

EPS = 1e-7

def get_uniform_levels(bits):
    """uniform (QSGD)"""
    num_levels = 2 << bits - 1
    levels_uni = np.linspace(-1, 1, num=num_levels)
    return levels_uni

def get_quantile_levels(bits, mean, sigma):
    """quantile levels """
    num_levels = 2 << bits - 1
    cdf_points = np.linspace(0, 1, num=num_levels)
    levels = [truncnorm.ppf(level, -1, 1, loc=mean, scale=sigma) for level in cdf_points]
    levels[0] = -1
    levels[-1] = 1
    return levels

def get_level(x, levels):
    for index, level in enumerate(levels[0:len(levels) - 1]):
        if x >= levels[index] and x < levels[index + 1]:
            return index

def normal_function(x, mean, sigma):
    f = truncnorm.pdf(x, -1, 1, loc=mean, scale=sigma)
    return f

def calculate_estimated_error(mean, sigma, levels):
    sum = []
    for index, level in enumerate(levels[:-1]):
        def inline_func(x):
            normal_func = normal_function(x, mean, sigma)
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
    for epoch in range(epochs):
        indexes = list(range(len(initial_levels)))[1:-1]
        for index in indexes:
            a = (initial_levels[index - 1] - mean) / (initial_levels[index + 1] - initial_levels[index - 1])
            b = truncnorm.cdf(initial_levels[index + 1], minimum, maximum, loc=mean, scale=sigma) - truncnorm.cdf(initial_levels[index - 1], minimum, maximum, loc=mean, scale=sigma)
            c = (truncnorm.pdf(initial_levels[index + 1], minimum, maximum, loc=mean, scale=sigma) - truncnorm.pdf(initial_levels[index - 1], minimum, maximum, loc=mean, scale=sigma)) / (initial_levels[index + 1] - initial_levels[index - 1])
            initial_levels[index] = truncnorm.ppf(
                truncnorm.cdf(initial_levels[index + 1], minimum, maximum, loc=mean, scale=sigma) + a * b + sigma ** 2 * c, minimum, maximum, loc=mean, scale=sigma
                )

        losses.append(calculate_estimated_error(mean, sigma, initial_levels))
        print('Epoch', epoch, 'error', losses[-1])
        # initial_levels = new_levels
        all_levels.append(initial_levels)
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


def qdqL2(x, levels, bucket_size, in_place):
    """
    Quantize and dequantize with L2 norm.

    x: the input vector (numpy.ndarray)
    levels: either levels_uni for QGSD or levels_exp for NUQSGD
    bucket_size: we split our big vector to multiple buckets of size
                 bucket_size for quantizing

    returns: the array of quanized elements with type (numpy.ndarray)

    you can remove np.asarray at the end and it becomes a
    pythonlist
    """
    assert isinstance(x, torch.cuda.FloatTensor)

    num_tail = math.ceil(x.numel()/bucket_size)*bucket_size-x.numel()
    xv = torch.cat((x.view(-1),
                    torch.zeros(num_tail, dtype=x.dtype, device=x.device)))
    xv = xv.view(-1, bucket_size)
    norm = xv.norm(p='fro', dim=1, keepdim=True).expand(
        xv.shape[0], xv.shape[1]).contiguous().view(-1).contiguous()
    r = torch.randint_like(xv, 1000001).long()
    num_levels = len(levels)
    in_vector = torch.flatten(xv)
    out_vector = torch.zeros_like(in_vector)
    rand_vector = torch.flatten(r)
    j = 0
    for i, val in enumerate(x):
        while (j+1 < num_levels):
            level_up =  levels[j+1]
            if in_vector[i]/(norm[i]+EPS)<=level_up:
                diff = level_up - levels[j]
                if in_vector[i]/(norm[i]+EPS)+diff*(rand_vector[i]%1000001 / 1000000.0)>level_up:
                    j = j+1
                break
            j = j+1			
        out_vector[i] = norm[i]*levels[j]
    out_vector = out_vector[0:x.view(-1).shape[0]]
    out_vector = out_vector.view(x.shape)
    return out_vector


def qdqLinf(x, levels, bucket_size, in_place):
    """
    Quantize and dequantize with L-inf norm.
    """

    assert isinstance(x, torch.cuda.FloatTensor)
    num_levels = levels.numel()
    num_tail = math.ceil(x.numel()/bucket_size)*bucket_size-x.numel()
    xv = torch.cat((x.view(-1),
                    torch.zeros(num_tail, dtype=x.dtype, device=x.device)))
    xv = xv.view(-1, bucket_size)
    norm = xv.norm(p=float('inf'), dim=1, keepdim=True).expand(
        xv.shape[0], xv.shape[1]).contiguous().view(-1).contiguous()
    out_vector = torch.zeros_like(x)
    r = torch.randint_like(x, 1000001).long()
    bucket_size = self.bucket_size

    in_vector = x
    rand_vector = r
    j = 0
    for i, val in enumerate(in_vector):
        while (j+1 < num_levels):
            level_up =  levels[j+1]
            if in_vector[i]/(norm[i]+EPS)<=level_up:
                diff = level_up - levels[j]
                if in_vector[i]/(norm[i]+EPS)+diff*(rand_vector[i]%1000001 / 1000000.0)>level_up:
                    j = j+1
                break
            j = j+1			
        out_vector[i] = norm[i]*levels[j];	 
    return out_vector


class QuantizeNumPy(object):
    def __init__(self, method, bits, bucket_size, **kwargs):
        """
        QSGD: qdqL2 + levels_uni
        NUQSGD: qdqL2 + levels_exp
        QSGD-inf: qdqLinf + levels_uni
        NUQ 2.0: Apadtive Normalized Layers
        """
        if method == 'q':
            self.levels = get_uniform_levels(bits)
            self.qdq = qdqL2
        elif method == 'nuq':
            self.levels = get_exp_levels(bits)
            self.qdq = qdqL2
        elif method == 'qinf':
            self.levels = get_uniform_levels(bits)
            self.qdq = qdqLinf
        elif method == 'nuq2':
            self.levels = get_quantile_levels(bits, 0, 0.1)
            self.qdq = qdqL2
        elif method == 'nuq2inf':
            self.levels = get_quantile_levels(bits, 0, 0.1)
            self.qdq = qdqLinf
        self.number_of_iterations = 0
        self.gradient_samples = []
        self.gradient_samples_overtime = []

        self.bucket_size = bucket_size
        self.bits = bits

    def quantize(self, x, in_place):
        if in_place == True:
            self.number_of_iterations += 1
            self.gradient_samples.append(x.view(-1))
            if self.number_of_iterations % 24 == 0:
                self.gradient_samples_overtime.append(torch.cat(self.gradient_samples))
                self.gradient_samples = []
                if self.number_of_iterations == 24 * 40:
                    self.number_of_iterations = 0
                    mean, variance = self.calculate_mean_variance(self.gradient_samples_overtime)
                    print('Mean is', mean, 'Variance is', variance)
                    initial_levels = get_quantile_levels(self.bits, mean.cpu(), variance.cpu())
                    self.levels, all_levels, losses = get_adaptive_levels_co(initial_levels, len(self.levels), mean.cpu(), variance.cpu(), 10, -1, 1)
                    self.gradient_samples_overtime = []
        return self.qdq(x, self.levels, self.bucket_size, in_place)

    def calculate_mean_variance(self, x):
        sum = torch.zeros_like(x[0])
        for epoch in x:
            sum += epoch
        mean = sum / len(x)

        variance = torch.zeros_like(x[0])
        for epoch in x:
            variance += (epoch - sum) ** 2
        variance = variance / len(x)
        number_of_weights = len(variance)

        variance = torch.sum(variance) / number_of_weights
        mean = torch.sum(mean) / number_of_weights
        return mean, variance
        


class QuantizeSingleBucket(object):
    def __init__(self, method, bits, bucket_size, **kwargs):
        """
        QSGD: qdqL2 + levels_uni
        NUQSGD: qdqL2 + levels_exp
        QSGD-inf: qdqLinf + levels_uni
        """
        self.method = method
        if method == 'q':
            self.levels = get_uniform_levels(bits)
            self.qdq = qdqL2
        elif method == 'nuq':
            self.levels = get_exp_levels(bits)
            self.qdq = qdqL2
        elif method == 'qinf':
            self.levels = get_uniform_levels(bits)
            self.qdq = qdqLinf

        self.bucket_size = bucket_size
        self.bits = bits
        self.levels = torch.as_tensor(self.levels, dtype=torch.float32).cuda()
        self.qdq = QDQ(self.levels)

    def quantize(self, x):
        q = x.clone()
        bucket_size = self.bucket_size
        num_bucket = int(np.ceil(len(x) / bucket_size))
        for bucket_i in range(num_bucket):

            start = bucket_i * bucket_size
            end = min((bucket_i+1) * bucket_size, len(x))
            x_bucket = x[start:end].clone()
            q_bucket = q[start:end].clone()

            norm = x_bucket.norm()
            self.qdq.qdqGPU(x_bucket, float(norm), q_bucket)
            q[start:end] = q_bucket

        return q


class QuantizeMultiBucket(object):
    def __init__(self, method, bits, bucket_size, multiplier, **kwargs):
        """
        QSGD: qdqL2 + levels_uni
        NUQSGD: qdqL2 + levels_exp
        QSGD-inf: qdqLinf + levels_uni
        """
        self.method = method
        if method == 'q':
            self.levels = get_uniform_levels(bits)
            self.norm_type = 'fro'
        elif method == 'nuq':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'qinf':
            self.levels = get_uniform_levels(bits)
            self.norm_type = float('inf')
        elif method == 'none':
            return

        self.bucket_size = bucket_size
        self.bits = bits
        self.levels = torch.as_tensor(self.levels, dtype=torch.float32).cuda()
        self.qdq = QDQ(self.levels)

    def quantize(self, x):
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
