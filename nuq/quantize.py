import numpy as np
import torch
from cuquant import QDQ
import math

from scipy.stats import truncnorm, norm
import scipy.integrate as integrate

EPS = 1e-7
 
def calculate_new_error(positive_levels, mean, sigma, min, max):
    sum = []
    for index, level in enumerate(positive_levels[1:-1]):
        def inline_func(x):
            normal_func = normal_function(x, mean, sigma, min, max)
            index_level = get_level(x, positive_levels)
            variance = (x - positive_levels[index_level]) * (positive_levels[index_level + 1] - x)
            return variance * normal_func
        sum.append(integrate.quad(lambda x: inline_func(x), positive_levels[index], positive_levels[index + 1]))

    sum.append(integrate.quad(lambda x: (positive_levels[0] ** 2 - x ** 2) * normal_function(x, mean, sigma, min, max), 0, positive_levels[0]))
    

    return 2 * np.sum(sum)

def get_uniform_levels(bits):
    """uniform (QSGD)"""
    num_levels = 2 << bits - 1
    levels_uni = np.linspace(-1, 1, num=num_levels)
    return levels_uni

def get_quantile_levels(bits, mean, sigma, min, max):
    """quantile levels """
    num_levels = 2 << bits - 1
    cdf_points = np.linspace(0, 1, num=num_levels - 2)
    levels = [truncnorm.ppf(level, min, max, loc=mean, scale=sigma) for level in cdf_points]
    
    levels = [min] + levels + [max]
    return levels

def get_level(x, levels):
    for index, level in enumerate(levels[0:len(levels) - 1]):
        if x >= levels[index] and x < levels[index + 1]:
            return index

def normal_function(x, mean, sigma, min, max):
    f = truncnorm.pdf(x, min, max, loc=mean, scale=sigma)
    return f

def truncated_cdf(x, minimum, maximum, loc, scale):
    a = norm.cdf(x, loc=loc, scale=scale) - norm.cdf(minimum, loc=loc, scale=scale)
    b = norm.cdf(maximum, loc=loc, scale=scale) - norm.cdf(minimum, loc=loc, scale=scale)
    return a / b

def truncated_pdf(x, minimum, maximum, loc, scale):
    return scale * norm.pdf(x, loc=loc, scale=scale) / (norm.cdf(maximum, loc=loc, scale=scale) - norm.cdf(minimum, loc=loc, scale=scale))

def truncated_ppf(x, minimum, maximum, loc, scale):
    y_hat = (norm.cdf(maximum, loc=loc, scale=scale) - norm.cdf(minimum, loc=loc, scale=scale)) * x + norm.cdf(minimum, loc=loc, scale=scale)
    return norm.ppf(y_hat, loc=loc, scale=scale)

def calculate_estimated_error(mean, sigma, levels, min, max):
    sum = []
    for index, level in enumerate(levels[:-1]):
        def inline_func(x):
            normal_func = normal_function(x, mean, sigma, min, max)
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
    for epoch in range(epochs):
        indexes = list(range(len(initial_levels)))[1:-1]
        for index in indexes:
            a = (initial_levels[index - 1] - mean) / (initial_levels[index + 1] - initial_levels[index - 1])
            b = truncnorm.cdf(initial_levels[index + 1], minimum, maximum, loc=mean, scale=sigma) - truncnorm.cdf(initial_levels[index - 1], minimum, maximum, loc=mean, scale=sigma)
            c = (truncnorm.pdf(initial_levels[index + 1], minimum, maximum, loc=mean, scale=sigma) - truncnorm.pdf(initial_levels[index - 1], minimum, maximum, loc=mean, scale=sigma)) / (initial_levels[index + 1] - initial_levels[index - 1])
            initial_levels[index] = truncnorm.ppf(
                truncnorm.cdf(initial_levels[index + 1], minimum, maximum, loc=mean, scale=sigma) + a * b + sigma ** 2 * c, minimum, maximum, loc=mean, scale=sigma
                )

        losses.append(calculate_estimated_error(mean, sigma, initial_levels, minimum, maximum))
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
            self.levels = get_quantile_levels(bits, 0, 0.1, min, max)
            self.qdq = qdqL2
        elif method == 'nuq2inf':
            self.levels = get_quantile_levels(bits, 0, 0.1, min, max)
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
                    print('Creating new levels')
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
        elif method == 'none':
            return

        self.number_of_iterations = 0
        self.gradient_samples = []
        self.gradient_samples_overtime = []

        self.bucket_size = bucket_size
        self.bits = bits
        self.co_epochs = kwargs['cd_epochs']
        print('kwargs is equal to:', kwargs)
        self.levels = torch.as_tensor(self.levels, dtype=torch.float32).cuda()
        self.qdq = QDQ(self.levels)
        self.mean = 0
        self.variance = 0.1
        self.error = None
    

    def exponential_gd(self, initial_point, learning_rate=0.7, epochs=10000):
        mean = self.mean.cpu().item()
        variance = torch.sqrt(self.variance).cpu().item()
        p = initial_point
        s = 2 ** (self.bits - 1) - 1 
        a, b = (-self.interval - mean) / variance, (self.interval - mean) / variance
        def trunc_cdf(val):
            return truncnorm.cdf(val, a, b, loc=mean, scale=variance)
        def trunc_pdf(val):
            return truncnorm.pdf(val, a, b, loc=mean, scale=variance)
        i = 0
        while True:
            def arg1_1(j):
                return mean * (j * p ** (j - 1) + (j + 1) * p ** j) - (2 * j + 1) * p ** (2 * j)
            arg1 = torch.sum(torch.tensor([ arg1_1(j) * (trunc_cdf(p ** j) - trunc_cdf(p ** (j+1))) for j in range(0, s)]))
            def arg2_1(j):
                return j * p ** (j - 1) + (j + 1) * p ** j
            arg2 = torch.sum(torch.tensor([arg2_1(j) * (trunc_pdf(p ** (j + 1)) - trunc_pdf(p ** (j))) for j in range(0, s)]))
            gradient = 2 * s * (p ** (2 * s - 1)) * (trunc_cdf(p ** s) - trunc_cdf(0)) + arg1 + variance ** 2 * arg2
            if i == 10:
                break
            p = p - learning_rate * gradient
            print('Multiplier value is', p, 'Epoch is ', i, 'gradient', gradient)
            i += 1
        return p

    def set_mean_variance(self, mean, variance):
        self.mean = mean
        self.variance = variance
        print('Current mean is', mean, 'current variance is', variance)
        sigma = torch.sqrt(variance.cpu())
        a, b = (-self.interval - mean.cpu().item()) / sigma, (self.interval - mean.cpu().item()) / sigma
        self.error = calculate_estimated_error(self.mean.cpu(), sigma, self.levels.cpu(), a, b)
        print('Error for CO is ', self.error)

    def update_levels(self):
        if self.method == 'nuq2':
            sigma = torch.sqrt(self.variance).cpu()
            a, b = (-self.interval - self.mean.cpu().item()) / sigma, (self.interval - self.mean.cpu().item()) / sigma
            self.levels = get_quantile_levels(self.bits, self.mean.cpu(), torch.sqrt(self.variance.cpu()), -self.interval, self.interval)
            initial_levels = self.levels
            self.levels, all_levels, losses = get_adaptive_levels_co(initial_levels, len(self.levels), self.mean.cpu(), torch.sqrt(self.variance.cpu()), self.co_epochs, a, b)
            print('Levels are', self.levels)
        elif self.method == 'nuq3':
            mean = self.mean.cpu()
            self.previous_best = None
            initial_points = []

            if self.previous_best == None:
                initial_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9]
            else:
                initial_points = [0.1, 0.2, 0.3, 0.4, self.previous_best,  0.8, 0.9]
            optimal_points = []
            sigma = torch.sqrt(self.variance).cpu()
            a, b = (-self.interval - mean) / sigma, (self.interval - mean) / sigma
            a = a.cpu()
            b = b.cpu()
            for point in initial_points:
                optimal_p = self.exponential_gd(point)
                optimal_points.append(optimal_p)
            half_point = 2 ** (self.bits - 1)
            optimal_points_costs = [calculate_new_error(get_exp_levels(self.bits, p)[half_point:], mean, sigma, a, b) for p in optimal_points]
            print('Costs are', optimal_points_costs)
            print('Points are', optimal_points)
            index = np.argmin(optimal_points_costs)
            self.multiplier = optimal_points[index]
            print('Chosen multiplier is', self.multiplier)
            self.previous_best = self.multiplier
            self.levels = get_exp_levels(self.bits, self.multiplier)
            print('Levels are', self.levels)

        elif self.method == 'nuq4':
            mean = self.mean.cpu()
            self.previous_best = None
    
            initial_points = []

            if self.previous_best == None:
                initial_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9]
            else:
                initial_points = [0.1, 0.2, 0.3, 0.4, self.previous_best,  0.8, 0.9]
            optimal_points = []
            sigma = torch.sqrt(self.variance).cpu()
            a, b = (-self.interval - mean) / sigma, (self.interval - mean) / sigma
            a = a.cpu()
            b = b.cpu()
            for point in initial_points:
                optimal_p = self.exponential_gd(point)
                optimal_points.append(optimal_p)
            half_point = 2 ** (self.bits - 1)
            optimal_points_costs = [calculate_new_error(get_exp_levels(self.bits, p)[half_point:], mean, sigma, a, b) for p in optimal_points]
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
