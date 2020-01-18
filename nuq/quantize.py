import numpy as np
import torch
from cuquant import QDQ
import math
from scipy.stats import truncnorm



def get_uniform_levels(bits):
    """uniform (QSGD)"""
    num_levels = 2 << bits - 1
    levels_uni = np.linspace(-1, 1, num=num_levels)
    return levels_uni

def get_quantile_levels(bits, mean, sigma):
    """quantile levels """
    cdf_points = np.linspace(0, 1, num=number_of_levels)
    levels = [truncnorm.ppf(level, -1, 1, loc=mean, scale=sigma) for level in cdf_points]
    return levels

def get_adaptive_levels_co(number_of_levels, mean, sigma, epochs):
    """Adaptive Levels"""
    num_levels = 2 << bits - 1
    initial_levels = get_quantile_levels(num_levels)
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
            new_levels[index] = truncnorm.ppf(
                truncnorm.cdf(initial_levels[index + 1], minimum, maximum, loc=mean, scale=sigma) + a * b + sigma ** 2 * c, minimum, maximum, loc=mean, scale=sigma
                )

        losses.append(calculate_estimated_error(mean, sigma, new_levels))
        print('Epoch', epoch, 'error', losses[-1])
        initial_levels = new_levels
        all_levels.append(initial_levels)
    return new_levels


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


def qdqL2(x, levels, bucket_size=1024):
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
                diff = level_up - levels[j];	
                if in_vector[i]/(norm[i]+EPS)+diff*(rand_vector[i]%1000001 / 1000000.0)>level_up:
                    j = j+1
                break
            j = j+1			
        out_vector[i] = norm[i]*levels[j];	 
    return out_vector


def qdqLinf(x, levels, bucket_size=1024):
    """
    Quantize and dequantize with L-inf norm.
    """

    assert isinstance(x, torch.cuda.FloatTensor)

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
                diff = level_up - levels[j];	
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
            self.levels = get_adaptive_normalized_layers(bits)
            self.qdq = qdqL2
        elif method == 'nuq2inf':
            self.levels = get_adaptive_normalized_layers(bits)
            self.qdq = qdqLinf


        self.bucket_size = bucket_size
        self.bits = bits

    def quantize(self, x, in_place):
        return self.qdq(x, self.levels, self.bucket_size, in_place)


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
