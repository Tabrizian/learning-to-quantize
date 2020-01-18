import numpy as np
import torch
from cuquant import QDQ
import math


def get_uniform_levels(bits):
    """uniform (QSGD)"""
    num_levels = 2 << bits - 1
    levels_uni = np.linspace(-1, 1, num=num_levels)
    return levels_uni


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

    num_bucket = int(np.ceil(len(x) / bucket_size))
    quant = []
    for bucket_i in range(1, num_bucket + 1):

        x_bucket = x[(bucket_i - 1) * bucket_size:bucket_i * bucket_size]

        norm = np.sqrt(x_bucket@x_bucket.T)
        uni_rand = np.random.rand(len(x_bucket))
        for i in range(len(x_bucket)):
            j = 0
            while j + 1 < len(levels):
                level_up = levels[j + 1]
                if x_bucket[i] / norm <= level_up:
                    diff = level_up - levels[j]
                    if x_bucket[i] / norm + diff * uni_rand[i] > level_up:
                        j = j + 1
                    break
                j = j + 1
            quant.append(norm * levels[j])
    return np.asarray(quant)


def qdqLinf(x, levels, bucket_size=1024):
    """
    Quantize and dequantize with L-inf norm.
    """

    num_bucket = int(np.ceil(len(x) / bucket_size))
    quant = []
    for bucket_i in range(1, num_bucket + 1):

        x_bucket = x[(bucket_i - 1) * bucket_size:bucket_i * bucket_size]

        max_val = x_bucket.abs().max(0)
        uni_rand = np.random.rand(len(x_bucket))
        for i in range(len(x_bucket)):
            j = 0
            while j + 1 < len(levels):
                level_up = levels[j + 1]
                if x_bucket[i] / max_val <= level_up:
                    diff = level_up - levels[j]
                    if x_bucket[i] / max_val + diff * uni_rand[i] > level_up:
                        j = j + 1
                    break
                j = j + 1
            quant.append(max_val * levels[j])
    return np.asarray(quant)


class QuantizeNumPy(object):
    def __init__(self, method, bits, bucket_size, **kwargs):
        """
        QSGD: qdqL2 + levels_uni
        NUQSGD: qdqL2 + levels_exp
        QSGD-inf: qdqLinf + levels_uni
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

        self.bucket_size = bucket_size
        self.bits = bits

    def quantize(self, x):
        return self.qdq(x, self.levels, self.bucket_size)


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
