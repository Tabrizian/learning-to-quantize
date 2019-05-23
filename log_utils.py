from collections import OrderedDict, defaultdict
import numpy as np
from tensorboardX import SummaryWriter
import time
import torch
import os


class TBXWrapper(object):
    def configure(self, logger_name, flush_secs=5, opt=None):
        self.writer = SummaryWriter(logger_name, flush_secs=flush_secs)
        self.logger_name = logger_name
        self.logobj = defaultdict(lambda: list())
        self.opt = opt

    def log_value(self, name, val, step):
        self.writer.add_scalar(name, val, step)
        self.logobj[name] += [(time.time(), step, float(val))]

    def add_scalar(self, name, val, step):
        self.log_value(name, val, step)

    def save_log(self, filename='log.pth.tar'):
        try:
            os.makedirs(self.opt.logger_name)
        except os.error:
            pass
        torch.save(dict(self.logobj), self.opt.logger_name+'/'+filename)

    def close(self):
        self.writer.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        if self.count == 0:
            return '%d' % self.val
        return '%.4f (%.4f)' % (self.val, self.avg)

    def tb_log(self, tb_logger, name, step=None):
        tb_logger.log_value(name, self.val, step=step)


class TimeMeter(object):
    """Store last K times"""

    def __init__(self, k=1000):
        self.k = k
        self.reset()

    def reset(self):
        self.vals = [0]*self.k
        self.i = 0
        self.mu = 0

    def update(self, val):
        self.vals[self.i] = val
        self.i = (self.i + 1) % self.k
        self.mu = (1-1./self.k)*self.mu+(1./self.k)*val

    def __str__(self):
        # return '%.4f +- %.2f' % (np.mean(self.vals), np.std(self.vals))
        return '%.4f +- %.2f' % (self.mu, np.std(self.vals))

    def tb_log(self, tb_logger, name, step=None):
        tb_logger.log_value(name, self.vals[0], step=step)


class StatisticMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mu = AverageMeter()
        self.std = AverageMeter()
        self.min = AverageMeter()
        self.max = AverageMeter()
        self.med = AverageMeter()

    def update(self, val, n=0):
        val = np.ma.masked_invalid(val)
        val = val.compressed()
        n = min(n, len(val))
        if n == 0:
            return
        self.mu.update(np.mean(val), n=n)
        self.std.update(np.std(val), n=n)
        self.min.update(np.min(val), n=n)
        self.max.update(np.max(val), n=n)
        self.med.update(np.median(val), n=n)

    def __str__(self):
        # return 'mu:{}|med:{}|std:{}|min:{}|max:{}'.format(
        #     self.mu, self.med, self.std, self.min, self.max)
        return 'mu:{}|med:{}'.format(self.mu, self.med)

    def tb_log(self, tb_logger, name, step=None):
        self.mu.tb_log(tb_logger, name+'_mu', step=step)
        self.med.tb_log(tb_logger, name+'_med', step=step)
        self.std.tb_log(tb_logger, name+'_std', step=step)
        self.min.tb_log(tb_logger, name+'_min', step=step)
        self.max.tb_log(tb_logger, name+'_max', step=step)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self, opt):
        self.meters = OrderedDict()
        self.log_keys = opt.log_keys.split(',')

    def reset(self):
        self.meters = OrderedDict()

    def update(self, k, v, n=0, log_scale=False, bins=100):
        if k not in self.meters:
            if type(v).__module__ == np.__name__:
                self.meters[k] = StatisticMeter()
            else:
                self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if k in self.log_keys or 'all' in self.log_keys:
                if i > 0:
                    s += '  '
                s += k+': '+str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        for k, v in self.meters.items():
            v.tb_log(tb_logger, prefix+k, step=step)


class Profiler(object):
    def __init__(self, k=10):
        self.k = k
        self.meters = OrderedDict()
        self.start()

    def tic(self):
        self.t = time.time()

    def toc(self, name):
        end = time.time()
        if name not in self.times:
            self.times[name] = []
        self.times[name] += [end-self.t]
        self.tic()

    def start(self):
        self.times = OrderedDict()
        self.tic()

    def end(self):
        for k, v in self.times.items():
            if k not in self.meters:
                self.meters[k] = TimeMeter(self.k)
            self.meters[k].update(sum(v))
        self.start()

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k+': ' + str(v)
        return s
