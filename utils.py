import shutil
import torch
import numpy as np


class DictWrapper(object):
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        if key in self.d:
            return self.d[key]
        else:
            return None


class SaveCheckpoint(object):
    def __init__(self):
        # remember best prec@1 and save checkpoint
        self.best_prec1 = 0

    def __call__(self, model, prec1, opt, optimizer,
                 filename='checkpoint.pth.tar', gvar=None):
        is_best = prec1 > self.best_prec1
        self.best_prec1 = max(prec1, self.best_prec1)
        state = {
            'epoch': optimizer.epoch + 1,
            'niters': optimizer.niters,
            'opt': opt.d,
            'model': model.state_dict(),
            'best_prec1': self.best_prec1,
        }
        if gvar is not None:
            state.update({'gvar': gvar.state_dict()})

        torch.save(state, opt.logger_name+'/'+filename)
        if is_best:
            shutil.copyfile(opt.logger_name+'/'+filename,
                            opt.logger_name+'/model_best.pth.tar')


def base_lr(optimizer, opt):
    lr = opt.lr
    return lr


def adjust_lr(optimizer, opt):
    if opt.niters > 0:
        niters = optimizer.niters
    else:
        niters = optimizer.niters//opt.epoch_iters
    if isinstance(opt.lr_decay_epoch, str):
        adjust_learning_rate_multi(optimizer, niters, opt)
    else:
        adjust_learning_rate(optimizer, niters, opt)


def adjust_learning_rate(optimizer, epoch, opt):
    """ Sets the learning rate to the initial LR decayed by 10 """
    if opt.exp_lr:
        """ test
        A=np.arange(200);
        np.round(np.power(.1, np.power(2., A/80.)-1), 6)[[0,80,120,160]]
        test """
        last_epoch = 2. ** (float(epoch) / int(opt.lr_decay_epoch)) - 1
    else:
        last_epoch = epoch // int(opt.lr_decay_epoch)
    lr = base_lr(optimizer, opt) * (0.1 ** last_epoch)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_multi(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr_decay_epoch = np.array(list(map(int, opt.lr_decay_epoch.split(','))))
    if len(lr_decay_epoch) == 1:
        return adjust_learning_rate(optimizer, epoch, opt)
    el = (epoch // lr_decay_epoch)
    ei = np.where(el > 0)[0]
    if len(ei) == 0:
        ei = [0]
    print(el)
    print(ei)
    # lr = opt.lr * (opt.lr_decay_rate ** (ei[-1] + el[ei[-1]]))
    lr = base_lr(optimizer, opt) * (
            opt.lr_decay_rate ** (ei[-1]+(el[ei[-1]] > 0)))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
