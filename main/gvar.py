from __future__ import print_function
import numpy as np
import logging
import os
import sys

import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import torch.multiprocessing

import utils
import models
from data import get_loaders
from args import get_opt
from log_utils import TBXWrapper
from log_utils import Profiler
from estim.optim import OptimizerFactory
from tensorboardX import SummaryWriter

tb_logger = TBXWrapper()


def test(tb_logger, model, test_loader,
         opt, niters, set_name='Test', prefix='V'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, idx in test_loader:
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = F.nll_loss(output, target, reduction='none')
            test_loss += loss.sum().item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        wrong = len(test_loader.dataset) - correct
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        error = 100. * wrong / len(test_loader.dataset)
        logging.info(
            '\n{0} set: Average loss: {1:.4f}'
            ', Accuracy: {2}/{3} ({4:.2f}%)'
            ', Error: {5}/{3} ({6:.2f}%)\n'.format(
                set_name, test_loss, correct, len(test_loader.dataset),
                accuracy, wrong, error))
        tb_logger.log_value('%sloss' % prefix, test_loss, step=niters)
        tb_logger.log_value('%scorrect' % prefix, correct, step=niters)
        tb_logger.log_value('%swrong' % prefix, wrong, step=niters)
        tb_logger.log_value('%sacc' % prefix, accuracy, step=niters)
        tb_logger.log_value('%serror' % prefix, error, step=niters)
    return accuracy


def train(tb_logger, epoch, train_loader, model, optimizer, opt, test_loader,
          save_checkpoint, train_test_loader):
    batch_time = Profiler()
    model.train()
    profiler = Profiler()
    init_iters = optimizer.niters
    optimizer.logger.reset()
    for batch_idx in range(init_iters, opt.epoch_iters):
        profiler.start()
        # sgd step
        loss = optimizer.step(profiler)

        batch_time.toc('Time')
        batch_time.end()
        optimizer.niters += 1
        niters = optimizer.niters

        # if True:
        if batch_idx % opt.log_interval == 0:
            gvar_log = ''
            prof_log = ''
            if optimizer.gvar.is_log_iter(niters):
                gvar_log = '\t' + optimizer.gvar.log_var(model, niters)
            if opt.log_profiler:
                prof_log = '\t' + str(profiler)

            logging.info(
                'Epoch: [{0}][{1}/{2}]({niters})\t'
                'Loss: {loss:.6f}\t'
                '{batch_time}\t'
                '{opt_log}{gvar_log}{prof_log}'.format(
                    epoch, batch_idx, len(train_loader),
                    loss=loss.item(),
                    batch_time=str(batch_time),
                    opt_log=str(optimizer.logger),
                    gvar_log=gvar_log,
                    prof_log=prof_log,
                    niters=niters))
        if batch_idx % opt.tblog_interval == 0:
            tb_logger.log_value('epoch', epoch, step=niters)
            lr = optimizer.param_groups[0]['lr']
            tb_logger.log_value('lr', lr, step=niters)
            tb_logger.log_value('niters', niters, step=niters)
            tb_logger.log_value('batch_idx', batch_idx, step=niters)
            tb_logger.log_value('loss', loss, step=niters)
            optimizer.logger.tb_log(tb_logger, step=niters)
        if optimizer.niters % opt.epoch_iters == 0:
            if opt.train_accuracy:
                test(tb_logger,
                     model, train_test_loader, opt, optimizer.niters,
                     'Train', 'T')
        if optimizer.niters % opt.chkpt_iter == 0:
            prec1 = test(tb_logger,
                         model, test_loader, opt, optimizer.niters)
            save_checkpoint(model, float(prec1), opt, optimizer,
                            gvar=optimizer.gvar)
            tb_logger.save_log()
    optimizer.niters = 0


def untrain(model, gvar, opt):
    steps = opt.untrain_steps
    lr = opt.untrain_lr
    std = opt.untrain_std
    for batch_idx in range(steps):
        loss = gvar.grad(-1)
        with torch.no_grad():
            for p in model.parameters():
                p += p.grad*lr  # ascent
                p += torch.zeros_like(p.grad).normal_(0, std)  # noise
        if batch_idx % opt.log_interval == 0:
            logging.info(
                'Untrain: [{0}/{1}]\t'
                'Loss: {loss:.6f}'.format(
                    batch_idx, steps, loss=loss.item()))


def main():
    opt = get_opt()
    tb_logger.configure(opt.logger_name, flush_secs=5, opt=opt)
    logfname = os.path.join(opt.logger_name, 'log.txt')
    logging.basicConfig(
        filename=logfname,
        format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(str(opt.d))

    torch.manual_seed(opt.seed)
    if opt.cuda:
        # TODO: remove deterministic
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(opt.seed)
        np.random.seed(opt.seed)
    # helps with wide-resnet by reducing memory and time 2x
    cudnn.benchmark = True

    train_loader, test_loader, train_test_loader = get_loaders(opt)

    if opt.epoch_iters == 0:
        opt.epoch_iters = int(
            np.ceil(1. * len(train_loader.dataset) / opt.batch_size))
    opt.maxiter = opt.epoch_iters * opt.epochs
    if opt.g_epoch:
        opt.gvar_start *= opt.epoch_iters
        opt.g_optim_start = (opt.g_optim_start * opt.epoch_iters) + 1

    model = models.init_model(opt)

    optimizer = OptimizerFactory(model, train_loader, tb_logger, opt)
    epoch = 0
    save_checkpoint = utils.SaveCheckpoint()

    # optionally resume from a checkpoint
    if not opt.noresume:
        model_path = os.path.join(opt.logger_name, opt.ckpt_name)
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            best_prec1 = checkpoint['best_prec1']
            optimizer.gvar.load_state_dict(checkpoint['gvar'])
            optimizer.niters = checkpoint['niters']
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            save_checkpoint.best_prec1 = best_prec1
            print("=> loaded checkpoint '{}' (epoch {}, best_prec {})"
                  .format(model_path, epoch, best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    if opt.niters > 0:
        max_iters = opt.niters
    else:
        max_iters = opt.epochs * opt.epoch_iters

    if opt.untrain_steps > 0:
        untrain(model, optimizer.gvar, opt)

    while optimizer.niters < max_iters:
        optimizer.epoch = epoch
        utils.adjust_lr(optimizer, opt)
        ecode = train(
            tb_logger,
            epoch, train_loader, model, optimizer, opt, test_loader,
            save_checkpoint, train_test_loader)
        if ecode == -1:
            break
        epoch += 1
    tb_logger.save_log()


if __name__ == '__main__':
    main()
