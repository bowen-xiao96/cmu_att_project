# define a general trainer for pytorch deep learning framework
import os, sys
import time
import shutil
import pickle
import numpy as np

import torch
import torch.autograd as A
import torch.nn as nn
import torch.optim as optim
from utils import add_gaussian_noise
#import matplotlib.pyplot as plt

# ========== state data ==========
model_ = None
optimizer_ = None

train_dataloader_ = None
test_dataloader_ = None
criterion_ = None

max_epoch_ = None
lr_sched_ = None
init_lr_ = None
lr_decay_ = None
lr_freq_ = None

call_back_ = None

display_freq_ = None
output_dir_ = None
save_every_ = None
max_keep_ = None
save_model_data_ = None
add_noise_ = None
test_model_ = None
start_loc_ = 0

train_loss_history = []
train_top1_history = []

test_loss_history = []
test_top1_history = []

# ========== end state data ==========


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def start(**kwargs):
    print('* Initializing Trainer module... *')

    for k, v in kwargs.items():
        print('%s: %s' % (k, v))
        k_ = k + '_'
        if k_ in globals():
            globals()[k_] = v

    main_loop()


def train_one_epoch(epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model_.train()

    end = time.time()
    for i, (x, y) in enumerate(train_dataloader_):
        # measure data loading time
        data_time.update(time.time() - end)
 
        x = A.Variable(x.cuda())
        y = A.Variable(y.cuda())

        # compute output
        pred = model_(x)
        loss = criterion_(pred, y)
        
        if isinstance(pred, tuple):
            pred = pred[-1]
        if isinstance(pred, list):
            pred = pred[-1]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(pred.data, y.data, topk=(1, 5))
        losses.update(loss.data[0], x.size(0))
        top1.update(prec1[0], x.size(0))
        top5.update(prec5[0], x.size(0))

        # compute gradient and do SGD step
        optimizer_.zero_grad()
        loss.backward()
        optimizer_.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % display_freq_ == 0:
            for param_group in optimizer_.param_groups:
                current_lr = param_group['lr']
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f}\t'
                  'Learning Rate {current_lr:.8f})'.format(
                epoch, i, len(train_dataloader_), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, current_lr=current_lr))

            train_loss_history.append(losses)
            train_top1_history.append(top1)

    print('\t* Epoch: [{0}] TRAIN *\t'
          'Loss {loss.avg:.4f}\t'
          'Prec@1 {top1.avg:.3f}\t'
          'Prec@5 {top5.avg:.3f}\t'.format(
        epoch, loss=losses, top1=top1, top5=top5))
    

def test(epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_.eval()

    for x, y in test_dataloader_:

        x = A.Variable(x.cuda(), volatile=True)
        y = A.Variable(y.cuda(), volatile=True)
        
        if add_noise_ == 1:
            x = add_gaussian_noise(x)

        # compute output
        pred = model_(x)
        loss = criterion_(pred, y)

        if isinstance(pred, tuple):
            pred = pred[-1]
        if isinstance(pred, list):
            pred = pred[-1]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(pred.data, y.data, topk=(1, 5))
        losses.update(loss.data[0], x.size(0))
        top1.update(prec1[0], x.size(0))
        top5.update(prec5[0], x.size(0))

    print('\t* Epoch: [{0}] TEST *\t'
          'Loss {loss.avg:.4f}\t'
          'Prec@1 {top1.avg:.3f}\t'
          'Prec@5 {top5.avg:.3f}\t'.format(
        epoch, loss=losses, top1=top1, top5=top5))

    return top1.avg


def save(i, test_accu, best=False):
    if not best:
        model_name = os.path.join(output_dir_, str(i) + '.pkl')
    else:
        model_name = os.path.join(output_dir_, 'best.pkl')

    torch.save(
        (i, test_accu, model_.state_dict()),  # saved data
        model_name,
        pickle_protocol=pickle.HIGHEST_PROTOCOL
    )
    return model_name


def main_loop():

    if not os.path.exists(output_dir_):
        os.makedirs(output_dir_)


    best_accu = -1.0
    saved_models = list()      
    
    save_accuracy = np.zeros((max_epoch_, 1))
    save_lr = np.zeros((max_epoch_, 1))

    for i in range(start_loc_, max_epoch_):
        if call_back_:
            call_back_(globals(), i)
        
        if lr_sched_:
            lr_sched_(optimizer_, i, init_lr=init_lr_, lr_decay=lr_decay_, lr_freq=lr_freq_)
        
        if test_model_ == 0:
            train_one_epoch(i)

        test_accu = test(i)

        save_accuracy[i] = test_accu
        for param_group in optimizer_.param_groups:
                save_lr[i] = param_group['lr']

        # save models
        if test_accu > best_accu:
            best_accu = test_accu
            save(i, test_accu, best=True)
        
        print("Best Top1:", best_accu)

        if save_every_ and i % save_every_ == 0:
            while len(saved_models) >= max_keep_:
                old_model_file = saved_models.pop(0)
                try:
                    os.remove(old_model_file)
                except:
                    pass

            model_file = save(i, test_accu)
            saved_models.append(model_file)

        np.savez(os.path.join(output_dir_, 'training_data.npz'), acc=save_accuracy, lr=save_lr)




