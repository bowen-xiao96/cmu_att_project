# define a general trainer for training classification networks in pytorch
import os, sys
import time
import pickle
import numpy as np

import torch
import torch.autograd as A

from utils.metric import *

# ========== state data ==========
model_ = None
optimizer_ = None
train_dataloader_ = None
test_dataloader_ = None
criterion_ = None
new_loss_ = None

start_epoch_ = None
max_epoch_ = None
lr_sched_ = None
call_back_ = None

display_freq_ = None
output_dir_ = None
save_every_ = None
max_keep_ = None

train_loss_ = None
train_top1_accu_ = None
test_loss_ = None
test_top1_accu_ = None
test_model_ = None

# ========== end state data ==========


def start(**kwargs):
    print('* Initializing Trainer module... *')

    for k, v in kwargs.items():
        print('%s: %s' % (k, v))
        k_ = k + '_'
        if k_ in globals():
            globals()[k_] = v
        else:
            print('WARNING: unsupported Trainer parameter: ' + k)

    # parse model loss function
    model = globals()['model_']
    if hasattr(model, 'module') and hasattr(model.module, 'loss_type'):
        globals()['new_loss_'] = True
    else:
        globals()['new_loss_'] = False

    # parse starting epoch number
    if globals()['start_epoch_'] is None:
        globals()['start_epoch_'] = 0

    if globals()['start_epoch_'] > 0:
        # load existing training log file
        log_file = np.load('training_log.npz')
        globals()['train_loss_'] = log_file['train_loss'].tolist()
        globals()['train_top1_accu_'] = log_file['train_top1_accu'].tolist()
        globals()['test_loss_'] = log_file['test_loss'].tolist()
        globals()['test_top1_accu_'] = log_file['test_top1_accu'].tolist()

    else:
        # start from scratch
        globals()['train_loss_'] = list()
        globals()['train_top1_accu_'] = list()
        globals()['test_loss_'] = list()
        globals()['test_top1_accu_'] = list()

    print('* Start training... *')
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

        if call_back_:
            call_back_(epoch, i, locals(), globals())

        x = A.Variable(x.cuda())
        y = A.Variable(y.cuda())
        batch_size = x.size(0)

        # compute output
        if new_loss_:
            pred, loss = model_(x, y)
            loss = torch.mean(loss)
        else:
            pred = model_(x)
            loss = criterion_(pred, y)

        # measure accuracy and record loss
        train_loss_.append(loss.data[0])
        losses.update(loss.data[0], batch_size)

        # if final prediction has multiple time steps, `accuracy` function will take care
        prec1, prec5 = accuracy(pred.data, y.data, topk=(1, 5))
        train_top1_accu_.append(prec1[0])
        top1.update(prec1[0], batch_size)
        top5.update(prec5[0], batch_size)

        # compute gradient and do SGD step
        optimizer_.zero_grad()
        loss.backward()
        optimizer_.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % display_freq_ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_dataloader_), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('\t* Epoch: [{0}] TRAIN *\t'
          'Loss {loss.avg:.4f}\t'
          'Prec@1 {top1.avg:.3f}\t'
          'Prec@5 {top5.avg:.3f}'.format(
        epoch, loss=losses, top1=top1, top5=top5))


def test(epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_.eval()

    for x, y in test_dataloader_:
        x = A.Variable(x.cuda(), volatile=True)
        y = A.Variable(y.cuda(), volatile=True)

        # compute output
        pred = model_(x)
        loss = criterion_(pred, y)

        # measure accuracy and record loss
        batch_size = x.size(0)

        if isinstance(pred, tuple):
            pred = pred[-1]

        test_loss_.append(loss.data[0])
        losses.update(loss.data[0], batch_size)

        prec1, prec5 = accuracy(pred.data, y.data, topk=(1, 5))
        test_top1_accu_.append(prec1[0])
        top1.update(prec1[0], batch_size)
        top5.update(prec5[0], batch_size)

    print('\t* Epoch: [{0}] TEST  *\t'
          'Loss {loss.avg:.4f}\t'
          'Prec@1 {top1.avg:.3f}\t'
          'Prec@5 {top5.avg:.3f}'.format(
        epoch, loss=losses, top1=top1, top5=top5))

    return top1.avg


def save(i, test_accu, best=False):
    # save training log
    np.savez(
        os.path.join(output_dir_, 'training_log.npz'),
        train_loss=np.array(train_loss_),
        train_top1_accu=np.array(train_top1_accu_),
        test_loss=np.array(test_loss_),
        test_top1_accu=np.array(test_top1_accu_)
    )

    # save model
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

    for i in range(start_epoch_, max_epoch_):
        if lr_sched_:
            lr_sched_(optimizer_, i)

        if test_model_ != 1:
            train_one_epoch(i)
        test_accu = test(i)

        # save models
        if test_accu > best_accu:
            best_accu = test_accu
            save(i, test_accu, best=True)

        if save_every_ and i % save_every_ == 0:
            while len(saved_models) >= max_keep_:
                old_model_file = saved_models.pop(0)
                try:
                    os.remove(old_model_file)
                except:
                    pass

            model_file = save(i, test_accu)
            saved_models.append(model_file)
