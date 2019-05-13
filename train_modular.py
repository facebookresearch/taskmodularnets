# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import tqdm
from data import dataset as dset
import torchvision.models as tmodels
import tqdm
from archs import models
import os
import itertools
import glob
import pdb
import math
import sys

import tensorboardX as tbx
from utils import utils

import torch.backends.cudnn as cudnn

from flags import parser
args = parser.parse_args()

os.makedirs(args.cv_dir + '/' + args.name, exist_ok=True)
utils.save_args(args)
print(' '.join(sys.argv))

torch.manual_seed(1992)
np.random.seed(1992)


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    factor = 1.0
    if epoch == args.steps[0]:
        factor = 0.1
    param_groups = optimizer.param_groups
    for p in param_groups:
        p['lr'] *= factor


#----------------------------------------------------------------#
def train(epoch):

    model.train()
    lossmeter = utils.AverageMeter()
    lossauxmeter = utils.AverageMeter()
    accmeter = utils.AverageMeter()

    train_loss = 0.0
    for idx, data in enumerate(trainloader):
        data = [d.cuda() for d in data]
        loss, all_losses, acc, _ = model(data)
        if acc is not None:
            accmeter.update(acc, data[0].shape[0])

        mainloss = all_losses['main_loss']
        lossmeter.update(mainloss.item(), data[0].shape[0])
        if 'aux_loss' in all_losses.keys():
            loss_aux = all_losses['aux_loss']
            lossauxmeter.update(loss_aux.item(), data[0].shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if idx % args.print_every == 0:
            print(
                'Epoch: {} Iter: {}/{} | Loss: {:.3f}, Loss Aux: {:.3f}, Acc: {:.2f}'.
                format(epoch, idx, len(trainloader), lossmeter.avg,
                       lossauxmeter.avg, accmeter.avg))
            print(','.join([
                '{}: {:.02f}  '.format(k, v.item())
                for k, v in all_losses.items()
            ]))

    train_loss = train_loss / len(trainloader)
    logger.add_scalar('train_loss', train_loss, epoch)
    for k, v in all_losses.items():
        logger.add_scalar('train_{}'.format(k), v.item(), epoch)
    print('Epoch: {} | Loss: {} | Acc: {}'.format(epoch, lossmeter.avg,
                                                  accmeter.avg))
    if epoch % args.save_every == 0:
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
        }
        torch.save(state,
                   args.cv_dir + '/{}/ckpt_E_{}.t7'.format(args.name, epoch))


def test(epoch):

    model.eval()

    accuracies = []
    all_attr_lab = []
    all_obj_lab = []
    all_pred = []
    pairs = valloader.dataset.pairs
    objs = valloader.dataset.objs
    attrs = valloader.dataset.attrs
    if args.test_set == 'test':
        val_pairs = valloader.dataset.test_pairs
    else:
        val_pairs = valloader.dataset.val_pairs
    train_pairs = valloader.dataset.train_pairs
    for idx, data in enumerate(valloader):
        data = [d.cuda() for d in data]
        attr_truth, obj_truth = data[1], data[2]
        _, _, _, predictions = model(data)
        predictions, feats = predictions
        all_pred.append(predictions)
        all_attr_lab.append(attr_truth)
        all_obj_lab.append(obj_truth)

        if idx % 100 == 0:
            print('Tested {}/{}'.format(idx, len(valloader)))
    all_attr_lab = torch.cat(all_attr_lab)
    all_obj_lab = torch.cat(all_obj_lab)
    all_pair_lab = [
        val_pairs.index((attrs[all_attr_lab[i]], objs[all_obj_lab[i]]))
        for i in range(len(all_attr_lab))
    ]

    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k] for i in range(len(all_pred))])
    all_accuracies = []

    # Calculate best unseen acc
    # put everything on cpu
    attr_truth, obj_truth = all_attr_lab.cpu(), all_obj_lab.cpu()
    pairs = list(
        zip(list(attr_truth.cpu().numpy()), list(obj_truth.cpu().numpy())))
    seen_ind = torch.LongTensor([
        i for i in range(len(attr_truth))
        if pairs[i] in evaluator_val.train_pairs
    ])
    unseen_ind = torch.LongTensor([
        i for i in range(len(attr_truth))
        if pairs[i] not in evaluator_val.train_pairs
    ])

    accuracies = []
    bias = 1e3
    args.bias = bias
    results = evaluator_val.score_model(
        all_pred_dict, all_obj_lab, bias=args.bias)
    match_stats = evaluator_val.evaluate_predictions(results, all_attr_lab,
                                                     all_obj_lab)
    accuracies.append(match_stats)
    meanAP = 0
    _, _, _, _, _, _, open_unseen_match = match_stats
    accuracies = zip(*accuracies)
    open_unseen_match = open_unseen_match.byte()
    accuracies = list(map(torch.mean, map(torch.cat, accuracies)))
    attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc, open_seen_acc, open_unseen_acc = accuracies
    max_seen_scores = results['scores'][
        unseen_ind][:, evaluator_val.seen_mask].max(1)[0]
    max_unseen_scores = results['scores'][
        unseen_ind][:, 1 - evaluator_val.seen_mask].max(1)[0]
    unseen_score_diff = max_seen_scores - max_unseen_scores
    correct_unseen_score_diff = unseen_score_diff[open_unseen_match] - 1e-4
    full_unseen_acc = [(
        epoch,
        attr_acc,
        obj_acc,
        closed_acc,
        open_acc,
        (open_seen_acc * open_unseen_acc)**0.5,
        0.5 * (open_seen_acc + open_unseen_acc),
        open_seen_acc,
        open_unseen_acc,
        objoracle_acc,
        meanAP,
        bias,
    )]
    print(
        '(val) E: %d | A: %.3f | O: %.3f | Cl: %.3f | Op: %.4f | OpHM: %.4f | OpAvg: %.4f | OpSeen: %.4f | OpUnseen: %.4f  | OrO: %.4f | maP: %.4f | bias: %.3f'
        % (
            epoch,
            attr_acc,
            obj_acc,
            closed_acc,
            open_acc,
            (open_seen_acc * open_unseen_acc)**0.5,
            0.5 * (open_seen_acc + open_unseen_acc),
            open_seen_acc,
            open_unseen_acc,
            objoracle_acc,
            meanAP,
            bias,
        ))
    return all_accuracies


#----------------------------------------------------------------#
trainset = dset.CompositionDatasetActivations(
    root=args.data_dir,
    phase='train',
    split=args.splitname,
    num_negs=args.num_negs,
    pair_dropout=args.pair_dropout,
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers)
valset = dset.CompositionDatasetActivations(
    root=args.data_dir,
    phase=args.test_set,
    split=args.splitname,
    subset=args.subset,
)
valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=args.workers)

if args.model == 'modularpretrained':
    model = models.GatedGeneralNN(
        trainset,
        args,
        num_layers=args.nlayers,
        num_modules_per_layer=args.nmods)
else:
    raise (NotImplementedError)

evaluator_train = models.Evaluator(trainset, model)
evaluator_val = models.Evaluator(valset, model)

if 'modular' in args.model:
    gating_params = [
        param for name, param in model.named_parameters()
        if 'gating_network' in name and param.requires_grad
    ]
    network_params = [
        param for name, param in model.named_parameters()
        if 'gating_network' not in name and param.requires_grad
    ]
    optim_params = [
        {
            'params': network_params,
        },
        {
            'params': gating_params,
            'lr': args.lrg
        },
    ]
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)
else:
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

model.cuda()
print(model)

start_epoch = 0
if args.load is None:
    for epoch in range(1000, -1, -1):
        ckpt_files = glob.glob(args.cv_dir +
                               '/{}/ckpt_E_{}*.t7'.format(args.name, epoch))
        if len(ckpt_files):
            args.load = ckpt_files[-1]
            break
if args.load is not None:
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    print('loaded model from', os.path.basename(args.load))

logger = tbx.SummaryWriter('logs/{}'.format(args.name))
if args.test_only:
    out = test(start_epoch)
    traintest(start_epoch)
else:
    for epoch in range(start_epoch, args.max_epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        if epoch % args.pair_dropout_epoch == 0:
            trainloader.dataset.reset_dropout()
        train(epoch)
        if (epoch) % args.eval_val_every == 0:
            with torch.no_grad():
                out = test(epoch)
