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
import collections

import tensorboardX as tbx
from utils import utils

import torch.backends.cudnn as cudnn

from flags import parser

args = parser.parse_args()

os.makedirs(args.cv_dir + '/' + args.name, exist_ok=True)
utils.save_args(args)


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
    all_pair_lab = torch.LongTensor([
        pairs.index((attrs[all_attr_lab[i]], objs[all_obj_lab[i]]))
        for i in range(len(all_attr_lab))
    ])
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
    match_stats = evaluator_val.evaluate_predictions(
        results, all_attr_lab, all_obj_lab, topk=args.topk)
    accuracies.append(match_stats)
    meanAP = 0
    _, _, _, _, _, _, open_unseen_match = match_stats
    accuracies = zip(*accuracies)
    open_unseen_match = open_unseen_match.byte()
    accuracies = list(map(torch.mean, map(torch.cat, accuracies)))
    attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc, open_seen_acc, open_unseen_acc = accuracies
    scores = results['scores']
    correct_scores = scores[torch.arange(scores.shape[0]), all_pair_lab][
        unseen_ind]
    max_seen_scores = results['scores'][
        unseen_ind][:, evaluator_val.seen_mask].topk(
            args.topk, dim=1)[0][:, args.topk - 1]
    unseen_score_diff = max_seen_scores - correct_scores
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

    correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
    magic_binsize = 20
    bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
    biaslist = correct_unseen_score_diff[::bias_skip]

    for bias in biaslist:
        accuracies = []
        args.bias = bias
        results = evaluator_val.score_model(
            all_pred_dict, all_obj_lab, bias=args.bias)
        match_stats = evaluator_val.evaluate_predictions(
            results, all_attr_lab, all_obj_lab, topk=args.topk)
        accuracies.append(match_stats)
        meanAP = 0

        accuracies = zip(*accuracies)
        accuracies = map(torch.mean, map(torch.cat, accuracies))
        attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc, open_seen_acc, open_unseen_acc = accuracies
        all_accuracies.append((
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
    all_accuracies.extend(full_unseen_acc)
    seen_accs = np.array([a[-5].item() for a in all_accuracies])
    unseen_accs = np.array([a[-4].item() for a in all_accuracies])
    area = np.trapz(seen_accs, unseen_accs)
    print(
        '(val) E: %d | A: %.3f | O: %.3f | Cl: %.3f | AUC: %.4f | Op: %.4f | OpHM: %.4f | OpAvg: %.4f | OpSeen: %.4f | OpUnseen: %.4f  | OrO: %.4f | bias: %.3f'
        % (
            epoch,
            attr_acc,
            obj_acc,
            closed_acc,
            area,
            open_acc,
            (open_seen_acc * open_unseen_acc)**0.5,
            0.5 * (open_seen_acc + open_unseen_acc),
            open_seen_acc,
            open_unseen_acc,
            objoracle_acc,
            bias,
        ))

    all_accuracies = [all_accuracies, area]
    return all_accuracies


#----------------------------------------------------------------#

#----------------------------------------------------------------#
trainset = dset.CompositionDatasetActivations(
    root=args.data_dir, phase='train', split=args.splitname)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers)
valset = dset.CompositionDatasetActivations(
    root=args.data_dir,
    phase=args.test_set,
    split=args.splitname,
    subset=args.subset)
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
all_acc = test(start_epoch)
