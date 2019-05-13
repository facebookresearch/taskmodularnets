# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import numpy as np
from torch.autograd import Variable
import itertools
import copy

# Save the training script and all the arguments
import shutil


def save_args(args):
    shutil.copy('train_modular.py', args.cv_dir + '/' + args.name + '/')
    shutil.copy('archs/models.py', args.cv_dir + '/' + args.name + '/')
    with open(args.cv_dir + '/' + args.name + '/args.txt', 'w') as f:
        f.write(str(args))


class UnNormalizer:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for b in range(tensor.size(0)):
            for t, m, s in zip(tensor[b], self.mean, self.std):
                t.mul_(s).add_(m)
        return tensor


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def flatten(l):
    return list(itertools.chain.from_iterable(l))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

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


def calc_pr_ovr_noref(counts, out):
    """
  [P, R, score, ap] = calc_pr_ovr(counts, out, K)
  Input    :
    counts : number of occurrences of this word in the ith image
    out    : score for this image
  Output   :
    P, R   : precision and recall
    score  : score which corresponds to the particular precision and recall
    ap     : average precision
  """
    #binarize counts
    out = out.astype(np.float64)
    counts = np.array(counts > 0, dtype=np.float32)
    tog = np.hstack((counts[:, np.newaxis].astype(np.float64),
                     out[:, np.newaxis].astype(np.float64)))
    ind = np.argsort(out)
    ind = ind[::-1]
    score = np.array([tog[i, 1] for i in ind])
    sortcounts = np.array([tog[i, 0] for i in ind])

    tp = sortcounts
    fp = sortcounts.copy()
    for i in range(sortcounts.shape[0]):
        if sortcounts[i] >= 1:
            fp[i] = 0.
        elif sortcounts[i] < 1:
            fp[i] = 1.

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    # P = np.cumsum(tp)/(np.cumsum(tp) + np.cumsum(fp));
    P = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    numinst = np.sum(counts)

    R = tp / (numinst + 1e-10)

    ap = voc_ap(R, P)
    return P, R, score, ap


def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def roll(x, n):
    return torch.cat((x[-n:], x[:-n]))


def load_word_embeddings(emb_file, vocab):

    vocab = [v.lower() for v in vocab]

    embeds = {}
    for line in open(emb_file, 'r'):
        line = line.strip().split(' ')
        wvec = torch.FloatTensor(list(map(float, line[1:])))
        embeds[line[0]] = wvec

    # for zappos (should account for everything)
    custom_map = {
        'Faux.Fur': 'fur',
        'Faux.Leather': 'leather',
        'Full.grain.leather': 'leather',
        'Hair.Calf': 'hair',
        'Patent.Leather': 'leather',
        'Nubuck': 'leather',
        'Boots.Ankle': 'boots',
        'Boots.Knee.High': 'knee-high',
        'Boots.Mid-Calf': 'midcalf',
        'Shoes.Boat.Shoes': 'shoes',
        'Shoes.Clogs.and.Mules': 'clogs',
        'Shoes.Flats': 'flats',
        'Shoes.Heels': 'heels',
        'Shoes.Loafers': 'loafers',
        'Shoes.Oxfords': 'oxfords',
        'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers',
        'traffic_light': 'light',
        'trash_can': 'trashcan'
    }
    for k in custom_map:
        embeds[k.lower()] = embeds[custom_map[k]]

    embeds = [embeds[k] for k in vocab]
    embeds = torch.stack(embeds)
    print('loaded embeddings', embeds.size())

    return embeds
