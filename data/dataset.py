# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch.utils.data as tdata
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import tqdm
import torchvision.models as tmodels
import torch.nn as nn
from torch.autograd import Variable
import torch
import bz2
from utils import utils
import h5py
import pdb
import archs
import itertools
import os
import collections
import scipy.io
from sklearn.model_selection import train_test_split


class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


def imagenet_transform(phase):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'test' or phase == 'val':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform


#------------------------------------------------------------------------------------------------------------------------------------#


class CompositionDataset(tdata.Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split',
            subset=False,
            num_negs=1,
            pair_dropout=0.0,
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.num_negs = num_negs
        self.pair_dropout = pair_dropout

        self.feat_dim = None
        self.transform = imagenet_transform(phase)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data
        if subset:
            ind = np.arange(len(self.data))
            ind = ind[::len(ind) // 1000]
            self.data = [self.data[i] for i in ind]

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        # fix later -- affordance thing
        # return {object: all attrs that occur with obj}
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [
                attr
                for (_, attr,
                     obj) in self.train_data + self.val_data + self.test_data
                if obj == _obj
            ]
            self.obj_affordance[_obj] = sorted(list(set(candidates)))

            candidates = [
                attr for (_, attr, obj) in self.train_data if obj == _obj
            ]
            self.train_obj_affordance[_obj] = sorted(list(set(candidates)))

        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

    def reset_dropout(self):
        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        shuffled_ind = np.random.permutation(len(self.train_pairs))
        n_pairs = int((1 - self.pair_dropout) * len(self.train_pairs))
        self.sample_pairs = [
            self.train_pairs[pi] for pi in shuffled_ind[:n_pairs]
        ]
        print('Using {} pairs out of {} pairs right now'.format(
            n_pairs, len(self.train_pairs)))
        self.sample_indices = [
            i for i in range(len(self.data))
            if (self.data[i][1], self.data[i][2]) in self.sample_pairs
        ]
        print('Using {} images out of {} images right now'.format(
            len(self.sample_indices), len(self.data)))

    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def sample_negative(self, attr, obj):
        new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]
        if new_attr == attr and new_obj == obj:
            return self.sample_negative(attr, obj)
        return (self.attr2idx[new_attr], self.obj2idx[new_obj])

    def sample_affordance(self, attr, obj):
        new_attr = np.random.choice(self.obj_affordance[obj])
        if new_attr == attr:
            return self.sample_affordance(attr, obj)
        return self.attr2idx[new_attr]

    def sample_train_affordance(self, attr, obj):
        new_attr = np.random.choice(self.train_obj_affordance[obj])
        if new_attr == attr:
            return self.sample_train_affordance(attr, obj)
        return self.attr2idx[new_attr]

    def __getitem__(self, index):
        index = self.sample_indices[index]
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        data = [
            img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr,
                                                                        obj)]
        ]

        if self.phase == 'train':
            all_neg_attrs = []
            all_neg_objs = []
            for _ in range(self.num_negs):
                neg_attr, neg_obj = self.sample_negative(
                    attr, obj)  # negative example for triplet loss
                all_neg_objs.append(neg_obj)
                all_neg_attrs.append(neg_attr)
            neg_attr = torch.LongTensor(all_neg_attrs)
            neg_obj = torch.LongTensor(all_neg_objs)
            inv_attr = self.sample_train_affordance(
                attr, obj)  # attribute for inverse regularizer
            comm_attr = self.sample_affordance(
                inv_attr, obj)  # attribute for commutative regularizer
            data += [neg_attr, neg_obj, inv_attr, comm_attr]
        return data

    def __len__(self):
        return len(self.sample_indices)


#------------------------------------------------------------------------------------------------------------------------------------#


class CompositionDatasetActivations(CompositionDataset):
    def __init__(
            self,
            root,
            phase,
            split,
            subset=False,
            num_negs=1,
            pair_dropout=0.0,
    ):
        super(CompositionDatasetActivations, self).__init__(
            root,
            phase,
            split,
            subset=subset,
            num_negs=num_negs,
            pair_dropout=pair_dropout,
        )

        # precompute the activations -- weird. Fix pls
        feat_file = '%s/features.t7' % root
        if not os.path.exists(feat_file):
            with torch.no_grad():
                self.generate_features(feat_file)

        activation_data = torch.load(feat_file)
        self.activations = dict(
            zip(activation_data['files'], activation_data['features']))
        self.feat_dim = activation_data['features'].size(1)

        print('%d activations loaded' % (len(self.activations)))

    def generate_features(self, out_file):

        data = self.train_data + self.val_data + self.test_data
        transform = imagenet_transform('test')
        feat_extractor = tmodels.resnet18(pretrained=True)
        feat_extractor.fc = nn.Sequential()
        feat_extractor.eval().cuda()

        image_feats = []
        image_files = []
        for chunk in tqdm.tqdm(
                utils.chunks(data, 512), total=len(data) // 512):
            files, attrs, objs = zip(*chunk)
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            feats = feat_extractor(torch.stack(imgs, 0).cuda())
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print('features for %d images generated' % (len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, out_file)

    def __getitem__(self, index):
        data = super(CompositionDatasetActivations, self).__getitem__(index)
        index = self.sample_indices[index]
        image, attr, obj = self.data[index]
        data[0] = self.activations[image]
        return data
