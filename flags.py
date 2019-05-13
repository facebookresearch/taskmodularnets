# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mitstates', help='mitstates|zappos')
parser.add_argument(
    '--data_dir', default='data/mit-states/', help='data root dir')
parser.add_argument(
    '--cv_dir', default='models', help='dir to save checkpoints to')
parser.add_argument(
    '--load', default=None, help='path to checkpoint to load from')
parser.add_argument(
    '--name', default='temp', help='Name of exp used to name models')
# model parameters
parser.add_argument(
    '--model',
    default='modularpretrained',
    help='Supports only modularpretrained right now')
parser.add_argument('--splitname', default='compositional-split-natural')
parser.add_argument('--test_set', default='val', help='val|test')
parser.add_argument(
    '--compose_type',
    default='nn',
    help='Form of gating function (nn: Simple Neural Network)')
parser.add_argument(
    '--emb_dim', type=int, default=16, help='output dimension of each module')
parser.add_argument(
    '--pair_dropout',
    type=float,
    default=0.0,
    help='Each epoch drop this fraction of train pairs')
parser.add_argument(
    '--pair_dropout_epoch',
    type=int,
    default=1,
    help='Shuffle pair dropout every N epochs')
parser.add_argument(
    '--neg_ratio',
    type=float,
    default=0.25,
)
parser.add_argument(
    '--randinit',
    action='store_true',
    default=False,
)
parser.add_argument(
    '--test_only',
    action='store_true',
    default=False,
)
parser.add_argument(
    '--nlayers', type=int, default=3, help='number of modular layers')
parser.add_argument(
    '--nmods', type=int, default=24, help='number of mods per layer')
parser.add_argument(
    '--glove_init',
    action='store_true',
    default=False,
    help='initialize inputs with word vectors')
parser.add_argument(
    '--clf_init',
    action='store_true',
    default=False,
    help='initialize inputs with SVM weights')
parser.add_argument(
    '--static_inp',
    action='store_true',
    default=False,
    help='do not optimize input representations')
parser.add_argument(
    '--subset',
    action='store_true',
    default=False,
    help='test on a 1000 image subset')
parser.add_argument('--logembed', action='store_true', default=False)
parser.add_argument('--adam', action='store_true', default=False)

# regularizers
parser.add_argument('--lambda_aux', type=float, default=0.0)
parser.add_argument('--lambda_gating_aux', type=float, default=0.0)
parser.add_argument('--lambda_inv', type=float, default=0.0)
parser.add_argument('--lambda_comm', type=float, default=0.0)
parser.add_argument('--lambda_ant', type=float, default=0.0)

# optimization
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--topk', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lrg', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=5e-5)
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--eval_val_every', type=int, default=1)
parser.add_argument('--print_every', type=int, default=10)
parser.add_argument('--max_epochs', type=int, default=300)
parser.add_argument(
    '--num_negs',
    type=int,
    default=200,
    help='Number of negatives to sample per positive')
parser.add_argument(
    '--embed_rank',
    type=int,
    default=64,
    help='intermediate dimension in the gating model')
parser.add_argument(
    '--steps', help='epochs to step lr', nargs='+', default=[2000], type=int)
