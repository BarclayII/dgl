import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
from rec.model.pinsage import PinSage
from rec.utils import cuda
from rec.comm.sender import NodeFlowSender
from dgl import DGLGraph
from dgl.contrib.sampling import PPRBipartiteSingleSidedNeighborSampler
from validation import *

import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=1)
parser.add_argument('--sched', type=str, default='none',
                    help='learning rate scheduler (none or decay)')
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--use-feature', action='store_true')
parser.add_argument('--sgd-switch', type=int, default=-1,
                    help='The number of epoch to switch to SGD (-1 = never)')
parser.add_argument('--n-negs', type=int, default=1)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--hard-neg-prob', type=float, default=0,
                    help='Probability to sample from hard negative examples (0 = never)')
# Reddit dataset in particular is not finalized and is only for my (GQ's) internal purpose.
parser.add_argument('--cache', type=str, default='/tmp/dataset.pkl',
                    help='File to cache the postprocessed dataset object')
parser.add_argument('--dataset', type=str, default='movielens')
parser.add_argument('--host', type=str, default='localhost')
parser.add_argument('--port', type=int, default=5901)
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()

print(args)

cache_file = args.cache
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        ml = pickle.load(f)
elif args.dataset == 'movielens1m':
    from rec.datasets.movielens import MovieLens
    ml = MovieLens('/efs/quagan/movielens/ml-1m')
    with open(cache_file, 'wb') as f:
        pickle.dump(ml, f, protocol=4)
elif args.dataset == 'movielens':
    from rec.datasets.movielens import MovieLens20M
    ml = MovieLens20M('/efs/quagan/movielens/ml-20m')
    with open(cache_file, 'wb') as f:
        pickle.dump(ml, f, protocol=4)
elif args.dataset == 'cikm':
    from rec.datasets.cikm import CIKM
    ml = CIKM('/efs/quagan/diginetica/dataset-train')
    with open(cache_file, 'wb') as f:
        pickle.dump(ml, f, protocol=4)
else:
    from rec.datasets.reddit import Reddit
    ml = Reddit('/efs/quagan/2018/subm-users.pkl')
    if args.hard_neg_prob > 0:
        raise ValueError('Hard negative examples currently not supported on reddit.')
    with open(cache_file, 'wb') as f:
        pickle.dump(ml, f, protocol=4)

neighbors = []

g = ml.g
n_layers = args.layers
batch_size = args.batch_size

n_users = len(ml.user_ids)
n_items = len(ml.product_ids)


def refresh_mask():
    ml.refresh_mask()
    g_prior_edges = g.filter_edges(lambda edges: (edges.data['prior'] | edges.data['train']))
    g_train_edges = g.filter_edges(lambda edges: (edges.data['prior'] | edges.data['train']) & ~edges.data['inv'])
    g_prior_train_edges = g.filter_edges(
            lambda edges: edges.data['prior'] | edges.data['train'])
    return g_prior_edges, g_train_edges, g_prior_train_edges

cache_mask_file = cache_file + '.mask'
if os.path.exists(cache_mask_file):
    with open(cache_mask_file, 'rb') as f:
        g_prior_edges, g_train_edges, g_prior_train_edges = pickle.load(f)
else:
    g_prior_edges, g_train_edges, g_prior_train_edges = refresh_mask()
    with open(cache_mask_file, 'wb') as f:
        pickle.dump((g_prior_edges, g_train_edges, g_prior_train_edges), f)

g_prior_src, g_prior_dst = g.find_edges(g_prior_train_edges)
g_prior = DGLGraph(multigraph=True)
g_prior.add_nodes(g.number_of_nodes())
g_prior.add_edges(g_prior_src, g_prior_dst)
if args.dataset == 'cikm':
    item_query_src, item_query_dst = g.find_edges(list(range(len(ml.ratings) * 2, g.number_of_edges())))
    g_prior.add_edges(item_query_src, item_query_dst)

sender = NodeFlowSender(args.host, args.port)
#seeds = torch.arange(n_users, n_users + n_items).long()

for epoch in range(500):
    seeds = torch.LongTensor(sender.recv()) + n_users
    sampler = PPRBipartiteSingleSidedNeighborSampler(
            g_prior,
            batch_size,
            n_layers + 1,
            10,
            50,
            max_visit_counts=3,
            max_frequent_visited_nodes=10,
            seed_nodes=seeds,
            restart_prob=0.5,
            prefetch=False,
            add_self_loop=True,
            shuffle=False,
            num_workers=25)

    with tqdm.tqdm(sampler) as tq:
        for i, nodeflow in enumerate(tq):
            sender.send(
                    nodeflow,
                    seeds[i * batch_size:(i + 1) * batch_size].numpy() - n_users)
    print('Completing')
    sender.complete()
