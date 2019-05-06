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
parser.add_argument('--port', type=int, default=5902)
args = parser.parse_args()

print(args)

cache_file = args.cache
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        ml = pickle.load(f)
elif args.dataset == 'movielens':
    from rec.datasets.movielens import MovieLens
    ml = MovieLens('./ml-1m')
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

if args.dataset == 'movielens':
    neighbors = ml.user_neighbors + ml.product_neighbors
else:
    neighbors = []

g = ml.g
n_layers = args.layers
batch_size = 16

n_negs = args.n_negs
hard_neg_prob = args.hard_neg_prob

n_users = len(ml.user_ids)
n_items = len(ml.product_ids)


def refresh_mask():
    ml.refresh_mask()
    g_prior_edges = g.filter_edges(lambda edges: edges.data['prior'])
    g_train_edges = g.filter_edges(lambda edges: edges.data['train'] & ~edges.data['inv'])
    g_prior_train_edges = g.filter_edges(
            lambda edges: edges.data['prior'] | edges.data['train'])
    return g_prior_edges, g_train_edges, g_prior_train_edges

def find_negs(dst, ml, neighbors, hard_neg_prob, n_negs):
    dst_neg = []
    for i in range(len(dst)):
        if np.random.rand() < hard_neg_prob:
            nb = torch.LongTensor(neighbors[dst[i].item()])
            mask = ~(g.has_edges_between(nb, src[i].item()).byte())
            dst_neg.append(np.random.choice(nb[mask].numpy(), n_negs))
        else:
            dst_neg.append(np.random.randint(
                len(ml.user_ids), len(ml.user_ids) + len(ml.product_ids), n_negs))
    dst_neg = torch.LongTensor(dst_neg)
    return dst_neg


cache_mask_file = cache_file + '.mask'
if os.path.exists(cache_mask_file):
    with open(cache_mask_file, 'rb') as f:
        g_prior_edges, g_train_edges, g_prior_train_edges = pickle.load(f)
else:
    g_prior_edges, g_train_edges, g_prior_train_edges = refresh_mask()
    with open(cache_mask_file, 'wb') as f:
        pickle.dump((g_prior_edges, g_train_edges, g_prior_train_edges), f)



sender = NodeFlowSender(args.host, args.port)

for epoch in range(500):
    g_prior_src, g_prior_dst = g.find_edges(g_prior_edges)
    g_prior = DGLGraph(multigraph=True)
    g_prior.add_nodes(g.number_of_nodes())
    g_prior.add_edges(g_prior_src, g_prior_dst)
    g_prior.ndata.update({k: v for k, v in g.ndata.items()})
    if args.dataset == 'cikm':
        item_query_src, item_query_dst = g.find_edges(list(range(len(ml.ratings) * 2, g.number_of_edges())))
        g_prior.add_edges(item_query_src, item_query_dst)

    # prepare seed nodes
    #edge_shuffled = g_train_edges[torch.randperm(g_train_edges.shape[0])]
    edge_shuffled = g_train_edges[torch.LongTensor(sender.recv())]
    src, dst = g.find_edges(edge_shuffled)
    dst_neg = find_negs(dst, ml, neighbors, hard_neg_prob, n_negs)
    # expand if we have multiple negative products for each training pair
    dst = dst.view(-1, 1).expand_as(dst_neg).flatten()
    src = src.view(-1, 1).expand_as(dst_neg).flatten()
    dst_neg = dst_neg.flatten()
    # Check whether these nodes have predecessors in *prior* graph.  Filter out
    # those who don't.
    mask = (g_prior.in_degrees(dst_neg) > 0) & \
           (g_prior.in_degrees(dst) > 0) & \
           (g_prior.in_degrees(src) > 0)
    src = src[mask]
    dst = dst[mask]
    dst_neg = dst_neg[mask]
    edge_shuffled = edge_shuffled[mask]
    # Chop the users, items and negative items into batches and reorganize
    # them into seed nodes.
    # Note that the batch size of DGL sampler here is 3 times our batch size,
    # since the sampler is handling 3 nodes per training example.
    edge_batches = edge_shuffled.split(batch_size)
    src_batches = src.split(batch_size)
    dst_batches = dst.split(batch_size)
    dst_neg_batches = dst_neg.split(batch_size)

    if args.dataset == 'cikm':
        anonymous_indices = torch.LongTensor(sender.recv())
        anonymous_dst_mask = np.isin(
                ml.anonymous_ratings.iloc[anonymous_indices.numpy()]['product_id'].values,
                np.array(ml.product_ids))
        anonymous_dst_mask = torch.ByteTensor(anonymous_dst_mask)
        anonymous_indices = anonymous_indices[anonymous_dst_mask]
        anonymous_product_id = ml.anonymous_ratings.iloc[anonymous_indices.numpy()]['product_id'].values
        num_users = len(ml.user_ids)
        dst = [num_users + ml.product_ids_invmap[i]
               for i in anonymous_product_id]
        dst = torch.LongTensor(dst)
        dst_neg = find_negs(dst, ml, neighbors, hard_neg_prob, n_negs)

        dst = dst.view(-1, 1).expand_as(dst_neg).flatten()
        dst_neg = dst_neg.flatten()

        mask = (g_prior.in_degrees(dst_neg) > 0) & (g_prior.in_degrees(dst) > 0)
        dst = dst[mask]
        dst_neg = dst_neg[mask]
        anonymous_indices = anonymous_indices[mask]

        anonymous_dst_batches = dst.split(batch_size)
        anonymous_dst_neg_batches = dst_neg.split(batch_size)
        anonymous_batches = anonymous_indices.split(batch_size)

    seed_nodes = []
    for i in range(len(src_batches)):
        seed_nodes.append(src_batches[i])
        seed_nodes.append(dst_batches[i])
        seed_nodes.append(dst_neg_batches[i])
        if args.dataset == 'cikm':
            seed_nodes.append(anonymous_dst_batches[i])
            seed_nodes.append(anonymous_dst_neg_batches[i])
    seed_nodes = torch.cat(seed_nodes)

    sampler = PPRBipartiteSingleSidedNeighborSampler(
            g_prior,
            batch_size * (3 if args.dataset != 'cikm' else 5),
            n_layers + 1,
            10,
            20,
            seed_nodes=seed_nodes,
            restart_prob=0.5,
            prefetch=True,
            add_self_loop=True,
            num_workers=20)
    sampler_iter = iter(sampler)

    with tqdm.tqdm(sampler) as tq:
        for batch_id, nodeflow in enumerate(tq):
            edges = edge_batches[batch_id]
            src = src_batches[batch_id]
            dst = dst_batches[batch_id]
            dst_neg = dst_neg_batches[batch_id]
            if args.dataset == 'cikm':
                # train an additional batch of anonymous queries
                row_indices = anonymous_batches[batch_id].numpy()
                anonymous_dst = anonymous_dst_batches[batch_id]
                anonymous_dst_neg = anonymous_dst_neg_batches[batch_id]

            sender.send(
                    nodeflow,
                    (edges.numpy(),
                     src.numpy(),
                     dst.numpy(),
                     dst_neg.numpy(),
                     row_indices,
                     anonymous_dst.numpy(),
                     anonymous_dst_neg.numpy()))
