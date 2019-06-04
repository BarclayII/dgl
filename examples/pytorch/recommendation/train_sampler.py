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
from dgl.contrib.sampling import PPRBipartiteSingleSidedNeighborSampler, NeighborSampler
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

n_negs = args.n_negs
hard_neg_prob = args.hard_neg_prob

n_users = len(ml.user_ids)
n_items = len(ml.product_ids)


def refresh_mask():
    ml.refresh_mask()
    g_prior_edges = g.filter_edges(lambda edges: (edges.data['prior'] | edges.data['train']))
    g_train_edges = g.filter_edges(lambda edges: (edges.data['prior'] | edges.data['train']) & ~edges.data['inv'])
    g_prior_train_edges = g.filter_edges(
            lambda edges: edges.data['prior'] | edges.data['train'])
    return g_prior_edges, g_train_edges, g_prior_train_edges

def find_negs(dst, ml, neighbors, hard_neg_prob, n_negs):
    dst_neg = []
    for i in range(len(dst)):
        if np.random.rand() < hard_neg_prob:
            if args.dataset == 'cikm':
                query_ids = ml.query_candidates[
                    ml.query_candidates['product_id'] == ml.product_ids[dst[i].item() - len(ml.users)]]['query_id'].values
                query_id = np.random.choice(query_ids)
                product_candidates = ml.query_candidates[
                    ml.query_candidates['query_id'] == query_id]['product_id'].values
                selected_products = np.random.choice(product_candidates, n_negs)
                dst_neg.append(np.array([ml.product_ids_invmap[i] + len(ml.users) for i in selected_products]))
            else:
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
    g_prior.readonly()
    if args.dataset == 'cikm':
        item_query_src, item_query_dst = g.find_edges(list(range(len(ml.ratings) * 2, g.number_of_edges())))
        g_prior.add_edges(item_query_src, item_query_dst)

    # prepare seed nodes
    #edge_shuffled = g_train_edges[torch.randperm(g_train_edges.shape[0])]
    edge_shuffled = g_train_edges[torch.LongTensor(sender.recv())]
    src, dst = g.find_edges(edge_shuffled)
    dst_neg = find_negs(dst, ml, neighbors, hard_neg_prob, n_negs)
    dst_neg = dst_neg.flatten()
    # Chop the users, items and negative items into batches and reorganize
    # them into seed nodes.
    # Note that the batch size of DGL sampler here is 3 times our batch size,
    # since the sampler is handling 3 nodes per training example.
    edge_batches = edge_shuffled.split(batch_size)
    src_batches = src.split(batch_size)
    dst_batches = dst.split(batch_size)
    dst_neg_batches = dst_neg.split(batch_size * n_negs)

    seed_nodes = []
    for i in range(len(src_batches)):
        seed_nodes.append(src_batches[i])
        seed_nodes.append(dst_batches[i])
        seed_nodes.append(dst_neg_batches[i])
    seed_nodes = torch.cat(seed_nodes)

    #sampler = PPRBipartiteSingleSidedNeighborSampler(
    #        g_prior,
    #        batch_size * (2 + n_negs),
    #        n_layers + 1,
    #        10,
    #        50,
    #        max_visit_counts=3,
    #        max_frequent_visited_nodes=10,
    #        seed_nodes=seed_nodes,
    #        restart_prob=0.5,
    #        prefetch=True,
    #        add_self_loop=True,
    #        num_workers=20)
    sampler = NeighborSampler(
            g_prior,
            batch_size * (2 + n_negs),
            5,
            n_layers,
            seed_nodes=seed_nodes,
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

            sender.send(
                    nodeflow,
                    (edges.numpy(),
                     src.numpy(),
                     dst.numpy(),
                     dst_neg.numpy()))
    sender.complete()
