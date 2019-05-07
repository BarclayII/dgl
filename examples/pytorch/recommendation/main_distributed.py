import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
from rec.model.pinsage import PinSage
from rec.utils import cuda
from rec.comm.receiver import NodeFlowReceiver
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

if args.dataset == 'cikm':
    ml.query_tokens = cuda(ml.query_tokens)

_compute_validation = {
        'movielens': compute_validation_ml,
        'reddit': compute_validation_ml,
        'cikm': compute_validation_cikm,
        }
compute_validation = _compute_validation[args.dataset]

g = ml.g

n_hidden = 100
n_layers = args.layers
batch_size = 16
margin = 0.9

n_negs = args.n_negs
hard_neg_prob = args.hard_neg_prob

sched_lambda = {
        'none': lambda epoch: 1,
        'decay': lambda epoch: max(0.98 ** epoch, 1e-4),
        }
loss_func = {
        'hinge': lambda diff: (diff + margin).clamp(min=0).mean(),
        'bpr': lambda diff: (1 - torch.sigmoid(-diff)).mean(),
        }

if args.dataset == 'cikm':
    emb_tokens = nn.Embedding(
            max(g.ndata['tokens'].max().item(), ml.query_tokens.max().item()) + 1,
            n_hidden)
    emb = {'tokens': emb_tokens}
else:
    emb = {}

model = cuda(PinSage(
    [n_hidden] * (n_layers + 1),
    use_feature=args.use_feature,
    G=g,
    emb=emb,
    ))

opt = getattr(torch.optim, args.opt)(
        list(model.parameters()),
        lr=args.lr)
sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_lambda[args.sched])


def cast_ppr_weight(nodeflow):
    for i in range(nodeflow.num_blocks):
        nodeflow.apply_block(i, lambda x: {'ppr_weight': cuda(x.data['ppr_weight']).float()})


def forward(model, nodeflow, train=True):
    if train:
        return model(nodeflow, None)
    else:
        with torch.no_grad():
            return model(nodeflow, None)


train_sampler = NodeFlowReceiver(5902)
train_sampler.waitfor(4)

def runtrain(g_prior_edges, g_train_edges, train):
    global opt
    if train:
        model.train()
    else:
        model.eval()

    g_prior_src, g_prior_dst = g.find_edges(g_prior_edges)
    g_prior = DGLGraph(multigraph=True)
    g_prior.add_nodes(g.number_of_nodes())
    g_prior.add_edges(g_prior_src, g_prior_dst)
    g_prior.ndata.update({k: v for k, v in g.ndata.items()})
    if args.dataset == 'cikm':
        item_query_src, item_query_dst = g.find_edges(list(range(len(ml.ratings) * 2, g.number_of_edges())))
        g_prior.add_edges(item_query_src, item_query_dst)

    edge_shuffled = torch.randperm(g_train_edges.shape[0])
    n_batches = len(edge_shuffled.split(batch_size))
    train_sampler.distribute(edge_shuffled.numpy())
    if args.dataset == 'cikm':
        anonymous_indices = np.random.permutation(len(ml.anonymous_ratings))
        train_sampler.distribute(anonymous_indices)

    train_sampler.set_parent_graph(g_prior)
    train_sampler_iter = iter(train_sampler)

    with tqdm.trange(train_sampler_iter) as tq:
        sum_loss = 0
        sum_acc = 0
        count = 0
        for batch_id, (nodeflow, aux_data) in enumerate(tq):
            # TODO: got stuck on making this sampling process distributed...
            # SAMPLING PROCESS BEGIN
            # find the source nodes (users), destination nodes (positive products), and
            # negative destination nodes (negative products) in *original* graph.
            edges, src, dst, dst_neg = aux_data[:4]
            edges = torch.LongTensor(edges)
            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst)
            dst_neg = torch.LongTensor(dst_neg)
            src_size = dst_size = dst_neg_size = src.shape[0]
            count += src_size

            if args.dataset == 'cikm':
                # train an additional batch of anonymous queries
                row_indices, anonymous_dst, anonymous_dst_neg = aux_data[4:7]
                anonymous_dst = torch.LongTensor(anonymous_dst)
                anonymous_dst_neg = torch.LongTensor(anonymous_dst_neg)
                anon_dst_size = anon_dst_neg_size = anonymous_dst.shape[0]
                count += anon_dst_size

            # Generate a NodeFlow given the sources/destinations/negative destinations.  We need
            # GCN output of those nodes.
            nodeset = torch.cat([dst, dst_neg] if args.dataset != 'cikm' else
                                [dst, dst_neg, anonymous_dst, anonymous_dst_neg])
            # SAMPLING PROCESS END
            # copy features from parent graph
            nodeflow.copy_from_parent()
            cast_ppr_weight(nodeflow)

            # The features on nodeflow is stored on CPUs for now.  We copy them to GPUs
            # in model.forward().
            node_output = forward(model, nodeflow, train)
            output_idx = nodeflow.map_from_parent_nid(-1, nodeset)
            h = node_output[output_idx]
            if args.dataset != 'cikm':
                h_dst, h_dst_neg = h.split([dst_size, dst_neg_size])
            else:
                h_dst, h_dst_neg, h_anon_dst, h_anon_dst_neg = h.split(
                        [dst_size, dst_neg_size, anon_dst_size, anon_dst_neg_size])
            h_src = model.emb['nid'](cuda(g.nodes[src].data['nid']))

            # For CIKM, add query/category embeddings to user embeddings.
            # This is somehow inspired by TransE
            if args.dataset == 'cikm':
                tokens_idx = cuda(g.edges[edges].data['tokens_idx'])
                tokens = ml.query_tokens[tokens_idx]
                category = cuda(g.edges[edges].data['category'])
                h_tokens = model.emb['tokens'](tokens).mean(1)
                h_category = model.emb['category'](category)
                h_src = h_src + h_tokens + h_category

                h_dst = torch.cat([h_dst, h_anon_dst])
                h_dst_neg = torch.cat([h_dst_neg, h_anon_dst_neg])

                tokens = ml.anonymous_ratings.iloc[row_indices]['tokens'].tolist()
                tokens = cuda(torch.LongTensor(tokens))
                category = ml.anonymous_ratings.iloc[row_indices]['categoryId'].tolist()
                category = cuda(torch.LongTensor(category))
                h_tokens = model.emb['tokens'](tokens).mean(1)
                h_category = model.emb['category'](category)
                h_anon_src = h_tokens + h_category
                h_src = torch.cat([h_src, h_anon_src])

            diff = (h_src * (h_dst_neg - h_dst)).sum(1)
            loss = loss_func[args.loss](diff)
            acc = (diff < 0).sum()
            assert loss.item() == loss.item()

            grad_sqr_norm = 0
            if train:
                opt.zero_grad()
                loss.backward()
                #for name, p in model.named_parameters():
                    #assert (p.grad != p.grad).sum() == 0
                    #grad_sqr_norm += p.grad.norm().item() ** 2
                grad_sqr_norm = 0
                opt.step()

            sum_loss += loss.item()
            sum_acc += acc.item() / n_negs
            avg_loss = sum_loss / (batch_id + 1)
            avg_acc = sum_acc / count
            tq.set_postfix({'loss': '%.6f' % loss.item(),
                            'avg_loss': '%.3f' % avg_loss,
                            'avg_acc': '%.3f' % avg_acc,
                            'grad_norm': '%.6f' % np.sqrt(grad_sqr_norm)})
            tq.update()

    return avg_loss, avg_acc

valid_sampler = NodeFlowReceiver(5901)
valid_sampler.waitfor(4)

def runtest(g_prior_edges, validation=True):
    model.eval()

    n_users = len(ml.user_ids)
    n_items = len(ml.product_ids)

    g_prior_src, g_prior_dst = g.find_edges(g_prior_edges)
    g_prior = DGLGraph(multigraph=True)
    g_prior.add_nodes(g.number_of_nodes())
    g_prior.add_edges(g_prior_src, g_prior_dst)
    g_prior.ndata.update({k: v for k, v in g.ndata.items()})
    if args.dataset == 'cikm':
        item_query_src, item_query_dst = g.find_edges(list(range(len(ml.ratings) * 2, g.number_of_edges())))
        g_prior.add_edges(item_query_src, item_query_dst)

    valid_sampler.set_parent_graph(g_prior)
    valid_sampler.distribute(np.arange(n_items))
    valid_sampler_iter = iter(valid_sampler)

    hs = []
    auxs = []
    with torch.no_grad():
        with tqdm.tqdm(valid_sampler_iter) as tq:
            for nodeflow, aux_data in tq:
                nodeflow.copy_from_parent()
                cast_ppr_weight(nodeflow)
                h = forward(model, nodeflow, False)
                hs.append(h)
                auxs.append(torch.LongTensor(aux_data))
    h = torch.cat(hs, 0)
    auxs = torch.cat(auxs, 0)
    assert (np.sort(auxs.numpy()) == np.arange(n_items)).all()
    h = h[auxs]     # reorder h
    h = torch.cat([
        model.emb['nid'](cuda(torch.arange(0, n_users).long() + 1)),
        h], 0)

    return compute_validation(ml, h, model)


def train():
    global opt, sched
    best_mrr = 0

    cache_mask_file = cache_file + '.mask'
    if os.path.exists(cache_mask_file):
        with open(cache_mask_file, 'rb') as f:
            g_prior_edges, g_train_edges, g_prior_train_edges = pickle.load(f)
    else:
        g_prior_edges, g_train_edges, g_prior_train_edges = refresh_mask()
        with open(cache_mask_file, 'wb') as f:
            pickle.dump((g_prior_edges, g_train_edges, g_prior_train_edges), f)

    for epoch in range(500):
        print('Epoch %d train' % epoch)
        runtrain(g_prior_edges, g_train_edges, True)

        if epoch == args.sgd_switch:
            opt = torch.optim.SGD(model.parameters(), lr=0.6)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_lambda['decay'])
        elif epoch < args.sgd_switch:
            sched.step()

        print('Epoch %d validation' % epoch)
        if 1:
            with torch.no_grad():
                valid_mrr = runtest(g_prior_train_edges, True)
                if best_mrr < valid_mrr.mean():
                    best_mrr = valid_mrr.mean()
                    torch.save(model.state_dict(), 'model.pt')
            print(pd.Series(valid_mrr).describe())
            print('Epoch %d test' % epoch)
            with torch.no_grad():
                test_mrr = runtest(g_prior_train_edges, False)
            print(pd.Series(test_mrr).describe())


if __name__ == '__main__':
    train()
