import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
from rec.model.pinsage import PinSage
from rec.utils import cuda
from dgl import DGLGraph
from dgl.contrib.sampling import PPRBipartiteSingleSidedNeighborSampler

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
args = parser.parse_args()

print(args)

cache_file = args.cache
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        ml = pickle.load(f)
else:
    from rec.datasets.reddit import Reddit
    ml = Reddit('/efs/quagan/2018/subm-users.pkl')
    if args.hard_neg_prob > 0:
        raise ValueError('Hard negative examples currently not supported on reddit.')
    with open(cache_file, 'wb') as f:
        pickle.dump(ml, f, protocol=4)

g = ml.g

n_hidden = 100
n_layers = args.layers
batch_size = 32
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

model = cuda(PinSage(
    [n_hidden] * (n_layers + 1),
    use_feature=args.use_feature,
    G=g,
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

def runtrain(g_prior_edges, g_train_edges, train):
    global opt
    if train:
        model.train()
    else:
        model.eval()

    g_prior_src, g_prior_dst = g.find_edges(g_prior_edges)
    g_prior = DGLGraph()
    g_prior.add_nodes(g.number_of_nodes())
    g_prior.add_edges(g_prior_src, g_prior_dst)
    g_prior.ndata.update({k: v for k, v in g.ndata.items()})

    # prepare seed nodes
    edge_shuffled = g_train_edges[torch.randperm(g_train_edges.shape[0])]
    src, dst = g.find_edges(edge_shuffled)
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
    # Chop the users, items and negative items into batches and reorganize
    # them into seed nodes.
    # Note that the batch size of DGL sampler here is 3 times our batch size,
    # since the sampler is handling 3 nodes per training example.
    src_batches = src.split(batch_size)
    dst_batches = dst.split(batch_size)
    dst_neg_batches = dst_neg.split(batch_size)
    seed_nodes = []
    for i in range(len(src_batches)):
        seed_nodes.append(src_batches[i])
        seed_nodes.append(dst_batches[i])
        seed_nodes.append(dst_neg_batches[i])
    seed_nodes = torch.cat(seed_nodes)

    sampler = PPRBipartiteSingleSidedNeighborSampler(
            g_prior,
            batch_size * 3,
            n_layers + 1,
            10,
            20,
            seed_nodes=seed_nodes,
            restart_prob=0.5,
            prefetch=True,
            add_self_loop=True,
            num_workers=20)
    sampler_iter = iter(sampler)

    with tqdm.trange(len(src_batches)) as tq:
        sum_loss = 0
        sum_acc = 0
        count = 0
        for batch_id in tq:
            # TODO: got stuck on making this sampling process distributed...
            # SAMPLING PROCESS BEGIN
            # find the source nodes (users), destination nodes (positive products), and
            # negative destination nodes (negative products) in *original* graph.
            src = src_batches[batch_id]
            dst = dst_batches[batch_id]
            dst_neg = dst_neg_batches[batch_id]
            src_size = dst_size = dst_neg_size = src.shape[0]
            count += src_size

            # Generate a NodeFlow given the sources/destinations/negative destinations.  We need
            # GCN output of those nodes.
            nodeset = torch.cat([src, dst, dst_neg])
            nodeflow = next(sampler_iter)
            for i in range(nodeflow.num_layers - 1):
                assert np.isin(nodeflow.layer_parent_nid(i + 1).numpy(),
                        nodeflow.layer_parent_nid(i).numpy()).all()
            last_nid = nodeflow.layer_parent_nid(-1).numpy()
            assert np.isin(src.numpy(), last_nid).all()
            assert np.isin(dst.numpy(), last_nid).all()
            assert np.isin(dst_neg.numpy(), last_nid).all()
            # SAMPLING PROCESS END
            # copy features from parent graph
            nodeflow.copy_from_parent()
            cast_ppr_weight(nodeflow)

            # The features on nodeflow is stored on CPUs for now.  We copy them to GPUs
            # in model.forward().
            node_output = forward(model, nodeflow, train)
            output_idx = nodeflow.map_from_parent_nid(-1, nodeset)
            h = node_output[output_idx]
            h_src, h_dst, h_dst_neg = h.split([src_size, dst_size, dst_neg_size])

            diff = (h_src * (h_dst_neg - h_dst)).sum(1)
            loss = loss_func[args.loss](diff)
            acc = (diff < 0).sum()
            assert loss.item() == loss.item()

            grad_sqr_norm = 0
            if train:
                opt.zero_grad()
                loss.backward()
                for name, p in model.named_parameters():
                    assert (p.grad != p.grad).sum() == 0
                    grad_sqr_norm += p.grad.norm().item() ** 2
                opt.step()

            sum_loss += loss.item()
            sum_acc += acc.item() / n_negs
            avg_loss = sum_loss / (batch_id + 1)
            avg_acc = sum_acc / count
            tq.set_postfix({'loss': '%.6f' % loss.item(),
                            'avg_loss': '%.3f' % avg_loss,
                            'avg_acc': '%.3f' % avg_acc,
                            'grad_norm': '%.6f' % np.sqrt(grad_sqr_norm)})

    return avg_loss, avg_acc


def runtest(g_prior_edges, validation=True):
    model.eval()

    n_users = len(ml.user_ids)
    n_items = len(ml.product_ids)

    g_prior_src, g_prior_dst = g.find_edges(g_prior_edges)
    g_prior = DGLGraph()
    g_prior.add_nodes(g.number_of_nodes())
    g_prior.add_edges(g_prior_src, g_prior_dst)
    g_prior.ndata.update({k: v for k, v in g.ndata.items()})
    sampler = PPRBipartiteSingleSidedNeighborSampler(
            g_prior,
            batch_size,
            n_layers + 1,
            10,
            20,
            restart_prob=0.5,
            prefetch=False,
            add_self_loop=True,
            num_workers=20)

    hs = []
    with torch.no_grad():
        for nodeflow in tqdm.tqdm(sampler):
            nodeflow.copy_from_parent()
            cast_ppr_weight(nodeflow)
            h = forward(model, nodeflow, False)
            hs.append(h)
    h = torch.cat(hs, 0)

    rr = []

    with torch.no_grad():
        with tqdm.trange(n_users) as tq:
            for u_nid in tq:
                uid = ml.user_ids[u_nid]
                pids_exclude = ml.ratings[
                        (ml.ratings['user_id'] == uid) &
                        (ml.ratings['train'] | ml.ratings['test' if validation else 'valid'])
                        ]['product_id'].values
                pids_candidate = ml.ratings[
                        (ml.ratings['user_id'] == uid) &
                        ml.ratings['valid' if validation else 'test']
                        ]['product_id'].values
                pids = np.setdiff1d(ml.product_ids, pids_exclude)
                p_nids = np.array([ml.product_ids_invmap[pid] for pid in pids])
                p_nids_candidate = np.array([ml.product_ids_invmap[pid] for pid in pids_candidate])

                dst = torch.from_numpy(p_nids) + n_users
                src = torch.zeros_like(dst).fill_(u_nid)
                h_dst = h[dst]
                h_src = h[src]

                score = (h_src * h_dst).sum(1)
                score_sort_idx = score.sort(descending=True)[1].cpu().numpy()

                rank_map = {v: i for i, v in enumerate(p_nids[score_sort_idx])}
                rank_candidates = np.array([rank_map[p_nid] for p_nid in p_nids_candidate])
                rank = 1 / (rank_candidates + 1)
                rr.append(rank.mean())
                tq.set_postfix({'rank': rank.mean()})

    return np.array(rr)


def refresh_mask():
    ml.refresh_mask()
    g_prior_edges = g.filter_edges(lambda edges: edges.data['prior'])
    g_train_edges = g.filter_edges(lambda edges: edges.data['train'] & ~edges.data['inv'])
    g_prior_train_edges = g.filter_edges(
            lambda edges: edges.data['prior'] | edges.data['train'])
    return g_prior_edges, g_train_edges, g_prior_train_edges


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
        print('Epoch %d validation' % epoch)
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

        print('Epoch %d train' % epoch)
        runtrain(g_prior_edges, g_train_edges, True)

        if epoch == args.sgd_switch:
            opt = torch.optim.SGD(model.parameters(), lr=0.6)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_lambda['decay'])
        elif epoch < args.sgd_switch:
            sched.step()


if __name__ == '__main__':
    train()
