import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from ogb.nodeproppred import DglNodePropPredDataset

class SAGEConvWithCV(nn.Module):
    def __init__(self, in_feats, out_feats, activation, alpha=0.1, dropout=0):
        super().__init__()
        self.W = nn.Linear(in_feats * 2, out_feats)
        self.activation = activation
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, block, H, HBar=None):
        if self.training:
            with block.local_scope():
                H_src, H_dst = H
                agg_HBar_dst = HBar
                block.dstdata['agg_hbar'] = agg_HBar_dst
                block.srcdata['hdelta'] = H_src
                block.update_all(fn.copy_u('hdelta', 'm'), fn.mean('m', 'hdelta_new'))
                HDelta_new = block.dstdata['hdelta_new']
                h_neigh = (1 - self.alpha) * (agg_HBar_dst - HDelta_new).detach() + HDelta_new
                h = self.W(self.dropout(th.cat([H_dst, h_neigh], 1)))
                if self.activation is not None:
                    h = self.activation(h)
                return h, h_neigh
        else:
            with block.local_scope():
                H_src, H_dst = H
                block.srcdata['h'] = H_src
                block.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_new'))
                h_neigh = block.dstdata['h_new']
                h = self.W(th.cat([H_dst, h_neigh], 1))
                if self.activation is not None:
                    h = self.activation(h)
                return h

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout=0):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConvWithCV(in_feats, n_hidden, activation, dropout=dropout))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConvWithCV(n_hidden, n_hidden, activation, dropout=dropout))
        self.layers.append(SAGEConvWithCV(n_hidden, n_classes, None, dropout=dropout))

    def forward(self, blocks):
        h = blocks[0].srcdata['feat']
        updates = []
        for layer, block in zip(self.layers, blocks):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            agg_hbar_dst = block.dstdata['agg_hist']
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h, h_neigh = layer(block, (h, h_dst), agg_hbar_dst)
            block.dstdata['agg_hist'] = h_neigh
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        ys = []
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                induced_nodes = block.srcdata[dgl.NID]

                h = x[induced_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))

                y[start:end] = h.cpu()

            ys.append(y)
            x = y
        return y, ys



class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(seeds)
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks

def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.

    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        inputs = g.ndata['feat']
        pred, _ = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid])

def load_subtensor(g, labels, blocks, dev_id):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    blocks[0].srcdata['feat'] = g.ndata['feat'][blocks[0].srcdata[dgl.NID]].to(dev_id)
    blocks[-1].dstdata['label'] = labels[blocks[-1].dstdata[dgl.NID]].to(dev_id)
    for i, block in enumerate(blocks):
        block.dstdata['agg_hist'] = g.ndata['agg_hist_%d' % i][block.dstdata[dgl.NID]].to(dev_id)

def init_history(g, model, dev_id):
    with th.no_grad():
        history = model.inference(g, g.ndata['feat'], 1000, dev_id)[1]
        for layer in range(args.num_layers + 1):
            if layer > 0:
                hist_col = 'hist_%d' % layer
                g.ndata['hist_%d' % layer] = history[layer - 1]
            else:
                hist_col = 'feat'
            agg_hist_col = 'agg_hist_%d' % layer
            g.update_all(fn.copy_u(hist_col, 'm'), fn.mean('m', agg_hist_col))
            #g.ndata[agg_hist_col].zero_()
            if layer > 0:
                del g.ndata[hist_col]

def update_history(g, blocks):
    with th.no_grad():
        for i, block in enumerate(blocks):
            ids = block.dstdata[dgl.NID]
            agg_hist_col = 'agg_hist_%d' % i

            agg_hist = block.dstdata['agg_hist'].detach().cpu()
            g.ndata[agg_hist_col][ids] = agg_hist

def run(args, dev_id, data):
    dropout = args.dropout

    th.cuda.set_device(dev_id)

    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, g = data

    # Create sampler
    sampler = NeighborSampler(g, [int(_) for _ in args.fan_out.split(',')])

    # Create PyTorch DataLoader for constructing blocks
    dataloader = DataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers_per_gpu)

    # Define model
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, dropout)

    # Move the model to GPU and define optimizer
    model = model.to(dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(dev_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Compute history tensor and their aggregation before training on CPU
    model.eval()
    init_history(g, model, dev_id)
    model.train()

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc, best_test_acc = 0, 0
    for epoch in range(args.num_epochs):
        tic = time.time()
        model.train()
        for step, blocks in enumerate(dataloader):
            tic_step = time.time()

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            input_nodes = blocks[0].srcdata[dgl.NID]
            seeds = blocks[-1].dstdata[dgl.NID]

            load_subtensor(g, labels, blocks, dev_id)

            # forward
            batch_pred = model(blocks)
            # update history
            update_history(g, blocks)
            # compute loss
            batch_labels = blocks[-1].dstdata['label']
            loss = loss_fcn(batch_pred, batch_labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f}'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:])))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            model.eval()
            eval_acc, test_acc = evaluate(model, g, labels, val_nid, test_nid, args.val_batch_size, dev_id)
            print('Eval Acc {:.4f}, Test Acc {:.4f}'.format(eval_acc, test_acc))
            if best_eval_acc < eval_acc:
                best_eval_acc = eval_acc
                best_test_acc = test_acc
            print('Best eval acc {:.4f}, corresponding test acc {:.4f}'.format(best_eval_acc, best_test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='0')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='1,1')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--val-batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--num-workers-per-gpu', type=int, default=0)
    argparser.add_argument('--dropout', type=float, default=0)
    args = argparser.parse_args()

    # load reddit data
    data = DglNodePropPredDataset(name='ogbn-products')
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    graph, labels = data[0]
    labels = labels[:, 0]
    graph = dgl.as_heterograph(graph)
    in_feats = graph.ndata['feat'].shape[1]
    n_classes = (labels.max() + 1).item()
    prepare_mp(graph)
    # Pack data
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, graph

    run(args, int(args.gpu), data)