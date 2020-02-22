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
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
from torch.nn.parallel import DistributedDataParallel

# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
#
# TODO: confirm if this is necessary for MXNet and Tensorflow.  If so, we need
# to standardize worker process creation since our operators are implemented with
# OpenMP.
def thread_wrapped_func(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

class SAGEConvWithCV(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.W = nn.Linear(in_feats * 2, out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, frontier_l, H_l, HBar_l):
        if self.training:
            with frontier_l.local_scope():
                frontier_l.ndata['hbar'] = HBar_l
                frontier_l.ndata['hdelta'] = H_l - HBar_l
                frontier_l.update_all(
                    fn.copy_u('hbar', 'm'), fn.mean('m', 'hbar_new'), etype='history')
                frontier_l.update_all(
                    fn.copy_u('hdelta', 'm'), fn.mean('m', 'hdelta_new'), etype='sampled')
                h_neigh = frontier_l.ndata['hbar_new'] + frontier_l.ndata['hdelta_new']
                h = F.relu(self.W(torch.cat([H_l, h], 1)))
                return h
        else:
            with frontier_l.local_scope():
                frontier_l.ndata['h'] = h
                frontier_l.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_new'))
                h_neigh = frontier_l.ndata['h_new']
                h = F.relu(self.W(torch.cat([h_neigh, h], 1)))
                return h

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConvWithCV(in_feats, n_hidden))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConvWithCV(n_hidden, n_hidden))
        self.layers.append(SAGEConvWithCV(n_hidden, n_classes))

    def forward(self, frontiers, x, hbars):
        h = x
        nodes_to_update_list = []
        values_to_update_list = []
        for layer, frontier, hbar in zip(self.layers, frontiers, hbars):
            h = layer(frontier, h, hbar)

            nodes_to_update = torch.unique(frontier.all_edges(etype='sampled')[1]).numpy()
            induced_nodes_to_update = (induced_nodes[:, None] == nodes_to_update[None, :]).argmax(0)

            nodes_to_update_list.append(nodes_to_update)
            values_to_update_list.append(h.detach().cpu().numpy()[induced_nodes_to_update])
        return h, nodes_to_update_list, values_to_update_list

class ControlVariateSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_frontiers(self, seeds):
        frontiers = []
        ntype = self.g.ntypes[0]
        for fanout in self.fanouts:
            # The neighbors needed for computing PHat (H - HBar)
            frontier_sampled = dgl.sampling.sample_neighbors(self.g, seeds, fanout)
            # The neighbors needed for computing P HBar
            frontier_history = dgl.in_subgraph(self.g, seeds)

            # Union the two types of frontiers
            sampled_src, sampled_dst = frontier_sampled.all_edges()
            history_src, history_dst = frontier_history.all_edges()
            frontier = dgl.heterograph({
                (ntype, 'sampled', ntype): (sampled_src, sampled_dst),
                (ntype, 'history', ntype): (history_src, history_dst)},
                card=g.number_of_nodes())
            frontiers.insert(0, frontier)

            # Determine which nodes are the seeds for next frontier.
            # We only need to grow from those that are "sampled".
            seeds = torch.unique(torch.cat([sampled_src, sampled_dst]))

        return frontiers

    def sample_frontiers_for_inference(self, seeds):
        frontiers = []
        for _ in self.fanouts:
            frontier = dgl.in_subgraph(self, seeds)
            frontiers.insert(0, frontier)
            seeds = torch.unique(torch.cat(frontier.all_edges(), 1))
        return frontiers

@thread_wrapped_func
def run(proc_id, n_gpus, args, devices, data):
    dropout = 0.2

    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)
    th.cuda.set_device(dev_id)

    # Unpack data
    train_mask, val_mask, in_feats, labels, n_classes, g = data
    train_nid = th.LongTensor(np.nonzero(train_mask)[0])
    val_nid = th.LongTensor(np.nonzero(val_mask)[0])
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)

    # Split train_nid
    train_nid = th.split(train_nid, len(train_nid) // n_gpus)[dev_id]

    # Create sampler
    sampler = NeighborSampler(g, [args.fan_out] * args.num_layers)
    val_frontiers = sampler.sample_frontiers(val_nid)
    val_induced_nodes = val_frontiers[0].ndata[dgl.NID]
    batch_val_inputs = g.ndata['features'][val_induced_nodes].to(dev_id)
    batch_val_labels = labels[val_induced_nodes].to(dev_id)
    induced_nodes_in_val = th.BoolTensor(
        np.isin(val_induced_nodes.numpy(), val_nid.numpy(), assume_unique=True)).to(dev_id)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, dropout)
    model = model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(dev_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()
        train_nid_batches = train_nid[th.randperm(len(train_nid))]
        n_batches = (len(train_nid_batches) + args.batch_size - 1) // args.batch_size
        model.train()
        for step in range(n_batches):
            seeds = train_nid_batches[step * args.batch_size:(step+1) * args.batch_size]
            if proc_id == 0:
                tic_step = time.time()

            frontiers = sampler.sample_frontiers(seeds)
            induced_nodes = frontiers[0].ndata[dgl.NID]
            batch_inputs = g.ndata['features'][induced_nodes].to(dev_id)
            batch_labels = labels[induced_nodes].to(dev_id)
            induced_nodes_in_seeds = th.BoolTensor(
                np.isin(induced_nodes.numpy(), seeds.numpy(), assume_unique=True)).to(dev_id)

            # forward
            batch_pred = model(frontiers, batch_inputs)[induced_nodes_in_seeds]
            batch_labels = batch_labels[induced_nodes_in_seeds]
            # compute loss
            loss = loss_fcn(batch_pred, batch_labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            if n_gpus > 1:
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        th.distributed.all_reduce(param.grad.data,
                                                  op=th.distributed.ReduceOp.SUM)
                        param.grad.data /= n_gpus
            optimizer.step()
            if proc_id == 0:
                iter_tput.append(len(seeds) * n_gpus / (time.time() - tic_step))
            if step % args.log_every == 0 and proc_id == 0:
                acc = compute_acc(batch_pred, batch_labels)
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f}'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:])))

        if n_gpus > 1:
            th.distributed.barrier()

        toc = time.time()
        if proc_id == 0:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 5:
                avg += toc - tic

            count = 0
            if epoch % args.eval_every == 0 and epoch != 0:
                model.eval()
                val_nid_batches = val_nid[th.randperm(len(val_nid))]
                n_batches = (len(train_nid_batches) + args.batch_size - 1) // args.batch_size
                for step in range(n_batches):
                    seeds = val_nid_batches[step * args.batch_size:(step+1) * args.batch_size]
                    if proc_id == 0:
                        tic_step = time.time()

                    frontiers = sampler.sample_frontiers_for_inference(seeds)
                    induced_nodes = frontiers[0].ndata[dgl.NID]
                    batch_inputs = g.ndata['features'][induced_nodes].to(dev_id)
                    batch_labels = labels[induced_nodes].to(dev_id)
                    induced_nodes_in_seeds = th.BoolTensor(
                        np.isin(induced_nodes.numpy(), seeds.numpy(), assume_unique=True)).to(dev_id)

                    # forward
                    batch_pred = model(frontiers, batch_inputs)[induced_nodes_in_seeds]
                    batch_labels = batch_labels[induced_nodes_in_seeds]
                    count += (batch_labels == batch_pred).sum().item()
                print('Eval Acc', count / len(val_nid))

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='0')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=64)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=int, default=10)
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    args = argparser.parse_args()
    
    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    # load reddit data
    data = RedditDataset(self_loop=True)
    train_mask = data.train_mask
    val_mask = data.val_mask
    features = th.Tensor(data.features)
    in_feats = features.shape[1]
    labels = th.LongTensor(data.labels)
    n_classes = data.num_labels
    # Construct graph
    g = dgl.graph(data.graph.all_edges())
    g.ndata['features'] = features
    # Pack data
    data = train_mask, val_mask, in_feats, labels, n_classes, g

    if n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices, data))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
