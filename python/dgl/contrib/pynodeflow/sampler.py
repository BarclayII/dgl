"""Seed node samplers."""

class NodeSampler(object):
    def __init__(self, graph, seed_nodes, batch_size, drop_last=True, shuffle=False):
        self.graph = graph
        self.seed_nodes = seed_nodes
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __len__(self):
        num_seeds = len(self.seed_nodes)
        if self.drop_last:
            return num_seeds // self.batch_size
        else:
            return (num_seeds + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        seed_nodes = self.seed_nodes
        indices = np.arange(seed_nodes)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(len(self)):
            start = i * self.batch_size
            batch_indices = indices[start:start+self.batch_size]
            yield seed_nodes[batch_indices], {'indices': batch_indices}


class EdgeSampler(object):
    def __init__(self, graph, seed_edges, batch_size, drop_last=True, shuffle=False):
        self.graph = graph
        self.seed_edges = seed_edges
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __len__(self):
        num_seeds = len(self.seed_edges)
        if self.drop_last:
            return num_seeds // self.batch_size
        else:
            return (num_seeds + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        seed_edges = self.seed_edges
        indices = np.arange(seed_edges)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(len(self)):
            start = i * self.batch_size
            batch_indices = indices[start:start+self.batch_size]
            sampled_edges = seed_edges[batch_indices]
            src, dst = self.graph.find_edges(sampled_edges)
            sampled_nodes = F.asnumpy(F.cat([src, dst], 0))
            yield sampled_nodes, {'edges': sampled_edges, 'indices': batch_indices}
