class PinSAGEConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, single_frontier):


class PinSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(PinSAGEConv(in_feats, n_hidden))
        for i in range(1, n_layers):
            self.layers.append(PinSAGEConv(n_hidden, n_hidden))


def find_etype(HG, utype, vtype):
    mg = HG.metagraph
    return list(mg[utype][vtype])


class ModularPinSAGESampler(object):
    # TODO
    pass


class BuiltinPinSAGESampler(object):
    """PinSAGE sampler that generates frontiers from a batch of seed nodes
    with type ``utype``.

    The graph must be a bidirectional bipartite graph, and only the nodes
    with type ``utype`` are sampled.
    """
    def __init__(
            self,
            HG,
            num_layers,
            num_neighbors,
            num_traces,
            restart_prob,
            max_trace_length,
            utype,
            vtype):
        self.HG = HG
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.num_traces = num_traces
        self.restart_prob = restart_prob
        self.max_trace_length = max_trace_length
        self.utype = utype
        self.vtype = vtype
        self.fwtype = HG.get_etype_id(find_etype(HG, utype, vtype))
        self.bwtype = HG.get_etype_id(find_etype(HG, vtype, utype))
        self.metapath = [self.fwtype, self.bwtype]

    def __call__(self, seed_nodes):
        frontiers = _CAPI_DGLPinSageNeighborSampling(
                self.HG._graph,
                utils.toindex(self.metapath).todgltensor(),
                utils.toindex(seed_nodes).todgltensor(),
                self.num_neighbors,
                self.num_traces,
                self.restart_prob,
                self.max_trace_length,
                self.num_layers)
        return frontiers
