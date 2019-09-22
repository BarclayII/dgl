import torch
import torch.nn as nn
import torch.nn.functional as F

class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type='mean',
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.fc_self = nn.Linear(in_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat):
        """
        graph: Bipartite.  Has two edge types.  The first one represents the connection to
        the desired nodes from neighbors.  The second one represents the computation
        dependence of the desired nodes themselves.
        """
        # local_var is not implemented for heterograph
        #graph = graph.local_var()
        feat = self.feat_drop(feat)
        h_self = feat

        neighbor_etype_name = 'block%d' % layer_id
        self_etype_name = 'block%d_self' % layer_id
        src_name = 'layer%d' % layer_id
        dst_name = 'layer%d' % (layer_id + 1)

        graph.nodes[src_name].data['h'] = feat

        # aggregate from neighbors
        graph[neighbor_etype_name].update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
        # copy h to dst nodes from corresponding src nodes, marked by "self etype"
        graph[self_etype_name].update_all(fn.copy_src('h', 'm'), fn.sum('m', 'h'))

        h_neigh = graph.nodes[dst_name].data['neigh']
        h_self = graph.nodes[dst_name].data['h']
        rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)

        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst
