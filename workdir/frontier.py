import .utils
@register_object('sampler.NodeFlowFrontier')
class DGLNodeFlowFrontier(DGLGraph):
    """The graph representing the computation flow of a single layer in
    minibatch computation.

    NOTE:
    This class can be merged with DGLSubgraph.
    """
    def __init__(self, parent, induced_nodes, induced_edges, graph_topology):
        super().__init__(graph_data=graph_topology, readonly=True)

        self._parent = parent
        self._parent_nid = utils.toindex(induced_nodes)
        self._parent_eid = utils.toindex(induced_edges)

        self._parent_nid_invmap = {v: i for i, v in enumerate(self._parent_nid.tonumpy())}
        self._parent_eid_invmap = {v: i for i, v in enumerate(self._parent_eid.tonumpy())}

    @property
    def parent_nid(self):
        return self._parent_nid.tousertensor()

    @property
    def parent_eid(self):
        return self._parent_eid.tousertensor()

    def map_to_subgraph_nid(self, parent_vids):
        result = [self._parent_nid_invmap.get(v, -1)
                  for v in utils.toindex(parent_vids).tonumpy()]
        return utils.toindex(result).tousertensor()

    def map_to_subgraph_eid(self, parent_eids):
        result = [self._parent_eid_invmap.get(v, -1)
                  for v in utils.toindex(parent_eids).tonumpy()]
        return utils.toindex(result).tousertensor()
