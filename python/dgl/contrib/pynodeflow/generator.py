import collections
from ...base import NTYPE, NID, ETYPE, EID
from ... import backend as F

def create_nodeflow(layer_mappings, block_mappings, nf_src, nf_dst, nf_ntype_ids, nf_etype_ids,
                    nf_ntype_names, nf_etype_names, seed_map, block_aux_data):
    g = graph((nf_src, nf_dst))
    g.ndata[NTYPE] = F.tensor(nf_ntype_ids)
    g.edata[ETYPE] = F.tensor(nf_etype_ids)
    hg = to_hetero(g, nf_ntype_names, nf_etype_names)
    hg.layer_mappings = layer_mappings
    hg.block_mappings = block_mappings
    hg.seed_map = seed_map
    hg.block_aux_data = block_aux_data
    return hg


def _remap(array):
    """Find the unique elements of the array and return another array with indices
    to the array of unique elements."""
    uniques = np.unique(array)
    invmap = {x: i for i, x in enumerate(uniques)}
    remapped = np.array([invmap[x] for x in array])
    return uniques, invmap, remapped

class Generator(object):
    def __init__(self, sampler=None, num_workers=0):
        self.sampler = sampler
        self.num_workers = num_workers

        if num_workers > 0:
            raise NotImplementedError('multiprocessing')

    def __call__(self, seeds, **auxiliary):
        """
        The __call__ function must take in an array of seeds, and any auxiliary data, and
        return a NodeFlow grown from the seeds and conditioned on the auxiliary data.
        """
        raise NotImplementedError

    def __iter__(self):
        for seeds, auxiliary in self.sampler:
            yield self(seeds, **auxiliary), auxiliary

class IterativeGenerator(Generator):
    """NodeFlow generator.  Only works on homographs.
    """
    hetero = False

    def __init__(self, graph, num_blocks, sampler=None, num_workers=0, coalesce=False):
        super().__init__(sampler, num_workers)
        if self.hetero:
            self.graph = to_homo(graph)
            self.heterograph = graph
        else:
            self.graph = graph
        self.num_blocks = num_blocks
        self.coalesce = coalesce

    def __call__(self, seeds, **auxiliary):
        """Default implementation of IterativeGenerator.
        """
        curr_frontier = seeds           # Current frontier to grow neighbors from
        layer_mappings = []             # Mapping from layer node ID to parent node ID
        block_mappings = []             # Mapping from block edge ID to parent edge ID, or -1 if nonexistent
        rel_graphs = []
        nf_srcs = array('l')
        nf_dsts = array('l')
        nf_ntype_ids = array('l')
        nf_etype_ids = array('l')
        nf_ntype_names = []
        nf_etype_names = []
        nf_ntype_names_invmap = {}
        nf_etype_names_invmap = {}
        block_aux_data = []             # Auxiliary data returned by generator's grow() function for each block

        if self.coalesce:
            curr_frontier, _, seed_map = _remap(seeds.numpy())
            curr_frontier = F.tensor(curr_frontier)
        else:
            seed_map = list(range(len(seeds)))

        if self.hetero:
            graph_ntype = F.asnumpy(self.graph.ndata[NTYPE])
            graph_nid = F.asnumpy(self.graph.ndata[NID])
            graph_etype = F.asnumpy(self.graph.edata[ETYPE])
            graph_eid = F.asnumpy(self.graph.edata[EID])

        layer_mappings.append(curr_frontier.numpy())
        nf_ntype_ids.extend([self.num_blocks] * len(curr_frontier))
        nf_ntype_names.insert(0, 'layer%d' % self.num_blocks)

        curr_node_offset = 0
        prev_node_offset = len(curr_frontier)
        ntype_offset = 0
        etype_offset = 0

        for i in reversed(range(self.num_blocks)):
            result = self.grow(i, curr_frontier, **auxiliary)
            if len(result) == 4:            # homograph
                assert not self.hetero
                neighbor_nodes, neighbor_edges, num_neighbors, aux_data = result
            elif len(result) == 5:          # heterograph
                assert self.hetero
                neighbor_nodes, neighbor_edges, neighbor_etypes, num_neighbors, aux_data = result
            else:
                raise ValueError('invalid return from grow() method.')

            # Layers such as GraphSAGE convolutions treat the features of the node itself and features
            # of neighbors differently, so we add another set of edges
            # The un-coalesced mapping from layer node ID to parent edge ID
            prev_frontier_srcs = F.cat([neighbor_nodes, curr_frontier], 0)
            # The un-coalesced mapping from block edge ID to parent edge ID
            prev_frontier_edges = F.asnumpy(F.cat(
                [neighbor_edges, curr_frontier.new(len(curr_frontier)).fill_(-1)], 0))

            # Coalesce nodes
            # block_srcs: NodeFlow global src ID for the current block
            # block_dsts: NodeFlow global dst ID for the current block
            if self.coalesce:
                prev_frontier, _, block_srcs = _remap(prev_frontier_srcs)
            else:
                prev_frontier = prev_frontier_srcs.numpy()
                block_srcs = np.arange(len(prev_frontier_edges))
            block_dsts = np.arange(len(curr_frontier)).repeat(num_neighbors)

            if not hetero:
                # neighbor connection
                nf_src.extend(block_srcs[:-len(curr_frontier)] + prev_node_offset)
                nf_dst.extend(block_dsts + curr_node_offset)
                nf_ntype_ids.extend([i] * len(prev_frontier))
                nf_etype_ids.extend([2 * i] * len(block_dsts))
                # self connection
                nf_src.extend(block_srcs[-len(curr_frontier):] + prev_node_offset)
                nf_dst.extend(np.arange(len(curr_frontier)) + curr_node_offset)
                nf_etype_ids.extend([2 * i + 1] * len(curr_frontier))

                nf_ntype_names.insert(0, 'layer%d' % i)
                nf_etype_names.insert(0, 'block%d:self' % i)
                nf_etype_names.insert(0, 'block%d' % i)

                layer_mappings.insert(0, prev_frontier)
                block_mappings.insert(0, prev_frontier_edges)
                block_aux_data.insert(0, aux_data)
            else:
                # neighbor connection
                nf_src.extend(block_srcs[:-len(curr_frontier)] + prev_node_offset)
                nf_dst.extend(block_dsts + curr_node_offset)

                prev_frontier_ntypes = F.asnumpy(graph_ntype[prev_frontier])
                unique_ntypes, _, prev_frontier_ntypes_mapped = _remap(prev_frontier_ntypes)
                nf_ntype_names.extend('layer%d:%d' % (i, nt) for nt in unique_ntypes)
                nf_ntype_ids.extend(prev_frontier_ntypes_mapped + ntype_offset)
                ntype_offset += len(unique_ntypes)

                block_etypes = F.asnumpy(neighbor_etypes)
                unique_etypes, _, block_etypes_mapped = _remap(block_etypes)
                nf_etype_ids.extend(block_etypes_mapped + etype_offset)
                nf_etype_names.extend('block%d:%d' % (i, et) for et in unique_etypes)
                etype_offset += len(unique_etypes)

                # self connection
                nf_src.extend(block_srcs[-len(curr_frontier):] + prev_node_offset)
                nf_dst.extend(np.arange(len(curr_frontier)) + curr_node_offset)

                curr_frontier_ntypes = F.asnumpy(graph_ntype[curr_frontier])
                unique_ntypes, _, curr_frontier_ntypes_mapped = _remap(curr_frontier_ntypes)
                nf_etype_names.extend('block%d:self:%d' % (i, nt) for nt in unique_ntypes)
                nf_etype_ids.extend(curr_frontier_ntypes_mapped + etype_offset)
                etype_offset += len(unique_etypes)

            curr_frontier = F.tensor(prev_frontier)
            curr_node_offset = prev_node_offset
            prev_node_offset += len(prev_frontier)

        return create_nodeflow(
            layer_mappings=layer_mappings,
            block_mappings=block_mappings,
            nf_src=nf_src,
            nf_dst=nf_dst,
            nf_ntype_ids=nf_ntype_ids,
            nf_etype_ids=nf_etype_ids,
            nf_ntype_names=nf_ntype_names,
            nf_etype_names=nf_etype_names,
            seed_map=seed_map,
            block_aux_data=block_aux_data)

    def grow(self, block_id, curr_frontier, **auxiliary):
        """Function that takes in the node set in the current layer, and returns the
        neighbors of each node.

        Parameters
        ----------
        block_id : int
        curr_frontier : Tensor
        auxiliary : any auxiliary data yielded by the sampler

        Returns
        -------
        neighbor_nodes, incoming_edges, num_neighbors, ... : Tensor, Tensor, Tensor
            num_neighbors[i] contains the number of neighbors generated for curr_frontier[i]

            neighbor_nodes[sum(num_neighbors[0:i]):sum(num_neighbors[0:i+1])] contains the actual
            neighbors as node IDs in the original graph for curr_frontier[i].

            incoming_edges[sum(num_neighbors[0:i]):sum(num_neighbors[0:i+1])] contains the actual
            incoming edges as edge IDs in the original graph for curr_frontier[i], or -1 if the
            edge does not exist, or if we don't care about the edge, in the original graph.

            Besides the three elements, this function can return arbitrary additional data for
            each block.  The additional data is concatenated into a list and returned as the
            member ``block_aux_data`` in the constructed NodeFlow.
        """
        raise NotImplementedError

# This generator grows the NodeFlow by taking all neighborhoods.
class DefaultGenerator(IterativeGenerator):
    def grow(self, block_id, curr_frontier, **auxiliary):
        # Relies on that the same dst node of in_edges are contiguous, and the dst nodes
        # are ordered the same as curr_frontier.
        src, _, eid = self.graph.in_edges(curr_frontier, form='all')
        num_neighbors = self.graph.in_degrees(curr_frontier)
        return src, eid, num_neighbors, {}


class NeighborSamplerGenerator(IterativeGenerator):
    def __init__(self, graph, num_blocks, num_neighbors, sampler=None, num_workers=0,
                 coalesce=False, with_replacement=False):
        super().__init__(graph, num_blocks, sampler, num_workers, coalesce)
        self.with_replacement = with_replacement
        self.num_neighbors = num_neighbors

    def _sample_without_replacement(self, segs):
        return [np.random.choice(n, self.num_neighbors) for n in segs]

    def _sample_with_replacement(self, segs):
        return [np.random.randint(0, n, self.num_neighbors) for n in segs]

    def grow(self, block_id, curr_frontier, **auxiliary):
        src, _, eid = self.graph.in_edges(curr_frontier, form='all')
        in_degrees = self.graph.in_degrees(curr_frontier)
        src = src.split(in_degrees)
        eid = eid.split(in_degrees)
        if self.with_replacement:
            sampled_indices = self._sample_with_replacement(in_degrees)
        else:
            sampled_indices = self._sample_without_replacement(in_degrees)
        sampled_indices = F.tensor(sampled_indices)

        src_sampled = [F.take(s, i, 0) for s, i in zip(src, sampled_indices)]
        eid_sampled = [F.take(e, i, 0) for e, i in zip(dst, sampled_indices)]

        return src_sampled, eid_sampled, F.full_1d(
                len(curr_frontier), self.num_neighbors, F.int64, F.cpu()), {}


class MetapathBasedDefaultGenerator(IterativeGenerator):
    def __init__(self, heterograph, num_blocks, metapaths, sampler=None, num_workers=0,
                 coalesce=False):
        self.heterograph = heterograph
        self.graph = to_homo(heterograph)
        super().__init__(self.graph, num_blocks, sampler, num_workers, coalesce)
        self.metapaths = metapaths
        self.metapath_by_seedtype_id = {}
        for metapath in metapaths:
            _, seedtype_id = self.heterograph._graph.metagraph.find_edge(self.metapath[-1])
            if seedtype_id not in self.metapath_by_seedtype_id:
                self.metapath_by_seedtype_id[seedtype_id] = []
            self.metapath_by_seedtype_id[seedtype_id].append(metapath)

    def grow(self, block_id, curr_frontier, **auxiliary):
        curr_frontier = F.asnumpy(curr_frontier)
        curr_frontier_ntypes = F.asnumpy(self.graph.ndata[NTYPE])[curr_frontier]
        curr_frontier_nids = F.asnumpy(self.graph.ndata[NID])[curr_frontier]

        src = []
        eid = []
        etype = []
        num_neighbors = []

        for ntype, nid in zip(curr_frontier_ntypes, curr_frontier_nids):
            curr_num_neighbors = 0
            for metapath_id, metapath in enumerate(self.metapath_by_seedtype_id):
                curr_nids = nid
                for metapath_etype in reversed(metapath):
                    metapath_etype_id = self.heterograph.get_etype_id(metapath_etype)
                    curr_src, curr_dst, curr_eid = self.graph.in_edges(curr_nids, form='all')
                    curr_etypes = F.take(self.graph.edata[ETYPE], curr_eid, 0)
                    curr_src = F.asnumpy(curr_src)
                    curr_dst = F.asnumpy(curr_dst)
                    curr_eid = F.asnumpy(curr_eid)
                    curr_etypes = F.asnumpy(curr_etypes)
                    curr_etype_mask = (curr_etypes == metapath_etype_id)
                    curr_nids = curr_src[curr_etype_mask]

                src.append(curr_nids)
                eid.append(np.full((len(curr_nids),), -1))
                etype.append(np.full((len(curr_nids),), -(metapath_id + 1)))
                curr_num_neighbors += len(curr_nids)
            num_neighbors.append(len(curr_nids))

        src = F.tensor(np.concatenate(src))
        eid = F.tensor(np.concatenate(eid))
        etype = F.tensor(np.concatenate(etype))
        num_neighbors = F.tensor(num_neighbors)

        return src, eid, etype, num_neighbors, {}
