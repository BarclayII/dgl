import numba
import numpy as np
import array
import torch

@numba.njit
def _choice_replace(x, p, size):
    return x[np.searchsorted(np.cumsum(p), np.random.random(size), side="right")]

@numba.njit
def _choice_no_replace(x, p, size):
    if len(x) <= size:
        return x
    p = p.copy()
    cdf = np.cumsum(p)
    r = np.zeros(size, dtype=x.dtype)
    for i in range(size):
        idx = np.searchsorted(cdf, np.random.random() * cdf[-1], side="right")
        r[i] = x[idx]
        p[idx] = 0
        cdf = np.cumsum(p)
    return r
        
@numba.njit
def choice(x, p, size, replace):
    return _choice_replace(x, p, size) if replace else _choice_no_replace(x, p, size)

ent_type = numba.types.DictType(numba.types.int64, numba.types.float32)
@numba.experimental.jitclass([
    ('data', numba.types.ListType(ent_type)),
    ('g_indptr', numba.types.int64[:]),
    ('g_indices', numba.types.int64[:]),
    ('g_ntypes', numba.types.int64[:]),
    ('g_etypes', numba.types.int64[:]),
    ('num_etypes', numba.types.int64),
    ('num_ntypes', numba.types.int64),
])
class Budget(object):
    def __init__(self, num_ntypes, num_etypes, g_indptr, g_indices, g_ntypes, g_etypes):
        self.data = numba.typed.List.empty_list(ent_type)
        for _ in range(num_ntypes):
            self.data.append(numba.typed.Dict.empty(numba.types.int64, numba.types.float32))
        self.g_indptr = g_indptr
        self.g_indices = g_indices
        self.g_ntypes = g_ntypes
        self.g_etypes = g_etypes
        self.num_etypes = num_etypes
        self.num_ntypes = num_ntypes
    
    def update(self, seed_node_set, new_nodes):
        for new_node in new_nodes:
            eid_start = self.g_indptr[new_node]
            eid_end = self.g_indptr[new_node + 1]
            etypes = self.g_etypes[eid_start:eid_end]
            deg = np.bincount(etypes, minlength=self.num_etypes)
            
            for eid in range(eid_start, eid_end):
                index = self.g_indices[eid]
                if seed_node_set is not None and index in seed_node_set:
                    continue
                data = self.data[self.g_ntypes[index]]
                delta = 1 / deg[self.g_etypes[eid]]
                if index not in data:
                    data[index] = delta
                else:
                    data[index] += delta
                    
        for new_node in new_nodes:
            data = self.data[self.g_ntypes[new_node]]
            if new_node in data:
                del data[new_node]
    
    def sample(self, num_nodes_to_sample):
        num_ntypes = self.num_ntypes
        neighbors = numba.typed.List.empty_list(numba.types.int64)
        for ntype in range(num_ntypes):
            data = self.data[ntype]
            indices = np.zeros(len(data), np.int64)
            values = np.zeros(len(data), np.float32)
            for i, (k, v) in enumerate(data.items()):
                indices[i] = k
                values[i] = v ** 2
            values /= values.sum()
            neighbors.extend(choice(indices, size=num_nodes_to_sample, p=values, replace=False))
        return np.asarray(neighbors)

timestamp_ent_type = numba.types.DictType(numba.types.int64, numba.types.int64)
@numba.experimental.jitclass([
    ('data', numba.types.ListType(ent_type)),
    ('timestamps', numba.types.ListType(timestamp_ent_type)),
    ('g_indptr', numba.types.int64[:]),
    ('g_indices', numba.types.int64[:]),
    ('g_ntypes', numba.types.int64[:]),
    ('g_etypes', numba.types.int64[:]),
    ('g_timestamps', numba.types.int64[:]),
    ('num_etypes', numba.types.int64),
    ('num_ntypes', numba.types.int64),
])
class BudgetWithTimestamps(object):
    def __init__(self, num_ntypes, num_etypes, g_indptr, g_indices, g_ntypes, g_etypes, g_timestamps):
        self.data = numba.typed.List.empty_list(ent_type)
        self.timestamps = numba.typed.List.empty_list(timestamp_ent_type)
        for _ in range(num_ntypes):
            self.data.append(numba.typed.Dict.empty(numba.types.int64, numba.types.float32))
            self.timestamps.append(numba.typed.Dict.empty(numba.types.int64, numba.types.int64))
        self.g_indptr = g_indptr
        self.g_indices = g_indices
        self.g_ntypes = g_ntypes
        self.g_etypes = g_etypes
        self.g_timestamps = g_timestamps
        self.num_etypes = num_etypes
        self.num_ntypes = num_ntypes
    
    def update(self, seed_node_set, new_nodes):
        for new_node in new_nodes:
            eid_start = self.g_indptr[new_node]
            eid_end = self.g_indptr[new_node + 1]
            etypes = self.g_etypes[eid_start:eid_end]
            deg = np.bincount(etypes, minlength=self.num_etypes)
            
            for eid in range(eid_start, eid_end):
                index = self.g_indices[eid]
                if seed_node_set is not None and index in seed_node_set:
                    continue
                data = self.data[self.g_ntypes[index]]
                timestamps = self.timestamps[self.g_ntypes[index]]
                
                delta = 1 / deg[self.g_etypes[eid]]
                if index not in data:
                    data[index] = delta
                    
                    # inductively inherit timestamp if the neighbor does not have an existing timestamp
                    if self.g_timestamps[index] == 0:
                        new_timestamp = self.g_timestamps[new_node]
                        if new_timestamp == 0:
                            new_timestamp = self.timestamps[self.g_ntypes[new_node]][new_node]
                    else:
                        new_timestamp = self.g_timestamps[index]
                    assert new_timestamp > 0
                    timestamps[index] = new_timestamp
                else:
                    data[index] += delta
                    
        for new_node in new_nodes:
            data = self.data[self.g_ntypes[new_node]]
            if new_node in data:
                del data[new_node]
    
    def sample(self, num_nodes_to_sample):
        num_ntypes = self.num_ntypes
        neighbors = numba.typed.List.empty_list(numba.types.int64)
        timestamps = numba.typed.List.empty_list(numba.types.int64)
        for ntype in range(num_ntypes):
            data = self.data[ntype]
            indices = np.zeros(len(data), np.int64)
            values = np.zeros(len(data), np.float32)
            for i, (k, v) in enumerate(data.items()):
                indices[i] = k
                values[i] = v ** 2
            values /= values.sum()
            
            chosen = choice(indices, size=num_nodes_to_sample, p=values, replace=False)
            neighbors.extend(chosen)
            for node in chosen:
                timestamps.append(self.timestamps[self.g_ntypes[node]][node])
        return np.asarray(neighbors), np.asarray(timestamps)

class HGTSampler(object):
    def __init__(self, g, num_ntypes, num_etypes, num_nodes_per_type, num_steps):
        self.num_etypes = num_etypes
        self.num_ntypes = num_etypes
        self.g = g
        self.g_csc = g.adjacency_matrix_scipy(False, 'csr', True)
        self.g_csc.indptr = self.g_csc.indptr.astype('int64')
        self.g_csc.indices = self.g_csc.indices.astype('int64')
        self.num_nodes_per_type = num_nodes_per_type
        self.num_steps = num_steps
        
    def sample_subgraph(self, seed_nodes):
        num_seed_nodes = len(seed_nodes)
        new_nodes = torch.LongTensor(seed_nodes).numpy()
        new_node_timestamps = self.g.ndata['timestamp'].numpy()[new_nodes]
        assert (new_node_timestamps > 0).all()
        seed_nodes = array.array('l')
        seed_nodes_with_timestamps = array.array('l')
        seed_node_set = None
        B = BudgetWithTimestamps(
            self.num_ntypes,
            self.num_etypes,
            self.g_csc.indptr,
            self.g_csc.indices,
            self.g.ndata['ntype'].numpy(),
            self.g.edata['etype'][self.g_csc.data].numpy(),
            self.g.ndata['timestamp'].numpy(),
        )
#         B = Budget(
#             self.num_ntypes,
#             self.num_etypes,
#             self.g_csc.indptr,
#             self.g_csc.indices,
#             self.g.ndata['ntype'].numpy(),
#             self.g.edata['etype'][self.g_csc.data].numpy(),
#         )
        for i in range(self.num_steps):
            B.update(seed_node_set, new_nodes)
            
            if seed_node_set is None:
                seed_node_set = set(new_nodes)
            else:
                assert len(seed_node_set & set(new_nodes)) == 0
                seed_node_set |= set(new_nodes)
            seed_nodes.extend(new_nodes)
            assert (new_node_timestamps > 0).all()
            seed_nodes_with_timestamps.extend(new_node_timestamps)
            
            new_nodes, new_node_timestamps = B.sample(self.num_nodes_per_type)
#             new_nodes = B.sample(self.num_nodes_per_type)
        seed_nodes.extend(new_nodes)
        seed_nodes_with_timestamps.extend(new_node_timestamps)
        sg = self.g.subgraph(torch.LongTensor(seed_nodes))
        sg.ndata['timestamp'] = torch.LongTensor(seed_nodes_with_timestamps)
        return sg, num_seed_nodes
