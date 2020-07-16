"""Negative samplers"""
from collections.abc import Mapping
import numpy as np
from .. import backend as F

class _BaseNegativeSampler(object):
    def _generate(self, g, eids, canonical_etype):
        raise NotImplementedError

    def __call__(self, g, eids):
        """Returns negative examples.

        Parameters
        ----------
        g : DGLHeteroGraph
            The graph.
        eids : Tensor or dict[etype, Tensor]
            The sampled edges in the minibatch.

        Returns
        -------
        tuple[Tensor, Tensor] or dict[etype, tuple[Tensor, Tensor]]
            The returned source-destination pairs as negative examples.
        """
        if isinstance(eids, Mapping):
            eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
            neg_pair = {k: self._generate(g, v, k) for k, v in eids.items()}
        else:
            assert len(g.etypes) == 1, \
                'please specify a dict of etypes and ids for graphs with multiple edge types'
            neg_pair = self._generate(g, eids, self.canonical_etypes[0])

        return neg_pair

class Uniform(_BaseNegativeSampler):
    """Negative sampler that randomly chooses negative destination nodes
    for each source node according to a uniform distribution.

    For each edge with type `(utype, etype, vtype)`, ``k`` pairs of nodes
    with node type ``utype`` and ``vtype`` will be returned.  The source nodes
    will always be the source node of the edge, while the destination nodes
    are chosen uniformly.

    Parameters
    ----------
    k : int
        The number of negative examples.

    Examples
    --------
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> neg_sampler = dgl.sampling.negative_sampler.Uniform(2)
    >>> neg_sampler(g, [0, 1])
    (tensor([0, 0, 1, 1]), tensor([1, 0, 2, 3]))
    """
    def __init__(self, k):
        self.k = k

    def _generate(self, g, eids, canonical_etype):
        _, _, vtype = canonical_etype
        shape = F.shape(v)
        dtype = F.dtype(v)
        ctx = F.context(v)
        shape = (shape[0] * self.k,)
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src = F.
        dst = F.randint(shape, dtype, ctx, 0, g.number_of_nodes(vtype))
        return src, dst

class DegreeExponential(_BaseNegativeSampler):
    """Negative sampler that randomly chooses negative destination nodes
    for each source node according to a distribution proportional to
    :math:`d^p`, where :math:`d` is the degree of the sampled destination node
    and :math:`p` is a given constant (usually between 0 and 1).

    For each edge with type `(utype, etype, vtype)`, ``k`` pairs of nodes
    with node type ``utype`` and ``vtype`` will be returned.  The source nodes
    will always be the source node of the edge, while the destination nodes
    are chosen uniformly.

    Parameters
    ----------
    g : DGLHeteroGraph
        The graph.
    k : int
        The number of negative examples.
    p : float
        The exponent of degree.

    Examples
    --------
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> neg_sampler = dgl.sampling.negative_sampler.Uniform(2, 0.9)
    >>> neg_sampler(g, [0, 1])
    (tensor([0, 0, 1, 1]), tensor([1, 0, 2, 3]))
    """
    pass
