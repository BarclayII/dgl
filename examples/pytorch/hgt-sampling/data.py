import dgl
import torch
import dgl.function as fn
import ogb.nodeproppred

def load_ogb_mag():
    # Add reverse edge types
    dataset = ogb.nodeproppred.DglNodePropPredDataset('ogbn-mag')
    hg = dataset.graph[0]
    edges = {etype: hg.edges(etype=etype) for etype in hg.canonical_etypes}
    edges.update({(v, e + '_inv', u): (dst, src) for (u, e, v), (src, dst) in edges.items()})
    hg2 = dgl.heterograph(edges)

    # Initialize year
    hg2.nodes['paper'].data['timestamp'] = hg.nodes['paper'].data['node_year'].squeeze()
    for ntype in hg.ntypes:
        if ntype != 'paper':
            hg2.nodes[ntype].data['timestamp'] = torch.zeros(hg2.num_nodes(ntype), dtype=torch.int64)

    # Aggregate bag-of-paper features
    hg2.nodes['paper'].data['feat'] = hg.nodes['paper'].data['feat']
    hg2.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'feat'), etype='has_topic')  # field_of_study
    hg2.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'feat'), etype='writes_inv') # author
    hg2.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'feat'), etype='affiliated_with') # institution

    # Attach log-degree to feature of each node type
    for ntype in hg2.ntypes:
        hg2.nodes[ntype].data['deg'] = torch.zeros(hg2.num_nodes(ntype))
    for utype, etype, vtype in hg2.canonical_etypes:
        hg2.nodes[vtype].data['deg'] += hg2.in_degrees(etype=etype)
    for ntype in hg2.ntypes:
        hg2.nodes[ntype].data['feat'] = torch.cat([
                hg2.nodes[ntype].data['feat'],
                torch.log10(hg2.nodes[ntype].data['deg'][:, None])], 1)
        del hg2.nodes[ntype].data['deg']

    # Attach train/validation/test mask
    split = dataset.get_idx_split()
    train_paper = split['train']['paper']
    val_paper = split['valid']['paper']
    test_paper = split['test']['paper']
    labels = dataset.labels
    for ntype in hg2.ntypes:
        hg2.nodes[ntype].data['train_mask'] = torch.zeros(hg2.num_nodes(ntype), dtype=torch.bool)
        hg2.nodes[ntype].data['val_mask'] = torch.zeros(hg2.num_nodes(ntype), dtype=torch.bool)
        hg2.nodes[ntype].data['test_mask'] = torch.zeros(hg2.num_nodes(ntype), dtype=torch.bool)
        hg2.nodes[ntype].data['label'] = torch.zeros(hg2.num_nodes(ntype), dtype=torch.int64)
        if ntype == 'paper':
            hg2.nodes[ntype].data['train_mask'][train_paper] = True
            hg2.nodes[ntype].data['val_mask'][val_paper] = True
            hg2.nodes[ntype].data['test_mask'][test_paper] = True
            hg2.nodes[ntype].data['label'] = labels['paper'][:, 0]

    # Convert to homogeneous graph and add self-loop
    g = dgl.to_homogeneous(hg2, ndata=['timestamp', 'feat', 'train_mask', 'val_mask', 'test_mask', 'label'])
    g.edata['etype'] = g.edata[dgl.ETYPE]
    g.ndata['ntype'] = g.ndata[dgl.NTYPE]
    g.ndata['nid'] = g.ndata[dgl.NID]
    del g.edata[dgl.ETYPE]
    del g.edata[dgl.EID]
    del g.ndata[dgl.NTYPE]
    del g.ndata[dgl.NID]
    num_nodes = g.num_nodes()
    g = dgl.add_self_loop(g)
    g.edata['etype'][-num_nodes:] = len(hg2.etypes)

    return g, len(hg2.ntypes), len(hg2.etypes) + 1, dataset.num_classes
