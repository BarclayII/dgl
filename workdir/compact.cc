namespace dgl {

namespace sampling {

// Set of nodes with same type
using nodeset_t = biset<dgl_id_t>;
// Node list with same type
using nodevec_t = std::vector<dgl_id_t>;
// src-dst tensor pairs, indexed by edge type, for one frontier
using edgelist_t = std::vector<std::pair<nodevec_t, nodevec_t>>;

namespace {

/*!
 * \brief Relabel the edge src-dst pairs of all frontier objects from the parent node ID
 * space to compacted space.
 *
 * \param frontiers The NodeFlowFrontier objects
 * \param[out] edgelist_by_frontier The relabeled src-dst pairs
 * \param[out] nodeset The set of involved nodes in the parent node ID space, indexed by
 * node type.
 */
void RelabelSrcDst(
    const std::vector<NodeFlowFrontier> *frontiers,
    std::vector<edgelist_t> *edgelist_by_frontier,
    std::unordered_map<dgl_type_t, nodeset_t> nodeset) {
  int n_frontiers = frontiers->size();

  for (int i = 0; i < n_frontiers; ++i) {
    edgelist_t edgelist;

    // data from un-compacted frontier
    const HeteroGraphPtr hg = (*frontiers)[i].graph;
    const GraphPtr mg = hg->meta_graph();
    const EdgeArray canonical_etypes = mg->Edges("eid");
    const IdArray utypes = canonical_etypes.src;
    const IdArray vtypes = canonical_etypes.dst;
    const IdArray etypes = canonical_etypes.eid;
    const dgl_type_t *utypes_data = static_cast<dgl_type_t *>(utypes->data);
    const dgl_type_t *vtypes_data = static_cast<dgl_type_t *>(vtypes->data);
    const dgl_type_t *etypes_data = static_cast<dgl_type_t *>(etypes->data);

    for (int64_t i = 0; i < etypes->shape[0]; ++i) {
      dgl_type_t etype = etypes_data[j];
      dgl_type_t utype = utypes_data[j];
      dgl_type_t vtype = vtypes_data[j];

      // data from un-compacted frontier
      const EdgeArray edges = hg->Edges(etype, "eid");
      const IdArray src = edges.src;
      const IdArray dst = edges.dst;
      const IdArray eid = edges.eid;
      const dgl_type_t *src_data = static_cast<dgl_id_t *>(src->data);
      const dgl_type_t *dst_data = static_cast<dgl_id_t *>(dst->data);
      const dgl_type_t *eid_data = static_cast<dgl_id_t *>(eid->data);

      // data to compacted frontier
      std::vector<dgl_id_t> src_remap, dst_remap;

      for (int64_t j = 0; j < eid->shape[0]; ++j) {
        dgl_id_t u_remap = (*nodeset)[utype].index_or_insert(src_data[j]);
        dgl_id_t v_remap = (*nodeset)[vtype].index_or_insert(dst_data[j]);
        src_remap.push_back(u_remap);
        dst_remap.push_back(v_remap);
      }

      edgelist.push_back(std::make_pair(src_remap, dst_remap));
    }

    edgelist_by_frontier->push_back(edgelist);
  }
}

/*!
 * \brief Count the number of nodes involved in the minibatch, and compute the mapping
 * from compacted node ID space to parent space as \c induced_nodes attribute of the
 * \c NodeFlowFrontier object.
 *
 * \param[in,out] frontiers The NodeFlowFrontier objects
 * \param nodeset The set of nodes computed from \c RelabelSrcDst function
 * \param[out] num_nodes_per_type The number of nodes involved for each node type
 */
void ComputeInducedNodes(
    std::vector<NodeFlowFrontier> *frontiers,
    const std::unordered_map<dgl_type_t, nodeset_t> &nodeset,
    std::unordered_map<dgl_type_t, size_t> *num_nodes_per_type) {
  int n_frontiers = frontiers->size();
  int num_ntypes = (*frontiers)[0]->meta_graph()->NumVertices();

  List<IdArray> induced_nodes;
  for (dgl_type_t ntype = 0; ntype < num_ntypes; ++ntype) {
    (*num_nodes_per_type)[ntype] = nodeset[ntype].values.size();
    induced_nodes.push_back(VecToIdArray(nodeset[ntype].values));
  }
  for (int i = 0; i < n_frontiers; ++i)
    (*frontiers)[i].induced_nodes = induced_nodes;
}

/*!
 * \brief Re-construct the heterographs for each frontier with the new mapping
 *
 * \param[in,out] frontiers The NodeFlowFrontier objects
 * \param num_nodes_per_type The number of nodes involved per type counted by \c
 * ComputeInducedNodes function
 * \param edgelist_by_frontier The relabeled src-dst pairs computed by \c RelabelSrcDst
 * function
 */
void ReconstructFrontierGraphs(
    std::vector<NodeFlowFrontier> *frontier,
    const std::unordered_map<dgl_type_t, size_t> &num_nodes_per_type,
    const std::vector<edgelist_t> &edgelist_by_frontier) {
  int n_frontiers = frontiers->size();

  for (int i = 0; i < n_frontiers; ++i) {
    const GraphPtr mg = (*frontiers)[i].graph->meta_graph();
    const EdgeArray canonical_etypes = mg->Edges("eid");
    const IdArray utypes = canonical_etypes.src;
    const IdArray vtypes = canonical_etypes.dst;
    const IdArray etypes = canonical_etypes.eid;
    const dgl_type_t *utypes_data = static_cast<dgl_type_t *>(utypes->data);
    const dgl_type_t *vtypes_data = static_cast<dgl_type_t *>(vtypes->data);
    const dgl_type_t *etypes_data = static_cast<dgl_type_t *>(etypes->data);

    std::vector<HeteroGraphPtr> rel_graphs;

    for (int64_t etype = 0; etype < etypes->shape[0]; ++etype) {
      dgl_type_t utype = utypes_data[etype];
      dgl_type_t vtype = vtypes_data[etype];
      const auto rel_graph = UnitGraph::CreateFromCOO(
          utype != vtype ? 2 : 1,
          num_nodes_per_type[utype],
          num_nodes_per_type[vtype],
          VecToIdArray(edgelist_by_frontier[i][etype].first),
          VecToIdArray(edgelist_by_frontier[i][etype].second));
      rel_graphs.push_back(rel_graph);
    }

    (*frontiers)[i].graph = CreateHeteroGraph(mg, rel_graphs);
  }
}

};  // namespace

/*!
 * \brief Compact the node ID space of the frontier graphs in-place and set the
 * induced node mapping.
 *
 * \param[in,out] frontiers The NodeFlowFrontier objects
 *
 * \note The frontier graphs still share the same node ID space between each other
 * \note The frontier graphs still share the same node type space with their parent graph.
 */
void CompactNodeFlowFrontiers(std::vector<NodeFlowFrontier> *frontiers) {
  std::unordered_map<dgl_type_t, nodeset_t> nodeset;
  std::vector<edgelist_t> edgelist_by_frontier;

  // Phase 1
  RelabelSrcDst(frontiers, &edgelist_by_frontier, &nodeset);

  // Phase 2
  std::unordered_map<dgl_type_t, size_t> num_nodes_per_type;
  ComputeInducedNodes(frontiers, nodeset, &num_nodes_per_type);

  // Phase 3
  ReconstructFrontierGraphs(frontiers, num_nodes_per_type, edgelist_by_frontier);
}

};  // namespace sampling

};  // namespace dgl
