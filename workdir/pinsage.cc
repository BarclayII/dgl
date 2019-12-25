using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {

namespace sampling {

namespace {

std::pair<IdArray, IdArray> _PinSageNeighborSampling(
    const HeteroGraphPtr hg,
    const IdArray etypes,
    const IdArray seeds,
    int num_neighbors,
    int num_traces,
    double restart_prob
    int trace_length) {
  const auto metagraph = hg->meta_graph();
  int64_t num_etypes = etypes->shape[0];
  int64_t num_seeds = seeds->shape[0];
  const dgl_type_t *etype_data = static_cast<dgl_type_t *>(etypes->data);
  const dgl_id_t *seed_data = static_cast<dgl_id_t *>(seeds->data);

  IdArray neighbors = IdArray::Empty(
      {num_seeds, num_neighbors},
      DLDataType{kDLInt, hg->NumBits(), 1},
      hg->Context());
  IdArray neighbor_weights = IdArray::Empty(
      {num_seeds, num_neighbors},
      DLDataType{kDLFloat, 64, 1},
      hg->Context());
  dgl_id_t *neighbor_data = static_cast<dgl_id_t *>(neighbors->data);
  size_t *neighbor_weight_data = static_cast<double *>(neighbor_weights->data);

  std::unordered_map<dgl_id_t, size_t> node_count;
  std::vector<std::pair<dgl_id_t, size_t>> node_count_sorted;
  for (int64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
    int64_t pos = seed_id * num_neighbors;

    node_count.clear();
    // pad the node counts to num_neighbors entries so that it can always return
    // num_neighbors neighbors.
    for (int i = 0; i < num_neighbors; ++i)
      node_count[i] = 0;

    for (int trace_id = 0; trace_id < num_traces; ++trace_id) {
      dgl_id_t curr = seed_data[seed_id];

      for (size_t hop_id = 0; hop_id < trace_length; ++hop_id) {
        bool halt = false;

        for (size_t i = 0; i < num_etypes; ++i) {
          const auto &succ = hg->SuccVec(etype_data[i], curr);
          if (succ.size() == 0) {
            halt = true;
            break;
          }
          curr = succ[RandomEngine::ThreadLocal()->RandInt(succ.size())];
        }

        if (halt || (RandomEngine::ThreadLocal()->Uniform() < restart_prob))
          break;

        ++node_count[curr];
      }
    }

    node_count_sorted.clear();
    node_count_sorted.insert(node_count_sorted.begin(), node_count.begin(), node_count.end());
    std::sort(
        node_count_sorted.begin(),
        node_count_sorted.end(),
        [] (std::pair<dgl_id_t, size_t> a, std::pair<dgl_id_t, size_t> b) {
          return a.second > b.second;
        });

    double total_weights = 0;
    for (auto it = node_count_sorted.begin();
         i < num_neighbors;
         ++i, ++it) {
      neighbor_data[pos + i] = it->first;
      total_weights += it->second;
    }
    for (auto it = node_count_sorted.begin();
         i < num_neighbors;
         ++i, ++it)
      neighbor_weight_data[pos + i] = it->second / total_weights;
  }

  std::pair<IdArray, IdArray> result = std::make_pair(neighbors, neighbor_weights);
  return result;
}

};  // namespace

NodeFlowFrontier PinSageSampleNeighbors(
    const HeteroGraphPtr hg,
    const IdArray etypes,
    const IdArray seeds,
    int num_neighbors,
    int num_traces,
    double restart_prob,
    int trace_length) {
  NodeFlowFrontier frontier;
  int64_t num_seeds = seeds->shape[0];
  int64_t num_etypes = etypes->shape[0];
  const dgl_type_t *etype_data = static_cast<dgl_type_t *>(etypes->data);
  const auto parent_metagraph = hg->meta_graph();
  dgl_type_t node_type = parent_metagraph->FindEdge(etypes[0]).first;
  // Check if the starting node type and the ending node type are the same.
  CHECK_EQ(node_type, parent_metagraph->FindEdge(etypes[num_etypes - 1]).second);

  const auto result = _PinSageNeighborSampling(
      hg, etypes, seeds, num_neighbors, num_traces, restart_prob, trace_length);

  const auto neighbors = result.first;
  const auto neighbor_weights = result.second;

  const dgl_id_t *seed_data = static_cast<dgl_id_t *>(seeds->data);
  const dgl_id_t *neighbor_weight_data = static_cast<double *>(neighbor_weights.data);
  const dgl_id_t *neighbor_data = static_cast<dgl_id_t *>(neighbors->data);

  std::vector<dgl_id_t> src, dst; // the source and destination of frontier
  std::vector<double> weights;    // weights to be assigned to the frontier edges
  for (int64_t i = 0; i < num_seeds; ++i) {
    dgl_id_t curr_seed = seed_data[i];

    for (int64_t j = 0; j < num_neighbors; ++j) {
      int64_t ij = i * num_neighbors + j;

      // if an edge has a weight bigger than 0, add it to the graph.
      if (neighbor_weight_data[ij] > 0) {
        src.push_back(neighbor_data[ij]);
        dst.push_back(curr_seed);
        weights.push_back(neighbor_weight_data[ij]);
      }
    }
  }

  int64_t num_vertices = hg->NumVertices(node_type);
  int64_t num_edges = src.size();

  // Create the graph within the same node ID space of the parent graph.
  // XXX: we have to force the graph's storage as COO, otherwise CSR's indptr would take
  // too much memory
  std::vector<HeteroGraphPtr> subrels;
  // The sampled metagraph should have the same node types with the parent graph, while
  // the edge types only contain the one from node_type to node_type.
  GraphPtr metagraph = ImmutableGraph::CreateFromCOO(
      hg->meta_graph()->NumVertices(),
      Full(node_type, 1, 64, DLContext{kDLCPU, 0}),
      Full(node_type, 1, 64, DLContext{kDLCPU, 0}));
  subrels.push_back(UnitGraph::CreateFromCOO(1, num_vertices, num_vertices, src, dst));
  frontier->graph = HeteroGraphPtr(new HeteroGraph(metagraph, subrels));

  frontier->induced_edges = Full(-1, num_edges, 64, DLContext{kDLCPU, 0});
  frontier->data.Set("weights", VecToFloatArray(weights, 64, DLContext{kDLCPU, 0}));

  return frontier;
}

std::vector<NodeFlowFrontier> PinSageNeighborSampling(
    const HeteroGraphPtr hg,
    const IdArray etypes,
    const IdArray seeds,
    int num_neighbors,
    int num_traces,
    double restart_prob,
    int trace_length,
    int num_layers) {
  std::vector<NodeFlowFrontier> frontiers;
  IdArray curr_seeds = seeds;

  for (int i = 0; i < num_layers; ++i) {
    NodeFlowFrontier frontier = PinSageSampleNeighbors(
        hg, etypes, curr_seeds, num_neighbors, num_traces, restart_prob, trace_length);
    frontiers.push_back(frontier);
    curr_seeds = Unique(frontier.graph->GetRelationGraph(0)->Edges().src);
  }

  CompactNodeFlowFrontiers(frontiers);

  return frontiers;
}

};  // namespace sampling

};  // namespace dgl
