/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/sampler/pinsage.cc
 * \brief PinSAGE-like neighbor sampler
 */

#include <dgl/array.h>
#include <dgl/random.h>
#include <dgl/packed_func_ext.h>
#include "../../c_api_common.h"
#include "randomwalk.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {

namespace sampling {

namespace {

std::pair<IdArray, IdArray> PinSageNeighborSampling(
    const HeteroGraphPtr hg,
    const IdArray etypes,
    const IdArray seeds,
    int num_neighbors,
    int num_traces,
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
      DLDataType{kDLInt, hg->NumBits(), 1},
      hg->Context());
  dgl_id_t *neighbor_data = static_cast<dgl_id_t *>(neighbors->data);
  size_t *neighbor_weight_data = static_cast<size_t *>(neighbor_weights->data);

  std::unordered_map<dgl_id_t, size_t> node_count;
  int64_t count = 0;
  std::vector<std::pair<dgl_id_t, size_t>> node_count_sorted;
  for (int64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
    int64_t pos = seed_id * num_neighbors;

    node_count.clear();
    for (int trace_id = 0; trace_id < num_traces; ++trace_id) {
      dgl_id_t curr = seed_data[seed_id];

      for (size_t hop_id = 0; hop_id < trace_length; ++hop_id) {
        for (size_t i = 0; i < num_etypes; ++i) {
          const auto &succ = hg->SuccVec(etype_data[i], curr);
          if (succ.size() == 0)
            break;
          curr = succ[RandomEngine::ThreadLocal()->RandInt(succ.size())];
          break;
        }

        ++node_count[curr];
        ++count;
      }
    }

    if (count < num_neighbors) {
      LOG(FATAL) << "number of nodes visited less than number of neighbors required";
      break;
    }

    node_count_sorted.clear();
    node_count_sorted.insert(node_count_sorted.begin(), node_count.begin(), node_count.end());
    std::sort(
        node_count_sorted.begin(),
        node_count_sorted.end(),
        [] (std::pair<dgl_id_t, size_t> a, std::pair<dgl_id_t, size_t> b) {
          return a.second > b.second;
        });

    int i = 0;
    for (auto it = node_count_sorted.begin();
         i < num_neighbors;
         ++i, ++it) {
      neighbor_data[pos] = it->first;
      neighbor_weight_data[pos] = it->second;
      ++pos;
    }
  }

  std::pair<IdArray, IdArray> result = std::make_pair(neighbors, neighbor_weights);
  return result;
}

};  // namespace

DGL_REGISTER_GLOBAL("sampler.randomwalk._CAPI_DGLPinSageNeighborSampling")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const HeteroGraphRef hg = args[0];
    const IdArray etypes = args[1];
    const IdArray seeds = args[2];
    int num_neighbors = args[3];
    int num_traces = args[4];
    int trace_length = args[5];

    const auto result = PinSageNeighborSampling(
        hg.sptr(), etypes, seeds, num_neighbors, num_traces, trace_length);
    *rv = ConvertPairToPackedFunc(result);
  });

};  // namespace sampling

};  // namespace dgl
