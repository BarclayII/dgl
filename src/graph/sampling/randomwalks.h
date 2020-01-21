/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampling/randomwalks.h
 * \brief DGL sampler - templated implementation definition of random walks
 */

#ifndef DGL_GRAPH_SAMPLING_RANDOMWALKS_H_
#define DGL_GRAPH_SAMPLING_RANDOMWALKS_H_

#include <dgl/base_heterograph.h>
#include <dgl/array.h>
#include <vector>
#include <utility>
#include <functional>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

/*!
 * \brief Random walk step function
 */
using StepFunc = std::function<
  //        ID        terminate?
  std::pair<dgl_id_t, bool>(
      void *,       // node IDs generated so far
      dgl_id_t,     // last node ID
      int64_t       // # of steps
      )>;

/*!
 * \brief Get the node types traversed by the metapath.
 * \return A 1D array of shape (len(metapath) + 1,) with node type IDs.
 */
template<DLDeviceType XPU, typename IdxType>
TypeArray GetNodeTypesFromMetapath(
    const HeteroGraphPtr hg,
    const TypeArray metapath);

/*!
 * \brief Generic Random Walk.
 * \param hg The heterograph.
 * \param seeds A 1D array of seed nodes, with the type the source type of the first
 *        edge type in the metapath.
 * \param max_num_steps The maximum number of steps of a random walk path.
 * \param step The random walk step function with type \c StepFunc.
 * \return A 2D array of shape (len(seeds), max_num_steps + 1) with node IDs.
 */
template<DLDeviceType XPU, typename IdxType>
IdArray GenericRandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    int64_t max_num_steps,
    StepFunc step);

/*!
 * \brief Metapath-based random walk with constant restart probability.
 * \param hg The heterograph.
 * \param seeds A 1D array of seed nodes, with the type the source type of the first
 *        edge type in the metapath.
 * \param metapath A 1D array of edge types representing the metapath.
 * \param prob A vector of 1D float arrays, indicating the transition probability of
 *        each edge by edge type.  An empty float array assumes uniform transition.
 * \return A 2D array of shape (len(seeds), len(metapath) + 1) with node IDs.
 * \note This function should be called together with GetNodeTypesFromMetapath to
 *       determine the node type of each node in the random walk traces.
 */
template<DLDeviceType XPU, typename IdxType>
IdArray RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl

#endif  // DGL_GRAPH_SAMPLING_RANDOMWALKS_H_