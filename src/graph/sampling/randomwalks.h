/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler/randomwalks.h
 * \brief DGL sampler - templated implementation definition of random walks
 */

#ifndef DGL_GRAPH_SAMPLING_RANDOMWALKS_H_
#define DGL_GRAPH_SAMPLING_RANDOMWALKS_H_

#include <dgl/runtime/container.h>
#include <dgl/base_heterograph.h>
#include <dgl/array.h>
#include <vector>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

/*!
 * \brief Get the node types traversed by the metapath.
 * \return A 1D array of shape (len(metapath) + 1,) with node type IDs.
 */
template<DLDeviceType XPU, typename IdxType>
TypeArray GetNodeTypesFromMetapath(
    const HeteroGraphPtr hg,
    const TypeArray metapath);

/*!
 * \brief Metapath-based random walk with constant restart probability.
 * \param hg The heterograph.
 * \param seeds A 1D array of seed nodes, with the type the source type of the first
 *        edge type in the metapath.
 * \param metapath A 1D array of edge types representing the metapath.
 * \param prob A vector of 1D float arrays, indicating the transition probability of
 *        each edge by edge type.  An empty float array assumes uniform transition.
 * \param restart_prob The restart probability
 * \return A 2D array of shape (len(seeds), len(metapath) + 1) with node IDs.
 * \note This function should be called together with GetNodeTypesFromMetapath to
 *       determine the node type of each node in the random walk traces.
 */
template<DLDeviceType XPU, typename IdxType>
IdArray RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob = 0);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl

#endif  // DGL_GRAPH_SAMPLING_RANDOMWALKS_H_
