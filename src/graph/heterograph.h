/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/heterograph.h
 * \brief Heterograph
 */

#ifndef DGL_GRAPH_HETEROGRAPH_H_
#define DGL_GRAPH_HETEROGRAPH_H_

#include <dgl/runtime/shared_mem.h>
#include <dgl/base_heterograph.h>
#include <dgl/lazy.h>
#include <utility>
#include <string>
#include <vector>
#include <set>
#include <tuple>
#include <memory>
#include "./unit_graph.h"
#include "shared_mem_manager.h"

namespace dgl {

/*! \brief Heterograph */
class HeteroGraph : public BaseHeteroGraph {
 public:
  HeteroGraph(
      GraphPtr meta_graph,
      const std::vector<HeteroGraphPtr>& rel_graphs,
      const std::vector<int64_t>& num_nodes_per_type = {});

  HeteroGraphPtr GetRelationGraph(dgl_type_t etype) const override {
    LOG(FATAL) << "not implemented";
    return nullptr;
  }

  void AddVertices(dgl_type_t vtype, uint64_t num_vertices) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void AddEdge(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void AddEdges(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void Clear() override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  DLDataType DataType() const override {
    LOG(FATAL) << "not implemented";
    return DLDataType();
  }

  DLContext Context() const override {
    LOG(FATAL) << "not implemented";
    return DLContext();
  }

  uint8_t NumBits() const override {
    LOG(FATAL) << "not implemented";
    return 0;
  }

  bool IsMultigraph() const override;

  bool IsReadonly() const override {
    return true;
  }

  uint64_t NumVertices(dgl_type_t vtype) const override {
    LOG(FATAL) << "not implemented";
    return 0;
  }

  inline std::vector<int64_t> NumVerticesPerType() const override {
    LOG(FATAL) << "not implemented";
    return std::vector<int64_t>();
  }

  uint64_t NumEdges(dgl_type_t etype) const override {
    LOG(FATAL) << "not implemented";
    return 0;
  }

  bool HasVertex(dgl_type_t vtype, dgl_id_t vid) const override {
    LOG(FATAL) << "not implemented";
    return false;
  }

  BoolArray HasVertices(dgl_type_t vtype, IdArray vids) const override;

  bool HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    LOG(FATAL) << "not implemented";
    return false;
  }

  BoolArray HasEdgesBetween(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) const override {
    LOG(FATAL) << "not implemented";
    return BoolArray();
  }

  IdArray Predecessors(dgl_type_t etype, dgl_id_t dst) const override {
    LOG(FATAL) << "not implemented";
    return IdArray();
  }

  IdArray Successors(dgl_type_t etype, dgl_id_t src) const override {
    LOG(FATAL) << "not implemented";
    return IdArray();
  }

  IdArray EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    LOG(FATAL) << "not implemented";
    return IdArray();
  }

  EdgeArray EdgeIdsAll(dgl_type_t etype, IdArray src, IdArray dst) const override {
    LOG(FATAL) << "not implemented";
    return EdgeArray();
  }

  IdArray EdgeIdsOne(dgl_type_t etype, IdArray src, IdArray dst) const override {
    LOG(FATAL) << "not implemented";
    return IdArray();
  }

  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_type_t etype, dgl_id_t eid) const override {
    LOG(FATAL) << "not implemented";
    return std::pair<dgl_id_t, dgl_id_t>(0, 0);
  }

  EdgeArray FindEdges(dgl_type_t etype, IdArray eids) const override {
    LOG(FATAL) << "not implemented";
    return EdgeArray();
  }

  EdgeArray InEdges(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(FATAL) << "not implemented";
    return EdgeArray();
  }

  EdgeArray InEdges(dgl_type_t etype, IdArray vids) const override {
    LOG(FATAL) << "not implemented";
    return EdgeArray();
  }

  EdgeArray OutEdges(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(FATAL) << "not implemented";
    return EdgeArray();
  }

  EdgeArray OutEdges(dgl_type_t etype, IdArray vids) const override {
    LOG(FATAL) << "not implemented";
    return EdgeArray();
  }

  EdgeArray Edges(dgl_type_t etype, const std::string &order = "") const override {
    LOG(FATAL) << "not implemented";
    return EdgeArray();
  }

  uint64_t InDegree(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(FATAL) << "not implemented";
    return 0;
  }

  DegreeArray InDegrees(dgl_type_t etype, IdArray vids) const override {
    LOG(FATAL) << "not implemented";
    return DegreeArray();
  }

  uint64_t OutDegree(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(FATAL) << "not implemented";
    return 0;
  }

  DegreeArray OutDegrees(dgl_type_t etype, IdArray vids) const override {
    LOG(FATAL) << "not implemented";
    return DegreeArray();
  }

  DGLIdIters SuccVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(FATAL) << "not implemented";
    return DGLIdIters();
  }

  DGLIdIters OutEdgeVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(FATAL) << "not implemented";
    return DGLIdIters();
  }

  DGLIdIters PredVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(FATAL) << "not implemented";
    return DGLIdIters();
  }

  DGLIdIters InEdgeVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(FATAL) << "not implemented";
    return DGLIdIters();
  }

  std::vector<IdArray> GetAdj(
      dgl_type_t etype, bool transpose, const std::string &fmt) const override {
    LOG(FATAL) << "not implemented";
    return std::vector<IdArray>();
  }

  aten::COOMatrix GetCOOMatrix(dgl_type_t etype) const override {
    LOG(FATAL) << "not implemented";
    return aten::COOMatrix();
  }

  aten::CSRMatrix GetCSCMatrix(dgl_type_t etype) const override {
    LOG(FATAL) << "not implemented";
    return aten::CSRMatrix();
  }

  aten::CSRMatrix GetCSRMatrix(dgl_type_t etype) const override {
    LOG(FATAL) << "not implemented";
    return aten::CSRMatrix();
  }

  SparseFormat SelectFormat(dgl_type_t etype, dgl_format_code_t preferred_formats) const override {
    LOG(FATAL) << "not implemented";
    return SparseFormat::kAny;
  }

  dgl_format_code_t GetAllowedFormats() const override {
    LOG(FATAL) << "not implemented";
    return 0;
  }

  dgl_format_code_t GetCreatedFormats() const override {
    LOG(FATAL) << "not implemented";
    return 0;
  }

  HeteroSubgraph VertexSubgraph(const std::vector<IdArray>& vids) const override;

  HeteroSubgraph EdgeSubgraph(
      const std::vector<IdArray>& eids, bool preserve_nodes = false) const override;

  HeteroGraphPtr GetGraphInFormat(dgl_format_code_t formats) const override;

  FlattenedHeteroGraphPtr Flatten(const std::vector<dgl_type_t>& etypes) const override;

  GraphPtr AsImmutableGraph() const override;

  /*! \return Load HeteroGraph from stream, using CSRMatrix*/
  bool Load(dmlc::Stream* fs);

  /*! \return Save HeteroGraph to stream, using CSRMatrix */
  void Save(dmlc::Stream* fs) const;

  /*! \brief Convert the graph to use the given number of bits for storage */
  static HeteroGraphPtr AsNumBits(HeteroGraphPtr g, uint8_t bits);

  /*! \brief Copy the data to another context */
  static HeteroGraphPtr CopyTo(HeteroGraphPtr g, const DLContext& ctx);

  /*! \brief Copy the data to shared memory.
  *
  * Also save names of node types and edge types of the HeteroGraph object to shared memory
  */
  static HeteroGraphPtr CopyToSharedMem(
      HeteroGraphPtr g, const std::string& name, const std::vector<std::string>& ntypes,
      const std::vector<std::string>& etypes, const std::set<std::string>& fmts);

  /*! \brief Create a heterograph from 
  *   \return the HeteroGraphPtr, names of node types, names of edge types
  */
  static std::tuple<HeteroGraphPtr, std::vector<std::string>, std::vector<std::string>>
      CreateFromSharedMem(const std::string &name);

  /*! \brief Creat a LineGraph of self */
  HeteroGraphPtr LineGraph(bool backtracking) const;

 private:
  // To create empty class
  friend class Serializer;

  // Empty Constructor, only for serializer
  HeteroGraph() : BaseHeteroGraph() {}

  /* \brief Concrete implementations for COO, CSR, and CSC format */
  HeteroGraphPtr coo_, csr_, csc_;

  /*! \brief The shared memory object for meta info*/
  std::shared_ptr<runtime::SharedMemory> shared_mem_;

  /*! \brief The name of the shared memory. Return empty string if it is not in shared memory. */
  std::string SharedMemName() const;

  /*! \brief template class for Flatten operation
  * 
  * \tparam IdType Graph's index data type, can be int32_t or int64_t
  * \param etypes vector of etypes to be falttened
  * \return pointer of FlattenedHeteroGraphh
  */
  template <class IdType>
  FlattenedHeteroGraphPtr FlattenImpl(const std::vector<dgl_type_t>& etypes) const;
};

}  // namespace dgl


namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::HeteroGraph, true);
}  // namespace dmlc


#endif  // DGL_GRAPH_HETEROGRAPH_H_
