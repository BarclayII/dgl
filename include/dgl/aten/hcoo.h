
#ifndef DGL_ATEN_HCOO_H_
#define DGL_ATEN_HCOO_H_

#include <dmlc/io.h>
#include <dmlc/serializer.h>
#include <vector>
#include "./array_ops.h"
#include "./types.h"
#include "./macro.h"
#include "./spmat.h"

namespace dgl {
namespace aten {

constexpr uint64_t kDGLSerialize_AtenHCooMatrixMagic = 0xDD6ab27705dff127;

struct HCOOMatrix {
  /*! \brief indptr of metagraph CSR */
  IdArray mg_indptr;
  /*! \brief indices of metagraph CSR */
  IdArray mg_indices;
  /*! \brief edge type ID of metagraph CSR */
  IdArray mg_data;
  /*! \brief COO matrix per edge type */
  std::vector<COOMatrix> data;
  /*! \brief default constructor */
  HCOOMatrix() = default;
  /*! \brief constructor */
  HCOOMatrix(
      IdArray mg_parr,
      IdArray mg_iarr,
      IdArray mg_darr,
      const std::vector<COOMatrix>& d)
    : mg_indptr(mg_parr),
      mg_indices(mg_iarr),
      mg_data(mg_darr),
      data(d) {
    CheckValidity();
  }

  inline void CheckValidity() const {
    /* TODO */
  }

  bool Load(dmlc::Stream* fs) {
    LOG(FATAL) << "not implemented";
    return true;
  }

  void Save(dmlc::Stream* fs) const {
    LOG(FATAL) << "not implemented";
  }

  inline HCOOMatrix CopyTo(const DLContext& ctx) const {
    if (ctx == mg_indptr->ctx)
      return *this;
    std::vector<COOMatrix> new_data(data.size());
    for (int64_t i = 0; i < data.size(); ++i)
      new_data[i] = data[i].CopyTo(ctx);
    return HCOOMatrix(
        mg_indptr.CopyTo(ctx),
        mg_indices.CopyTo(ctx),
        mg_data.CopyTo(ctx),
        new_data);
  }
};

}  // namespace aten
}  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::aten::CSRMatrix, true);
}  // namespace dmlc

#endif  // DGL_ATEN_HCSR_H_

