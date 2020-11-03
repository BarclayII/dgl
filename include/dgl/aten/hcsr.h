
#ifndef DGL_ATEN_HCSR_H_
#define DGL_ATEN_HCSR_H_

#include <dmlc/io.h>
#include <dmlc/serializer.h>
#include "./array_ops.h"
#include "./types.h"
#include "./macro.h"
#include "./spmat.h"

namespace dgl {
namespace aten {

constexpr uint64_t kDGLSerialize_AtenHCsrMatrixMagic = 0xDD6fc10605dff127;

struct HCSRMatrix {
  /*! \brief indptr of metagraph CSR */
  IdArray mg_indptr;
  /*! \brief indices of metagraph CSR */
  IdArray mg_indices;
  /*! \brief edge type ID of metagraph CSR */
  IdArray mg_data;
  /*!
   * \brief rowtypeptr[i] denotes the starting position of the first node of type i in
   *        etypeptr array.
   */
  IdArray rowtypeptr;
  /*!
   * \brief etypeptr[i] denotes the starting position of the first neighbor of the
   * considered node with the considered edge type.
   */
  IdArray etypeptr;
  /*!
   * \brief the neighbor ID array.
   */
  IdArray indices;
  /*!
   * \brief the data array.
   */
  IdArray data;
  /*! \brief whether the column indices per row are sorted */
  bool sorted = false;
  /*! \brief default constructor */
  HCSRMatrix() = default;
  /*! \brief constructor */
  HCSRMatrix(
      IdArray mg_parr,
      IdArray mg_iarr,
      IdArray mg_darr,
      IdArray rparr,
      IdArray eparr,
      IdArray darr = aten::NullArray(),
      bool sorted_flag = false)
    : mg_indptr(mg_parr),
      mg_indices(mg_iarr),
      mg_data(mg_darr),
      rowtypeptr(rparr),
      etypeptr(eparr),
      data(darr),
      sorted(sorted_flag) {
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

  inline HCSRMatrix CopyTo(const DLContext& ctx) const {
    if (ctx == mg_indptr->ctx)
      return *this;
    return HCSRMatrix(
        mg_indptr.CopyTo(ctx),
        mg_indices.CopyTo(ctx),
        mg_data.CopyTo(ctx),
        rowtypeptr.CopyTo(ctx),
        etypeptr.CopyTo(ctx),
        indices.CopyTo(ctx),
        aten::IsNullArray(data) ? data : data.CopyTo(ctx),
        sorted);
  }
};

}  // namespace aten
}  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::aten::CSRMatrix, true);
}  // namespace dmlc

#endif  // DGL_ATEN_HCSR_H_
