/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/csr_sort.cc
 * \brief CSR sorting
 */
#include <dgl/array.h>
#include <numeric>
#include <algorithm>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

///////////////////////////// CSRIsSorted /////////////////////////////
template <DLDeviceType XPU, typename IdType>
bool CSRIsSorted(CSRMatrix csr) {
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  const IdType* indices = csr.indices.Ptr<IdType>();
  bool ret = true;
  // (BarclayII) Rarely it will return true even if the arrays are not sorted on Windows
  // if critical section is not enabled, causing the unit test to fail.  I tested it on
  // Linux multiple times and I wasn't able to observe the same failure.  This is very
  // strange, as one of the MSVC tutorials about migrating OpenMP cancellation to its
  // PPL library used pretty much the same code as ours.  I Googled and didn't find
  // any literature on this behavior either.
  // Here I'm assuming that this is specific to VS2017-.  If someone find CSRIsSorted
  // still failing, please report a bug.
#ifdef WIN32
#pragma omp parallel for
#endif
  for (int64_t row = 0; row < csr.num_rows; ++row) {
    if (!ret)
      continue;
    for (IdType i = indptr[row] + 1; i < indptr[row + 1]; ++i) {
      if (indices[i - 1] > indices[i]) {
        ret = false;
        break;
      }
    }
  }
  return ret;
}

template bool CSRIsSorted<kDLCPU, int64_t>(CSRMatrix csr);
template bool CSRIsSorted<kDLCPU, int32_t>(CSRMatrix csr);

///////////////////////////// CSRSort /////////////////////////////

template <DLDeviceType XPU, typename IdType>
void CSRSort_(CSRMatrix* csr) {
  typedef std::pair<IdType, IdType> ShufflePair;
  const int64_t num_rows = csr->num_rows;
  const int64_t nnz = csr->indices->shape[0];
  const IdType* indptr_data = static_cast<IdType*>(csr->indptr->data);
  IdType* indices_data = static_cast<IdType*>(csr->indices->data);
  if (!CSRHasData(*csr)) {
    csr->data = aten::Range(0, nnz, csr->indptr->dtype.bits, csr->indptr->ctx);
  }
  IdType* eid_data = static_cast<IdType*>(csr->data->data);
#pragma omp parallel
  {
    std::vector<ShufflePair> reorder_vec;
#pragma omp for
    for (int64_t row = 0; row < num_rows; row++) {
      const int64_t num_cols = indptr_data[row + 1] - indptr_data[row];
      IdType *col = indices_data + indptr_data[row];
      IdType *eid = eid_data + indptr_data[row];

      reorder_vec.resize(num_cols);
      for (int64_t i = 0; i < num_cols; i++) {
        reorder_vec[i].first = col[i];
        reorder_vec[i].second = eid[i];
      }
      std::sort(reorder_vec.begin(), reorder_vec.end(),
                [](const ShufflePair &e1, const ShufflePair &e2) {
                  return e1.first < e2.first;
                });
      for (int64_t i = 0; i < num_cols; i++) {
        col[i] = reorder_vec[i].first;
        eid[i] = reorder_vec[i].second;
      }
    }
  }
  csr->sorted = true;
}

template void CSRSort_<kDLCPU, int64_t>(CSRMatrix* csr);
template void CSRSort_<kDLCPU, int32_t>(CSRMatrix* csr);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
