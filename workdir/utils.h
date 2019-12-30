namespace dgl {

/*! \brief ordered set with index_or_insert functionality */
template<typename Value>
struct biset {
  std::vector<Value> values;
  std::unordered_map<Value, size_t> lookup_table;

  /*!
   * \brief find the index of an object, or insert the object if it does not exist.
   */
  size_t index_or_insert(Value v) {
    auto search = lookup_table.find(v);
    if (search == lookup_table.end()) {
      size_t result = values.size();
      lookup_table[v] = result;
      values.push_back(v);
      return result;
    } else
      return search->second;
  }
};

template<typename T>
NDArray VecToFloatArray(const std::vector<T> &vec,
                        uint8_t nbits,
                        DLContext ctx) {
  NDArray ret = NDArray::Empty(
      {vec.size()}, DLDataType{kDLFloat, nbits, 1}, DLContext{kDLCPU, 0});
  if (nbits == 32) {
    std::copy(vec.begin(), vec.end(), static_cast<float *>(ret->data));
  } else if (nbits == 64) {
    std::copy(vec.begin(), vec.end(), static_cast<double *>(ret->data));
  } else {
    LOG(FATAL) << "Only float32 or float64 is supported.";
  }
  return ret.CopyTo(ctx);
}

IdArray Unique(IdArray a) {
  ATEN_DTYPE_SWITCH(a.dtype, DType, {
    std::unordered_set<DType> set;
    int64_t n = a->shape[0];
    const DType *data = static_cast<DType *>(a->data);
    for (int64_t i = 0; i < n; ++i)
      set.insert(data[i]);

    IdArray result = NewIdArray(set.size(), DLContext{kDLCPU, 0}, a.dtype.nbits);
    DType *result_data = static_cast<DType *>(result->data);
    int64_t i = 0;
    for (DType v : set)
      result_data[i++] = v;

    return result;
  });
}

};  // namespace dgl
