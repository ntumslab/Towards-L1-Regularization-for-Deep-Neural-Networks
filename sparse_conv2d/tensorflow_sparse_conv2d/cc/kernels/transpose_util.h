#ifndef TENSORFLOW_SPARSE_CONV2D_CC_KERNELS_TRANSPOSE_UTIL_H
#define TENSORFLOW_SPARSE_CONV2D_CC_KERNELS_TRANSPOSE_UTIL_H


#include "tensorflow/core/framework/types.h"


namespace tensorflow {


namespace {


static constexpr int64 kBlockSize = 16;


template <typename T>
inline void TransposeBlock(const T* input, T* output, const int64 M, const int64 N) {
  for (int64 j = 0; j < kBlockSize; ++j) {
    for (int64 i = 0; i < kBlockSize; ++i) {
      output[j * M + i] = input[i * N + j];
    }
  }
}


template <typename T>
inline void TransposeRemain(const T* input, T* output, const int64 M, const int64 N, const int64 rows, const int64 cols) {
  for (int64 j = 0; j < cols; ++j) {
    for (int64 i = 0; i < rows; ++i) {
      output[j * M + i] = input[i * N + j];
    }
  }
}

} // namespace


// input: M x N, output: N x M
template <typename T>
void Transpose(const T* input, T* output, const int64 M, const int64 N) {
  const int64 M2 = (M / kBlockSize) * kBlockSize;
  const int64 N2 = (N / kBlockSize) * kBlockSize;

  for (int64 j = 0; j < N2; j += kBlockSize) {
    for (int64 i = 0; i < M2; i += kBlockSize) {
      TransposeBlock(input + i * N + j, output + j * M + i, M, N);
    }
    if (M2 < M) {
      TransposeRemain(input + M2 * N + j, output + j * M + M2, M, N, M - M2, kBlockSize);
    }
  }
  if (N2 < N) {
    for (int64 i = 0; i < M2; i += kBlockSize) {
      TransposeRemain(input + i * N + N2, output + N2 * M + i, M, N, kBlockSize, N - N2);
    }
    if (M2 < M) {
      TransposeRemain(input + M2 * N + N2, output + N2 * M + M2, M, N, M - M2, N - N2);
    }
  }
}


} // namespace tensorflow

#endif // TENSORFLOW_SPARSE_CONV2D_CC_KERNELS_TRANSPOSE_UTIL_H
