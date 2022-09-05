#define EIGEN_USE_THREADS


#include <algorithm>
#include <cstring>
#include <immintrin.h>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow_sparse_conv2d/cc/kernels/eigen_packet_math.h"
#include "tensorflow_sparse_conv2d/cc/kernels/transpose_util.h"


namespace tensorflow {


namespace {

static constexpr int kNumUnrollPackets = 16;

// Copy from tensorflow/core/kernels/conv_ops.h and
// tensorflow/core/kernels/conv_ops.cc

struct Conv2DDimensions {
  int batch;
  int input_rows;
  int input_cols;
  int in_depth;

  int filter_rows;
  int filter_cols;
  int patch_depth;
  int out_depth;

  int stride_rows;
  int stride_cols;

  int dilation_rows;
  int dilation_cols;

  int64 out_rows;
  int64 out_cols;
  int64 pad_rows_before;
  int64 pad_rows_after;
  int64 pad_cols_before;
  int64 pad_cols_after;

  int64 padded_input_rows;
  int64 padded_input_cols;
};


struct Conv2DParameters {
  std::vector<int32> dilations;
  std::vector<int32> strides;
  Padding padding;
  TensorFormat data_format;
};


#define TF_REQUIRES(EXP, STATUS)                \
  do {                                          \
    if (!TF_PREDICT_TRUE(EXP)) return (STATUS); \
  } while (false)

Status InitConv2DParameters(const OpKernelConstruction* context,
                            Conv2DParameters* params) {
  TF_RETURN_IF_ERROR(context->GetAttr("dilations", &params->dilations));
  TF_RETURN_IF_ERROR(context->GetAttr("strides", &params->strides));
  TF_RETURN_IF_ERROR(context->GetAttr("padding", &params->padding));
  string data_format_string;
  TF_RETURN_IF_ERROR(context->GetAttr("data_format", &data_format_string));
  TF_REQUIRES(FormatFromString(data_format_string, &params->data_format),
              errors::InvalidArgument("Invalid data format"));

  const auto& strides = params->strides;
  const auto& dilations = params->dilations;
  const auto& data_format = params->data_format;

  TF_REQUIRES(dilations.size() == 4,
              errors::InvalidArgument("Sliding window dilations field must "
                                      "specify 4 dimensions"));
  TF_REQUIRES(strides.size() == 4,
              errors::InvalidArgument("Sliding window strides field must "
                                      "specify 4 dimensions"));
  const int64 stride_n = GetTensorDim(strides, data_format, 'N');
  const int64 stride_c = GetTensorDim(strides, data_format, 'C');
  const int64 stride_h = GetTensorDim(strides, data_format, 'H');
  const int64 stride_w = GetTensorDim(strides, data_format, 'W');
  TF_REQUIRES(
      stride_n == 1 && stride_c == 1,
      errors::InvalidArgument("Current implementation does not yet support "
                              "strides in the batch and depth dimensions."));
  TF_REQUIRES(stride_h > 0 && stride_w > 0,
              errors::InvalidArgument(
                  "Row and column strides should be larger than 0."));

  const int64 dilation_n = GetTensorDim(dilations, data_format, 'N');
  const int64 dilation_c = GetTensorDim(dilations, data_format, 'C');
  const int64 dilation_h = GetTensorDim(dilations, data_format, 'H');
  const int64 dilation_w = GetTensorDim(dilations, data_format, 'W');
  TF_REQUIRES(
      dilation_n == 1 && dilation_c == 1,
      errors::InvalidArgument("Current implementation does not yet support "
                              "dilations in the batch and depth dimensions."));
  TF_REQUIRES(
      dilation_h > 0 && dilation_w > 0,
      errors::InvalidArgument("Dilated rates should be larger than 0."));

  return Status::OK();
}


Status ComputeConv2DDimension(const Conv2DParameters& params,
                              const Tensor& input, const Tensor& filter_indices_tensor,
                              const Tensor& filter_values_tensor, const Tensor& filter_shape_tensor,
                              Conv2DDimensions* dimensions) {
  const FilterTensorFormat filter_format = FORMAT_HWIO;
  // Check that 2D convolution input and filter have exactly 4 dimensions.
  TF_REQUIRES(input.dims() == 4,
              errors::InvalidArgument("input must be 4-dimensional",
                                      input.shape().DebugString()));
  TF_REQUIRES(
    TensorShapeUtils::IsMatrix(filter_indices_tensor.shape()),
    errors::InvalidArgument("filter_indices is not a matrix"));
  TF_REQUIRES(
    TensorShapeUtils::IsVector(filter_values_tensor.shape()),
    errors::InvalidArgument("filter_values is not a vector"));
  TF_REQUIRES(
    TensorShapeUtils::IsVector(filter_shape_tensor.shape()),
    errors::InvalidArgument("filter_shape is not a vector"));
  TF_REQUIRES(
    filter_indices_tensor.dim_size(1) == 4,
    errors::InvalidArgument("filter_indices.shape[1] != 4"));
  TF_REQUIRES(
    filter_shape_tensor.dim_size(0) == 4,
    errors::InvalidArgument("filter_shape.shape[0] != 4"));

  TensorShape filter_shape;
  TF_REQUIRES(
    TensorShapeUtils::MakeShape(filter_shape_tensor.vec<int64>().data(), 4, &filter_shape).ok(),
    errors::InvalidArgument("invalid filter_shape"));

  const int64 filter_nnz = filter_indices_tensor.dim_size(0);
  TF_REQUIRES(
    filter_nnz == filter_values_tensor.dim_size(0),
    errors::InvalidArgument(
      "filter_indices.shape[0] != filter_values.shape[0], "
      "got: ", filter_nnz, " and ", filter_values_tensor.dim_size(0)));

  for (int i = 0; i < 3; i++) {
    TF_REQUIRES(
        FastBoundsCheck(filter_shape.dim_size(i), std::numeric_limits<int>::max()),
        errors::InvalidArgument("filter too large"));
  }

  // The last dimension for input is in_depth. Check that it is the same as the
  // filter's in_depth or it is evenly divisible by filter's in_depth.
  const int64 in_depth_raw = GetTensorDim(input, params.data_format, 'C');
  const int64 patch_depth_raw = GetFilterDim(filter_shape, filter_format, 'I');
  TF_REQUIRES(FastBoundsCheck(in_depth_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input depth too large"));
  TF_REQUIRES(FastBoundsCheck(patch_depth_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Patch depth too large"));
  const int in_depth = static_cast<int>(in_depth_raw);
  const int patch_depth = static_cast<int>(patch_depth_raw);
  TF_REQUIRES(in_depth == patch_depth,
              errors::InvalidArgument(
                  "input and filter must have the same depth: ",
                  in_depth, " vs ", patch_depth));

  // The last dimension for filter is out_depth.
  const int out_depth = static_cast<int>(GetFilterDim(filter_shape, filter_format, 'O'));

  // The second dimension for input is rows/height.
  // The first dimension for filter is rows/height.
  const int64 input_rows_raw = GetTensorDim(input, params.data_format, 'H');
  TF_REQUIRES(FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input rows too large"));
  const int input_rows = static_cast<int>(input_rows_raw);
  const int filter_rows = static_cast<int>(GetFilterDim(filter_shape, filter_format, 'H'));

  // The third dimension for input is columns/width.
  // The second dimension for filter is columns/width.
  const int64 input_cols_raw = GetTensorDim(input, params.data_format, 'W');
  TF_REQUIRES(FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input cols too large"));
  const int input_cols = static_cast<int>(input_cols_raw);
  const int filter_cols = static_cast<int>(GetFilterDim(filter_shape, filter_format, 'W'));

  // The first dimension for input is batch.
  const int64 batch_raw = GetTensorDim(input, params.data_format, 'N');
  TF_REQUIRES(FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("batch is too large"));
  const int batch = static_cast<int>(batch_raw);

  // Take the stride and dilation from the second and third dimensions only (we
  // do not support striding or dilation on the batch or depth dimension).
  const int stride_rows = GetTensorDim(params.strides, params.data_format, 'H');
  const int stride_cols = GetTensorDim(params.strides, params.data_format, 'W');
  const int dilation_rows =
      GetTensorDim(params.dilations, params.data_format, 'H');
  const int dilation_cols =
      GetTensorDim(params.dilations, params.data_format, 'W');

  // Compute windowed output sizes for rows and columns.
  int64 out_rows = 0, out_cols = 0;
  int64 pad_rows_before = 0, pad_cols_before = 0, pad_rows_after = 0, pad_cols_after = 0;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
      input_rows, filter_rows, dilation_rows, stride_rows, params.padding,
      &out_rows, &pad_rows_before, &pad_rows_after));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
      input_cols, filter_cols, dilation_cols, stride_cols, params.padding,
      &out_cols, &pad_cols_before, &pad_cols_after));


  const int64 padded_input_rows = input_rows + pad_rows_before + pad_rows_after;
  const int64 padded_input_cols = input_cols + pad_cols_before + pad_cols_after;

  dimensions->batch = batch;
  dimensions->input_rows = input_rows;
  dimensions->input_cols = input_cols;
  dimensions->in_depth = in_depth;
  dimensions->filter_rows = filter_rows;
  dimensions->filter_cols = filter_cols;
  dimensions->patch_depth = patch_depth;
  dimensions->out_depth = out_depth;
  dimensions->stride_rows = stride_rows;
  dimensions->stride_cols = stride_cols;
  dimensions->dilation_rows = dilation_rows;
  dimensions->dilation_cols = dilation_cols;
  dimensions->out_rows = out_rows;
  dimensions->out_cols = out_cols;
  dimensions->pad_rows_before = pad_rows_before;
  dimensions->pad_cols_before = pad_cols_before;
  dimensions->pad_rows_after = pad_rows_after;
  dimensions->pad_cols_after = pad_cols_after;
  dimensions->padded_input_rows = padded_input_rows;
  dimensions->padded_input_cols = padded_input_cols;

  return Status::OK();
}

#undef TF_REQUIRES


// copy from tensorflow/core/kernels/matmul_op.cc
// Converts a TensorFlow Tensor to an Eigen Matrix.
template <typename T>
Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
ToEigenMatrix(const Tensor& tensor) {
  auto matrix = tensor.matrix<T>();
  return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(
      matrix.data(), matrix.dimension(0), matrix.dimension(1));
}


// Converts a TensorFlow Tensor to an Eigen Vector.
template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> ToEigenVector(Tensor* tensor) {
  auto v = tensor->flat<T>();
  return Eigen::Matrix<T, Eigen::Dynamic, 1>::Map(v.data(), v.dimension(0));
}


template<typename T, int InW, int OutH, int OutW>
struct ComputeConv2DUnitStrideStaticDimension {
  void operator()(const typename TTypes<T, 3>::ConstTensor& input,
                  const typename TTypes<int64>::Vec& filter_indptr,
                  const typename TTypes<int64>::Vec& filter_offsets,
                  const typename TTypes<T>::ConstVec& filter_values,
                  typename TTypes<T, 3>::Tensor& output,
                  const int64& out_depth) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    constexpr int kPacketSize = Eigen::internal::packet_traits<T>::size;
    const int64 num_col_packets = OutW / kPacketSize;
    const int64 num_row_packets = OutH;
    const int64 num_packets = num_row_packets * num_col_packets;
    const int64 num_unrolled_packets = (num_packets / kNumUnrollPackets) * kNumUnrollPackets;

    for (int64 out_channel = 0; out_channel < out_depth; ++out_channel) {
      const int64 nnz_idx_begin = filter_indptr.data()[out_channel];
      const int64 nnz_idx_end = filter_indptr.data()[out_channel + 1];

      // can unroll & full packet
      for (int64 packet_idx_begin = 0; packet_idx_begin < num_unrolled_packets; packet_idx_begin += kNumUnrollPackets) {
        Packet sum[kNumUnrollPackets];
        int64 rows[kNumUnrollPackets];
        int64 cols[kNumUnrollPackets];

        for (int64 packet_idx = 0; packet_idx < kNumUnrollPackets; ++packet_idx) {
          sum[packet_idx] = Eigen::internal::pzero<Packet>(sum[packet_idx]);

          const int64 packet_row = (packet_idx + packet_idx_begin) / num_col_packets;
          const int64 packet_col = (packet_idx + packet_idx_begin) % num_col_packets;
          rows[packet_idx] = packet_row;
          cols[packet_idx] = packet_col * kPacketSize;
        }

        for (int64 nnz_idx = nnz_idx_begin; nnz_idx < nnz_idx_end; ++nnz_idx) {
          const auto& filter_offset = filter_offsets.data()[nnz_idx];
          const auto& filter_val = filter_values.data()[nnz_idx];
          Packet w_packet = Eigen::internal::pset1<Packet>(filter_val);

          // should unroll loop
          for (int64 packet_idx = 0; packet_idx < kNumUnrollPackets; ++packet_idx) {
            const int64 row = rows[packet_idx];
            const int64 col = cols[packet_idx];
            const int64 in_idx = filter_offset + row * InW + col;

            Packet in_packet = Eigen::internal::ploadu<Packet>(input.data() + in_idx);
            sum[packet_idx] = Eigen::internal::pmadd<Packet>(w_packet, in_packet, sum[packet_idx]);
          }
        }

        for (int64 packet_idx = 0; packet_idx < kNumUnrollPackets; ++packet_idx) {
          const int64 row = rows[packet_idx];
          const int64 col = cols[packet_idx];
          const int64 out_idx = (out_channel * OutH + row) * OutW + col;
          Eigen::internal::pstoreu<T, Packet>(output.data() + out_idx, sum[packet_idx]);
        }
      }

      // can't unroll & full packet
      if (num_unrolled_packets < num_packets) {
        const int64 num_remained_packets = num_packets % kNumUnrollPackets;
        Packet sum[num_remained_packets];
        int64 rows[num_remained_packets];
        int64 cols[num_remained_packets];

        for (int64 packet_idx = 0; packet_idx < num_remained_packets; ++packet_idx) {
          sum[packet_idx] = Eigen::internal::pzero<Packet>(sum[packet_idx]);

          const int64 packet_row = (packet_idx + num_unrolled_packets) / num_col_packets;
          const int64 packet_col = (packet_idx + num_unrolled_packets) % num_col_packets;
          rows[packet_idx] = packet_row;
          cols[packet_idx] = packet_col * kPacketSize;
        }

        for (int64 nnz_idx = nnz_idx_begin; nnz_idx < nnz_idx_end; ++nnz_idx) {
          const auto& filter_offset = filter_offsets.data()[nnz_idx];
          const auto& filter_val = filter_values.data()[nnz_idx];
          Packet w_packet = Eigen::internal::pset1<Packet>(filter_val);

          for (int64 packet_idx = 0; packet_idx < num_remained_packets; ++packet_idx) {
            const int64& row = rows[packet_idx];
            const int64& col = cols[packet_idx];
            const int64 in_idx = filter_offset + row * InW + col;

            Packet in_packet = Eigen::internal::ploadu<Packet>(input.data() + in_idx);
            sum[packet_idx] = Eigen::internal::pmadd<Packet>(w_packet, in_packet, sum[packet_idx]);
          }
        }

        for (int64 packet_idx = 0; packet_idx < num_remained_packets; ++packet_idx) {
          const int64& row = rows[packet_idx];
          const int64& col = cols[packet_idx];
          const int64 out_idx = (out_channel * OutH + row) * OutW + col;
          Eigen::internal::pstoreu<T, Packet>(output.data() + out_idx, sum[packet_idx]);
        }
      }

      if (OutW % kPacketSize != 0) {
#ifdef EIGEN_VECTORIZE_AVX512
        const int64 num_unrolled_rows = (OutH / kNumUnrollPackets) * kNumUnrollPackets;
        const int64 col = num_col_packets * kPacketSize;
        typename Eigen::internal::mask_traits<Packet>::type mask = (1 << (OutW % kPacketSize)) - 1;

        // can unroll & partial packet
        for (int64 row_begin = 0; row_begin < num_unrolled_rows; row_begin += kNumUnrollPackets) {
          Packet sum[kNumUnrollPackets];
          int64 rows[kNumUnrollPackets];

          for (int64 packet_idx = 0; packet_idx < kNumUnrollPackets; ++packet_idx) {
            sum[packet_idx] = Eigen::internal::pzero<Packet>(sum[packet_idx]);
            rows[packet_idx] = row_begin + packet_idx;
          }

          for (int64 nnz_idx = nnz_idx_begin; nnz_idx < nnz_idx_end; ++nnz_idx) {
            const auto& filter_offset = filter_offsets.data()[nnz_idx];
            const auto& filter_val = filter_values.data()[nnz_idx];
            Packet w_packet = Eigen::internal::pset1<Packet>(filter_val);

            // should unroll loop
            for (int64 packet_idx = 0; packet_idx < kNumUnrollPackets; ++packet_idx) {
              const int64& row = rows[packet_idx];
              const int64 in_idx = filter_offset + row * InW + col;

              Packet in_packet = Eigen::internal::pmask_loadu<Packet>(input.data() + in_idx, mask);
              sum[packet_idx] = Eigen::internal::pmadd<Packet>(w_packet, in_packet, sum[packet_idx]);
            }
          }

          for (int64 packet_idx = 0; packet_idx < kNumUnrollPackets; ++packet_idx) {
            const int64& row = rows[packet_idx];
            const int64 out_idx = (out_channel * OutH + row) * OutW + col;
            Eigen::internal::pmask_storeu<Packet>(output.data() + out_idx, sum[packet_idx], mask);
          }
        }

        // can't unroll & partial packet
        if (num_unrolled_rows < OutH) {
          const int64 num_remained_rows = OutH % kNumUnrollPackets;
          Packet sum[num_remained_rows];
          int64 rows[num_remained_rows];

          for (int64 packet_idx = 0; packet_idx < num_remained_rows; ++packet_idx) {
            sum[packet_idx] = Eigen::internal::pzero<Packet>(sum[packet_idx]);
            rows[packet_idx] = num_unrolled_rows + packet_idx;
          }

          for (int64 nnz_idx = nnz_idx_begin; nnz_idx < nnz_idx_end; ++nnz_idx) {
            const auto& filter_offset = filter_offsets.data()[nnz_idx];
            const auto& filter_val = filter_values.data()[nnz_idx];
            Packet w_packet = Eigen::internal::pset1<Packet>(filter_val);

            for (int64 packet_idx = 0; packet_idx < num_remained_rows; ++packet_idx) {
              const int64& row = rows[packet_idx];
              const int64 in_idx = filter_offset + row * InW + col;

              Packet in_packet = Eigen::internal::pmask_loadu<Packet>(input.data() + in_idx, mask);
              sum[packet_idx] = Eigen::internal::pmadd<Packet>(w_packet, in_packet, sum[packet_idx]);
            }
          }

          for (int64 packet_idx = 0; packet_idx < num_remained_rows; ++packet_idx) {
            const int64& row = rows[packet_idx];
            const int64 out_idx = (out_channel * OutH + row) * OutW + col;
            Eigen::internal::pmask_storeu<Packet>(output.data() + out_idx, sum[packet_idx], mask);
          }
        }
#else
        for (int64 nnz_idx = nnz_idx_begin; nnz_idx < nnz_idx_end; ++nnz_idx) {
          const auto& filter_offset = filter_offsets.data()[nnz_idx];
          const auto& filter_val = filter_values.data()[nnz_idx];

          for (int64 row = 0; row < OutH; ++row) {
            for (int64 col = num_col_packets * kPacketSize; col < OutW; ++col) {
              const int64 in_idx = filter_offset + row * InW + col;
              const int64 out_idx = (out_channel * OutH + row) * OutW + col;
              output.data()[out_idx] += filter_val * input.data()[in_idx];
            }
          }
        }
#endif // EIGEN_VECTORIZE_AVX512
      }
    }
  }
};


template<typename T>
struct ComputeConv2DUnitStrideDynamicDimension {
  void operator()(const typename TTypes<T, 3>::ConstTensor& padded_input,
                  const typename TTypes<int64>::Vec& filter_indptr,
                  const typename TTypes<int64>::Vec& filter_offsets,
                  const typename TTypes<T>::ConstVec& filter_values,
                  typename TTypes<T, 3>::Tensor& output,
                  const Conv2DDimensions& dimensions) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    constexpr int kPacketSize = Eigen::internal::packet_traits<T>::size;

    const int64 num_col_packets = dimensions.out_cols / kPacketSize;
    const int64 num_row_packets = dimensions.out_rows;
    const int64 num_packets = num_row_packets * num_col_packets;
    const int64 num_unrolled_packets = (num_packets / kNumUnrollPackets) * kNumUnrollPackets;

    for (int64 out_channel = 0; out_channel < dimensions.out_depth; ++out_channel) {
      const int64 nnz_idx_begin = filter_indptr.data()[out_channel];
      const int64 nnz_idx_end = filter_indptr.data()[out_channel + 1];

      // can unroll & full packet
      for (int64 packet_idx_begin = 0; packet_idx_begin < num_unrolled_packets; packet_idx_begin += kNumUnrollPackets) {
        Packet sum[kNumUnrollPackets];
        int64 rows[kNumUnrollPackets];
        int64 cols[kNumUnrollPackets];

        for (int64 packet_idx = 0; packet_idx < kNumUnrollPackets; ++packet_idx) {
          sum[packet_idx] = Eigen::internal::pzero<Packet>(sum[packet_idx]);

          const int64 packet_row = (packet_idx + packet_idx_begin) / num_col_packets;
          const int64 packet_col = (packet_idx + packet_idx_begin) % num_col_packets;
          rows[packet_idx] = packet_row;
          cols[packet_idx] = packet_col * kPacketSize;
        }

        for (int64 nnz_idx = nnz_idx_begin; nnz_idx < nnz_idx_end; ++nnz_idx) {
          const auto& filter_offset = filter_offsets.data()[nnz_idx];
          const auto& filter_val = filter_values.data()[nnz_idx];
          Packet w_packet = Eigen::internal::pset1<Packet>(filter_val);

          // should unroll loop
          for (int64 packet_idx = 0; packet_idx < kNumUnrollPackets; ++packet_idx) {
            const int64 row = rows[packet_idx];
            const int64 col = cols[packet_idx];
            const int64 in_idx = filter_offset + row * dimensions.padded_input_cols + col;

            Packet in_packet = Eigen::internal::ploadu<Packet>(padded_input.data() + in_idx);
            sum[packet_idx] = Eigen::internal::pmadd<Packet>(w_packet, in_packet, sum[packet_idx]);
          }
        }

        for (int64 packet_idx = 0; packet_idx < kNumUnrollPackets; ++packet_idx) {
          const int64 row = rows[packet_idx];
          const int64 col = cols[packet_idx];
          const int64 out_idx = (out_channel * dimensions.out_rows + row) * dimensions.out_cols + col;
          Eigen::internal::pstoreu<T, Packet>(output.data() + out_idx, sum[packet_idx]);
        }
      }

      // can't unroll & full packet
      if (num_unrolled_packets < num_packets) {
        const int64 num_remained_packets = num_packets % kNumUnrollPackets;
        Packet sum[num_remained_packets];
        int64 rows[num_remained_packets];
        int64 cols[num_remained_packets];

        for (int64 packet_idx = 0; packet_idx < num_remained_packets; ++packet_idx) {
          sum[packet_idx] = Eigen::internal::pzero<Packet>(sum[packet_idx]);

          const int64 packet_row = (packet_idx + num_unrolled_packets) / num_col_packets;
          const int64 packet_col = (packet_idx + num_unrolled_packets) % num_col_packets;
          rows[packet_idx] = packet_row;
          cols[packet_idx] = packet_col * kPacketSize;
        }

        for (int64 nnz_idx = nnz_idx_begin; nnz_idx < nnz_idx_end; ++nnz_idx) {
          const auto& filter_offset = filter_offsets.data()[nnz_idx];
          const auto& filter_val = filter_values.data()[nnz_idx];
          Packet w_packet = Eigen::internal::pset1<Packet>(filter_val);

          for (int64 packet_idx = 0; packet_idx < num_remained_packets; ++packet_idx) {
            const int64& row = rows[packet_idx];
            const int64& col = cols[packet_idx];
            const int64 in_idx = filter_offset + row * dimensions.padded_input_cols + col;

            Packet in_packet = Eigen::internal::ploadu<Packet>(padded_input.data() + in_idx);
            sum[packet_idx] = Eigen::internal::pmadd<Packet>(w_packet, in_packet, sum[packet_idx]);
          }
        }

        for (int64 packet_idx = 0; packet_idx < num_remained_packets; ++packet_idx) {
          const int64& row = rows[packet_idx];
          const int64& col = cols[packet_idx];
          const int64 out_idx = (out_channel * dimensions.out_rows + row) * dimensions.out_cols + col;
          Eigen::internal::pstoreu<T, Packet>(output.data() + out_idx, sum[packet_idx]);
        }
      }

      if (dimensions.out_cols % kPacketSize != 0) {
#ifdef EIGEN_VECTORIZE_AVX512
        const int64 num_unrolled_rows = (dimensions.out_rows / kNumUnrollPackets) * kNumUnrollPackets;
        const int64 col = num_col_packets * kPacketSize;
        typename Eigen::internal::mask_traits<Packet>::type mask = (1 << (dimensions.out_cols % kPacketSize)) - 1;

        // can unroll & partial packet
        for (int64 row_begin = 0; row_begin < num_unrolled_rows; row_begin += kNumUnrollPackets) {
          Packet sum[kNumUnrollPackets];
          int64 rows[kNumUnrollPackets];

          for (int64 packet_idx = 0; packet_idx < kNumUnrollPackets; ++packet_idx) {
            sum[packet_idx] = Eigen::internal::pzero<Packet>(sum[packet_idx]);
            rows[packet_idx] = row_begin + packet_idx;
          }

          for (int64 nnz_idx = nnz_idx_begin; nnz_idx < nnz_idx_end; ++nnz_idx) {
            const auto& filter_offset = filter_offsets.data()[nnz_idx];
            const auto& filter_val = filter_values.data()[nnz_idx];
            Packet w_packet = Eigen::internal::pset1<Packet>(filter_val);

            // should unroll loop
            for (int64 packet_idx = 0; packet_idx < kNumUnrollPackets; ++packet_idx) {
              const int64& row = rows[packet_idx];
              const int64 in_idx = filter_offset + row * dimensions.padded_input_cols + col;

              Packet in_packet = Eigen::internal::pmask_loadu<Packet>(padded_input.data() + in_idx, mask);
              sum[packet_idx] = Eigen::internal::pmadd<Packet>(w_packet, in_packet, sum[packet_idx]);
            }
          }

          for (int64 packet_idx = 0; packet_idx < kNumUnrollPackets; ++packet_idx) {
            const int64& row = rows[packet_idx];
            const int64 out_idx = (out_channel * dimensions.out_rows + row) * dimensions.out_cols + col;
            Eigen::internal::pmask_storeu<Packet>(output.data() + out_idx, sum[packet_idx], mask);
          }
        }

        // can't unroll & partial packet
        if (num_unrolled_rows < dimensions.out_rows) {
          const int64 num_remained_rows = dimensions.out_rows % kNumUnrollPackets;
          Packet sum[num_remained_rows];
          int64 rows[num_remained_rows];

          for (int64 packet_idx = 0; packet_idx < num_remained_rows; ++packet_idx) {
            sum[packet_idx] = Eigen::internal::pzero<Packet>(sum[packet_idx]);
            rows[packet_idx] = num_unrolled_rows + packet_idx;
          }

          for (int64 nnz_idx = nnz_idx_begin; nnz_idx < nnz_idx_end; ++nnz_idx) {
            const auto& filter_offset = filter_offsets.data()[nnz_idx];
            const auto& filter_val = filter_values.data()[nnz_idx];
            Packet w_packet = Eigen::internal::pset1<Packet>(filter_val);

            for (int64 packet_idx = 0; packet_idx < num_remained_rows; ++packet_idx) {
              const int64& row = rows[packet_idx];
              const int64 in_idx = filter_offset + row * dimensions.padded_input_cols + col;

              Packet in_packet = Eigen::internal::pmask_loadu<Packet>(padded_input.data() + in_idx, mask);
              sum[packet_idx] = Eigen::internal::pmadd<Packet>(w_packet, in_packet, sum[packet_idx]);
            }
          }

          for (int64 packet_idx = 0; packet_idx < num_remained_rows; ++packet_idx) {
            const int64& row = rows[packet_idx];
            const int64 out_idx = (out_channel * dimensions.out_rows + row) * dimensions.out_cols + col;
            Eigen::internal::pmask_storeu<Packet>(output.data() + out_idx, sum[packet_idx], mask);
          }
        }
#else
        T* begin = output.data() + out_channel * dimensions.out_rows + dimensions.out_cols + num_col_packets * kPacketSize;
        const int64 num_remained_cols = dimensions.out_cols % kPacketSize;
        for (int64 row = 0; row < dimensions.out_rows; ++row) {
          std::fill(begin, begin + num_remained_cols, T(0));
          begin += dimensions.out_cols;
        }
        for (int64 nnz_idx = nnz_idx_begin; nnz_idx < nnz_idx_end; ++nnz_idx) {
          const auto& filter_offset = filter_offsets.data()[nnz_idx];
          const auto& filter_val = filter_values.data()[nnz_idx];

          for (int64 row = 0; row < dimensions.out_rows; ++row) {
            for (int64 col = num_col_packets * kPacketSize; col < dimensions.out_cols; ++col) {
              const int64 in_idx = filter_offset + row * dimensions.padded_input_cols + col;
              const int64 out_idx = (out_channel * dimensions.out_rows + row) * dimensions.out_cols + col;
              output.data()[out_idx] += filter_val * padded_input.data()[in_idx];
            }
          }
        }
#endif // EIGEN_VECTORIZE_AVX512
      }
    }
  }
};


// data format is NCHW
template<typename T>
void PadInput(const Conv2DDimensions& dimensions, const T* input, T* output) {
  for (int64 c = 0; c < dimensions.in_depth; ++c) {
    output += dimensions.pad_rows_before * dimensions.padded_input_cols;
    for (int64 i = 0; i < dimensions.input_rows; ++i) {
      memcpy(output + dimensions.pad_cols_before, input, sizeof(T) * dimensions.input_cols);
      output += dimensions.padded_input_cols;
      input += dimensions.input_cols;
    }
    output += dimensions.pad_rows_after * dimensions.padded_input_cols;
  }
}

} // namespace


template<typename T>
class SparseConv2DOp : public OpKernel {
 public:
  explicit SparseConv2DOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, InitConv2DParameters(ctx, &params_));

    const int32 dilation_h = GetTensorDim(params_.dilations, params_.data_format, 'H');
    const int32 dilation_w = GetTensorDim(params_.dilations, params_.data_format, 'W');
    OP_REQUIRES(
      ctx,
      dilation_h == 1 && dilation_w == 1,
      errors::InvalidArgument(
        "Current implementation does not yet support "
        "dilations in the height and width dimensions."));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    const Tensor& filter_indices_tensor = ctx->input(1);
    const Tensor& filter_values_tensor = ctx->input(2);
    const Tensor& filter_shape_tensor = ctx->input(3);

    Conv2DDimensions dimensions;
    OP_REQUIRES_OK(
      ctx,
      ComputeConv2DDimension(
        params_, input_tensor, filter_indices_tensor,
        filter_values_tensor, filter_shape_tensor, &dimensions));

    TensorShape out_shape = ShapeFromFormat(
      params_.data_format, dimensions.batch, dimensions.out_rows,
      dimensions.out_cols, dimensions.out_depth);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output_tensor));

    if (out_shape.num_elements() == 0) {
      return;
    }

    const auto input = input_tensor.tensor<T, 4>();
    const int64 filter_nnz = filter_indices_tensor.dim_size(0);
    const auto filter_indices = filter_indices_tensor.matrix<int64>();
    const auto filter_values = filter_values_tensor.vec<T>();
    auto output = output_tensor->tensor<T, 4>();

    // calculate (in_channel * padded_input_rows + filter_row) * padded_input_cols + filter_col
    Tensor filter_offsets_tensor;
    OP_REQUIRES_OK(
      ctx,
      ctx->allocate_temp(
        DataTypeToEnum<int64>::value,
        filter_values_tensor.shape(),
        &filter_offsets_tensor));
    {
      auto a_m = ToEigenMatrix<int64>(filter_indices_tensor);
      auto out_v = ToEigenVector<int64>(&filter_offsets_tensor);
      Eigen::Matrix<int64, 4, 1> coeff;
      coeff << dimensions.padded_input_cols, 1, dimensions.padded_input_rows * dimensions.padded_input_cols, 0;
      out_v.noalias() = a_m * coeff;
    }
    auto filter_offsets = filter_offsets_tensor.vec<int64>();

    // calculate CSR format
    Tensor filter_indptr_tensor;
    OP_REQUIRES_OK(
      ctx,
      ctx->allocate_temp(
        DataTypeToEnum<int64>::value,
        TensorShape({dimensions.out_depth + 1}),
        &filter_indptr_tensor));
    auto filter_indptr = filter_indptr_tensor.vec<int64>();
    std::fill(filter_indptr.data(), filter_indptr.data() + dimensions.out_depth + 1, int64(0));
    filter_indptr.data()[0] = 0;
    for (int i = 0; i < filter_nnz; ++i) {
      const auto& out_channel = filter_indices.data()[i * 4 + 3];
      ++filter_indptr.data()[out_channel + 1];
    }
    std::partial_sum(filter_indptr.data(), filter_indptr.data() + dimensions.out_depth + 1, filter_indptr.data());

    Tensor padded_input_tensor;
    if (NeedPadding(dimensions)) {
      OP_REQUIRES_OK(
        ctx,
        ctx->allocate_temp(
          DataTypeToEnum<T>::value,
          TensorShape({dimensions.in_depth, dimensions.padded_input_rows, dimensions.padded_input_cols}),
          &padded_input_tensor));
      auto padded_input = padded_input_tensor.flat<T>();
      std::fill(padded_input.data(), padded_input.data() + dimensions.in_depth * dimensions.padded_input_rows * dimensions.padded_input_cols, T(0));
    }

    Eigen::array<Eigen::IndexPair<int>, 3> eigen_padding;
    eigen_padding[0] = Eigen::IndexPair<int>(0, 0);
    eigen_padding[1] = Eigen::IndexPair<int>(dimensions.pad_rows_before, dimensions.pad_rows_after);
    eigen_padding[2] = Eigen::IndexPair<int>(dimensions.pad_cols_before, dimensions.pad_cols_after);

    Tensor trans_input_tensor;
    Tensor trans_output_tensor;
    if (NeedTranspose()) {
      OP_REQUIRES_OK(
        ctx,
        ctx->allocate_temp(
          DataTypeToEnum<T>::value,
          TensorShape({dimensions.in_depth, dimensions.input_rows, dimensions.input_cols}),
          &trans_input_tensor));
      OP_REQUIRES_OK(
        ctx,
        ctx->allocate_temp(
          DataTypeToEnum<T>::value,
          TensorShape({dimensions.out_depth, dimensions.out_rows, dimensions.out_cols}),
          &trans_output_tensor));
    }

    const int64 input_spatial_size = dimensions.input_rows * dimensions.input_cols;
    const int64 input_image_size = input_spatial_size * dimensions.in_depth;
    const int64 output_spatial_size = dimensions.out_rows * dimensions.out_cols;
    const int64 output_image_size = output_spatial_size * dimensions.out_depth;

    for (int64 b = 0; b < dimensions.batch; ++b) {
      const T* trans_input_ptr = nullptr;
      T* trans_output_ptr = nullptr;
      if (NeedTranspose()) {
        Transpose<T>(
            input.data() + b * input_image_size,
            trans_input_tensor.flat<T>().data(),
            input_spatial_size, dimensions.in_depth);
        trans_input_ptr = trans_input_tensor.flat<T>().data();
        trans_output_ptr = trans_output_tensor.flat<T>().data();
      } else {
        trans_input_ptr = input.data() + b * input_image_size;
        trans_output_ptr = output.data() + b * output_image_size;
      }
      typename TTypes<T, 3>::Tensor trans_output(trans_output_ptr, dimensions.out_depth, dimensions.out_rows, dimensions.out_cols);

      const T* padded_input_ptr = nullptr;
      if (NeedPadding(dimensions)) {
        PadInput(
            dimensions,
            trans_input_ptr,
            padded_input_tensor.flat<T>().data());
        padded_input_ptr = padded_input_tensor.flat<T>().data();
      } else {
        padded_input_ptr = trans_input_ptr;
      }
      typename TTypes<T, 3>::ConstTensor padded_input(padded_input_ptr, dimensions.in_depth, dimensions.padded_input_rows, dimensions.padded_input_cols);

      if (dimensions.stride_rows == 1 && dimensions.stride_cols == 1 && dimensions.out_rows > 1) {
        // TODO: use a registry similar to op registry
        // SSH VGG 800
        if (dimensions.out_rows == 800 && dimensions.out_cols == 1200 && dimensions.padded_input_cols == 1202) {
          ComputeConv2DUnitStrideStaticDimension<T, 1202, 800, 1200>()(padded_input, filter_indptr, filter_offsets, filter_values, trans_output, dimensions.out_depth);
        } else if (dimensions.out_rows == 400 && dimensions.out_cols == 600 && dimensions.padded_input_cols == 602) {
          ComputeConv2DUnitStrideStaticDimension<T, 602, 400, 600>()(padded_input, filter_indptr, filter_offsets, filter_values, trans_output, dimensions.out_depth);
        } else if (dimensions.out_rows == 200 && dimensions.out_cols == 300 && dimensions.padded_input_cols == 302) {
          ComputeConv2DUnitStrideStaticDimension<T, 302, 200, 300>()(padded_input, filter_indptr, filter_offsets, filter_values, trans_output, dimensions.out_depth);
        } else if (dimensions.out_rows == 100 && dimensions.out_cols == 150 && dimensions.padded_input_cols == 152) {
          ComputeConv2DUnitStrideStaticDimension<T, 152, 100, 150>()(padded_input, filter_indptr, filter_offsets, filter_values, trans_output, dimensions.out_depth);
        } else if (dimensions.out_rows == 100 && dimensions.out_cols == 150 && dimensions.padded_input_cols == 150) {
          ComputeConv2DUnitStrideStaticDimension<T, 150, 100, 150>()(padded_input, filter_indptr, filter_offsets, filter_values, trans_output, dimensions.out_depth);
        } else if (dimensions.out_rows == 50 && dimensions.out_cols == 75 && dimensions.padded_input_cols == 77) {
          ComputeConv2DUnitStrideStaticDimension<T, 77, 50, 75>()(padded_input, filter_indptr, filter_offsets, filter_values, trans_output, dimensions.out_depth);
        } else if (dimensions.out_rows == 50 && dimensions.out_cols == 75 && dimensions.padded_input_cols == 75) {
          ComputeConv2DUnitStrideStaticDimension<T, 75, 50, 75>()(padded_input, filter_indptr, filter_offsets, filter_values, trans_output, dimensions.out_depth);
        } else if (dimensions.out_rows == 25 && dimensions.out_cols == 37 && dimensions.padded_input_cols == 39) {
          ComputeConv2DUnitStrideStaticDimension<T, 39, 25, 37>()(padded_input, filter_indptr, filter_offsets, filter_values, trans_output, dimensions.out_depth);
        } else if (dimensions.out_rows == 25 && dimensions.out_cols == 37 && dimensions.padded_input_cols == 37) {
          ComputeConv2DUnitStrideStaticDimension<T, 37, 25, 37>()(padded_input, filter_indptr, filter_offsets, filter_values, trans_output, dimensions.out_depth);
        } else {
          ComputeConv2DUnitStrideDynamicDimension<T>()(padded_input, filter_indptr, filter_offsets, filter_values, trans_output, dimensions);
        }
      } else {
        std::fill(trans_output.data(), trans_output.data() + output_image_size, T(0));

        for (int64 nnz_idx = 0; nnz_idx < filter_nnz; ++nnz_idx) {
          const auto& out_channel = filter_indices.data()[nnz_idx * 4 + 3];
          const auto& in_channel = filter_indices.data()[nnz_idx * 4 + 2];
          const auto& filter_row = filter_indices.data()[nnz_idx * 4];
          const auto& filter_col = filter_indices.data()[nnz_idx * 4 + 1];
          const auto& filter_val = filter_values.data()[nnz_idx];

          const int64 out_row_min = std::max(
              0LL,
              static_cast<int64>(ceil((dimensions.pad_rows_before - filter_row) / static_cast<float>(dimensions.stride_rows))));
          const int64 out_row_max = std::min(
              dimensions.out_rows,
              static_cast<int64>(ceil((dimensions.input_rows + dimensions.pad_rows_before - filter_row) / static_cast<float>(dimensions.stride_rows))));
          const int64 out_col_min = std::max(
              0LL,
              static_cast<int64>(ceil((dimensions.pad_cols_before - filter_col) / static_cast<float>(dimensions.stride_cols))));
          const int64 out_col_max = std::min(
              dimensions.out_cols,
              static_cast<int64>(ceil((dimensions.input_cols + dimensions.pad_cols_before - filter_col) / static_cast<float>(dimensions.stride_cols))));

          auto* out_base = output.data() + (b * dimensions.out_depth + out_channel) * output_spatial_size;
          auto* in_base = input.data() + (b * dimensions.in_depth + in_channel) * input_spatial_size;
          for (int64 i = out_row_min; i < out_row_max; ++i) {
            for (int64 j = out_col_min; j < out_col_max; ++j) {
              const int64 out_idx = i * dimensions.out_cols + j;
              const int64 in_row = filter_row + i * dimensions.stride_rows - dimensions.pad_rows_before;
              const int64 in_col = filter_col + j * dimensions.stride_cols - dimensions.pad_cols_before;
              const int64 in_idx = in_row * dimensions.input_cols + in_col;
              out_base[out_idx] += filter_val * in_base[in_idx];
            }
          }
        }
      }

      if (NeedTranspose()) {
        Transpose<T>(
            trans_output.data(),
            output.data() + b * output_image_size,
            dimensions.out_depth, output_spatial_size);
      }
    }
  }

 private:
  inline bool NeedPadding(const Conv2DDimensions& dimensions) const {
    // pad_after always >= pad_before
    return dimensions.pad_rows_after > 0 || dimensions.pad_cols_after > 0;
  }

  inline bool NeedTranspose() const {
    return false;
    //return params_.data_format == FORMAT_NHWC;
  }

  typedef typename Eigen::internal::packet_traits<T>::type Packet;
  static const int kPacketSize = Eigen::internal::packet_traits<T>::size;

  Conv2DParameters params_;
};



REGISTER_KERNEL_BUILDER(
  Name("SparseConv2D").Device(DEVICE_CPU).TypeConstraint<float>("T"),
  SparseConv2DOp<float>);

} // namespace tensorflow
