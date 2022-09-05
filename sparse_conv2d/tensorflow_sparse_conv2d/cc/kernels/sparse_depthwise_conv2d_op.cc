#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow_sparse_conv2d/cc/kernels/transpose_util.h"

#if SCONV_PARALLEL
#include "tensorflow/core/util/work_sharder.h"
#endif // SCONV_PARALLEL


namespace tensorflow {


typedef Eigen::ThreadPoolDevice CPUDevice;


namespace {
// Copy from tensorflow/core/kernels/conv_ops.h and

struct DepthwiseConv2DDimensions {
  int batch;
  int input_rows;
  int input_cols;
  int in_depth;

  int filter_rows;
  int filter_cols;
  int depth_multiplier;
  int out_depth;

  int stride;

  int64 out_rows;
  int64 out_cols;
  int64 pad_rows;
  int64 pad_cols;
};


struct DepthwiseConv2DParameters {
  // dense depthwise conv2d native ignores dilations
  // See https://github.com/tensorflow/tensorflow/issues/19992
  std::vector<int32> dilations;
  std::vector<int32> strides;
  Padding padding;
  TensorFormat data_format = FORMAT_NHWC;
};


#define TF_REQUIRES(EXP, STATUS)                \
  do {                                          \
    if (!TF_PREDICT_TRUE(EXP)) return (STATUS); \
  } while (false)

Status InitConv2DParameters(const OpKernelConstruction* context,
                            DepthwiseConv2DParameters* params) {
  TF_RETURN_IF_ERROR(context->GetAttr("dilations", &params->dilations));
  TF_RETURN_IF_ERROR(context->GetAttr("strides", &params->strides));
  TF_RETURN_IF_ERROR(context->GetAttr("padding", &params->padding));

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
  TF_REQUIRES(
      stride_h == stride_w,
      errors::InvalidArgument("Current implementation only supports equal length "
                              "strides in the row and column dimensions."));
  TF_REQUIRES(stride_h > 0,
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
      dilation_h == 1 && dilation_w == 1,
      errors::InvalidArgument("Tensorflow ignores dliations in dense DepthwiseConv2dNativeOp."));

  return Status::OK();
}


Status ComputeConv2DDimension(const DepthwiseConv2DParameters& params,
                              const Tensor& input, const Tensor& filter_indices_tensor,
                              const Tensor& filter_values_tensor, const Tensor& filter_shape_tensor,
                              DepthwiseConv2DDimensions* dimensions) {
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

  const int depth_multiplier = static_cast<int>(GetFilterDim(filter_shape, filter_format, 'O'));
  const int out_depth = in_depth * depth_multiplier;

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
  // stride_rows == stride_cols
  const int stride = GetTensorDim(params.strides, params.data_format, 'H');

  // Compute windowed output sizes for rows and columns.
  int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
  TF_RETURN_IF_ERROR(GetWindowedOutputSize(
      input_rows, filter_rows, stride, params.padding,
      &out_rows, &pad_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSize(
      input_cols, filter_cols, stride, params.padding,
      &out_cols, &pad_cols));

  dimensions->batch = batch;
  dimensions->input_rows = input_rows;
  dimensions->input_cols = input_cols;
  dimensions->in_depth = in_depth;
  dimensions->filter_rows = filter_rows;
  dimensions->filter_cols = filter_cols;
  dimensions->depth_multiplier = depth_multiplier;
  dimensions->out_depth = out_depth;
  dimensions->stride = stride;
  dimensions->out_rows = out_rows;
  dimensions->out_cols = out_cols;
  dimensions->pad_rows = pad_rows;
  dimensions->pad_cols = pad_cols;

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

} // namespace


template<typename T>
class SparseDepthwiseConv2dNativeOp : public OpKernel {
 public:
  explicit SparseDepthwiseConv2dNativeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, InitConv2DParameters(ctx, &params_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    const Tensor& filter_indices_tensor = ctx->input(1);
    const Tensor& filter_values_tensor = ctx->input(2);
    const Tensor& filter_shape_tensor = ctx->input(3);

    DepthwiseConv2DDimensions dimensions;
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

    const auto input = input_tensor.flat<T>();
    const int64 filter_nnz = filter_indices_tensor.dim_size(0);
    const auto filter_indices = filter_indices_tensor.matrix<int64>();
    const auto filter_values = filter_values_tensor.vec<T>();
    auto output = output_tensor->flat<T>();

#if 0
    // calculate h * in_cols * in_depth + w * in_depth + i - (pad_rows * W * I + pad_cols * I)
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
      coeff << dimensions.input_cols * dimensions.in_depth, dimensions.in_depth, 1, 0;
      out_v.noalias() = ((a_m * coeff).array() - (dimensions.pad_rows * dimensions.input_cols + dimensions.pad_cols) * dimensions.in_depth).matrix();
    }
    auto filter_offsets = filter_offsets_tensor.vec<int64>();

    // calculate out_channel = in_channel * depth_multiplier + indices[3]
    Tensor out_channels_tensor;
    OP_REQUIRES_OK(
      ctx,
      ctx->allocate_temp(
        DataTypeToEnum<int64>::value,
        filter_values_tensor.shape(),
        &out_channels_tensor));
    {
      auto mat = ToEigenMatrix<int64>(filter_indices_tensor);
      auto in_channels = mat.col(2);
      auto depth_multiplier_indices = mat.col(3);
      auto out_v = ToEigenVector<int64>(&out_channels_tensor);
      out_v.noalias() = (in_channels * dimensions.depth_multiplier + depth_multiplier_indices).eval().transpose();
    }
    auto out_channels = out_channels_tensor.vec<int64>();

    output.setZero();

    T* output_ptr = output.data();
    const T* input_ptr = input.data();

    for (int64 b = 0; b < dimensions.batch; ++b) {
      int64 in_row_offset = -dimensions.pad_rows;
      const T* input_ptr_row = input_ptr;

      for (int64 i = 0; i < dimensions.out_rows; ++i) {
        int64 in_col_offset = -dimensions.pad_cols;
        const T* input_ptr_col = input_ptr_row;
        const bool bounds_check_row = !FastBoundsCheck(in_row_offset, dimensions.input_rows - dimensions.filter_rows + 1);

        for (int64 j = 0; j < dimensions.out_cols; ++j) {
          const bool bounds_check_col = !(FastBoundsCheck(in_col_offset, dimensions.input_cols - dimensions.filter_cols + 1));

          if (bounds_check_row || bounds_check_col) {
            for (int64 nnz_idx = 0; nnz_idx < filter_nnz; ++nnz_idx) {
              const auto& out_channel = out_channels.data()[nnz_idx];
              const auto& filter_row = filter_indices.data()[nnz_idx * 4];
              const auto& filter_col = filter_indices.data()[nnz_idx * 4 + 1];
              const auto& filter_val = filter_values.data()[nnz_idx];
              const auto& filter_offset = filter_offsets.data()[nnz_idx];

              const int64 in_row = filter_row + in_row_offset;;
              const int64 in_col = filter_col + in_col_offset;
              if (!FastBoundsCheck(in_row, dimensions.input_rows) || !FastBoundsCheck(in_col, dimensions.input_cols)) continue;
              output_ptr[out_channel] += filter_val * input_ptr_col[filter_offset];
            }
          } else {
            for (int64 nnz_idx = 0; nnz_idx < filter_nnz; ++nnz_idx) {
              const auto& out_channel = out_channels.data()[nnz_idx];
              const auto& filter_val = filter_values.data()[nnz_idx];
              const auto& filter_offset = filter_offsets.data()[nnz_idx];

              output_ptr[out_channel] += filter_val * input_ptr_col[filter_offset];
            }
          }

          output_ptr += dimensions.out_depth;
          in_col_offset += dimensions.stride;
          input_ptr_col += dimensions.stride * dimensions.in_depth;
        }

        in_row_offset += dimensions.stride;
        input_ptr_row += dimensions.stride * dimensions.input_cols * dimensions.in_depth;
      }

      input_ptr += dimensions.input_rows * dimensions.input_cols * dimensions.in_depth;
    }
#else
    Tensor trans_input_tensor;
    OP_REQUIRES_OK(
      ctx,
      ctx->allocate_temp(
        DataTypeToEnum<T>::value,
        TensorShape({dimensions.in_depth, dimensions.input_rows, dimensions.input_cols}),
        &trans_input_tensor));
    auto trans_input = trans_input_tensor.flat<T>();

    Tensor trans_output_tensor;
    OP_REQUIRES_OK(
      ctx,
      ctx->allocate_temp(
        DataTypeToEnum<T>::value,
        TensorShape({dimensions.out_depth, dimensions.out_rows, dimensions.out_cols}),
        &trans_output_tensor));
    auto trans_output = trans_output_tensor.flat<T>();

    const int64 input_spatial_size = dimensions.input_rows * dimensions.input_cols;
    const int64 input_image_size = input_spatial_size * dimensions.in_depth;
    const int64 output_spatial_size = dimensions.out_rows * dimensions.out_cols;
    const int64 output_image_size = output_spatial_size * dimensions.out_depth;

    for (int64 b = 0; b < dimensions.batch; ++b) {
      Transpose<T>(
          input.data() + b * input_image_size,
          trans_input.data(),
          input_spatial_size, dimensions.in_depth);
      std::fill(trans_output.data(), trans_output.data() + output_image_size, T(0));

      // special case
      if (dimensions.stride == 1 && dimensions.filter_rows == 1 && dimensions.filter_cols == 1) {
        for (int64 nnz_idx = 0; nnz_idx < filter_nnz; ++nnz_idx) {
          const auto& depth_multiplier_index = filter_indices.data()[nnz_idx * 4 + 3];
          const auto& in_channel = filter_indices.data()[nnz_idx * 4 + 2];
          const auto& filter_val = filter_values.data()[nnz_idx];
          const auto out_channel = in_channel * dimensions.depth_multiplier + depth_multiplier_index;

          auto* out_base = trans_output.data() + out_channel * output_spatial_size;
          auto* in_base = trans_input.data() + in_channel * input_spatial_size;
          for (int64 i = 0; i < output_spatial_size; ++i) {
            out_base[i] += filter_val * in_base[i];
          }
        }
      } else {
        for (int64 nnz_idx = 0; nnz_idx < filter_nnz; ++nnz_idx) {
          const auto& depth_multiplier_index = filter_indices.data()[nnz_idx * 4 + 3];
          const auto& in_channel = filter_indices.data()[nnz_idx * 4 + 2];
          const auto& filter_row = filter_indices.data()[nnz_idx * 4];
          const auto& filter_col = filter_indices.data()[nnz_idx * 4 + 1];
          const auto& filter_val = filter_values.data()[nnz_idx];
          const auto out_channel = in_channel * dimensions.depth_multiplier + depth_multiplier_index;

          const int64 out_row_min = std::max(
              0LL,
              static_cast<int64>(ceil((dimensions.pad_rows - filter_row) / static_cast<float>(dimensions.stride))));
          const int64 out_row_max = std::min(
              dimensions.out_rows,
              static_cast<int64>(ceil((dimensions.input_rows + dimensions.pad_rows - filter_row) / static_cast<float>(dimensions.stride))));
          const int64 out_col_min = std::max(
              0LL,
              static_cast<int64>(ceil((dimensions.pad_cols - filter_col) / static_cast<float>(dimensions.stride))));
          const int64 out_col_max = std::min(
              dimensions.out_cols,
              static_cast<int64>(ceil((dimensions.input_cols + dimensions.pad_cols - filter_col) / static_cast<float>(dimensions.stride))));

          for (int64 i = out_row_min; i < out_row_max; ++i) {
            for (int64 j = out_col_min; j < out_col_max; ++j) {
              const int64 out_idx = (out_channel * dimensions.out_rows + i) * dimensions.out_cols + j;
              const int64 in_row = filter_row + i * dimensions.stride - dimensions.pad_rows;
              const int64 in_col = filter_col + j * dimensions.stride - dimensions.pad_cols;
              const int64 in_idx = (in_channel * dimensions.input_rows + in_row) * dimensions.input_cols + in_col;
              trans_output.data()[out_idx] += filter_val * trans_input.data()[in_idx];
            }
          }
        }
      }

      Transpose<T>(
          trans_output.data(),
          output.data() + b * output_image_size,
          dimensions.out_depth, output_spatial_size);
    }
#endif
  }

 private:
  DepthwiseConv2DParameters params_;
};


REGISTER_KERNEL_BUILDER(
  Name("SparseDepthwiseConv2dNative").Device(DEVICE_CPU).TypeConstraint<float>("T"),
  SparseDepthwiseConv2dNativeOp<float>);
REGISTER_KERNEL_BUILDER(
  Name("SparseDepthwiseConv2dNative").Device(DEVICE_CPU).TypeConstraint<double>("T"),
  SparseDepthwiseConv2dNativeOp<double>);

} // namespace tensorflow
