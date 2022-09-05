#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"


namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status SparseConv2DShape(InferenceContext* c) {
  constexpr int num_spatial_dims = 2;
  const FilterTensorFormat filter_format = FORMAT_HWIO;

  string data_format_str;
  if (!c->GetAttr("data_format", &data_format_str).ok()) {
    data_format_str = "NHWC";
  }
  TensorFormat data_format;
  if (!FormatFromString(data_format_str, &data_format)) {
    return errors::InvalidArgument("Invalid data format string: ",
                                   data_format_str);
  }

  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

  {
    ShapeHandle filter_indices_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &filter_indices_shape));
    DimensionHandle filter_indices_dim1 = c->Dim(filter_indices_shape, 1);
    TF_RETURN_IF_ERROR(c->WithValue(filter_indices_dim1, 4, &filter_indices_dim1));
  }

  {
    ShapeHandle unused;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused)); // filter_values
  }

  ShapeHandle filter_shape;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &filter_shape));
  TF_RETURN_IF_ERROR(c->WithRank(filter_shape, 4, &filter_shape));

  std::vector<int32> dilations;
  TF_RETURN_IF_ERROR(c->GetAttr("dilations", &dilations));

  if (dilations.size() != 4) {
    return errors::InvalidArgument(
      "SparseConv2D requires the dilations attribute to contain "
      "4 values, but got: ", dilations.size());
  }

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));

  if (strides.size() != 4) {
    return errors::InvalidArgument(
      "SparseConv2D requires the strides attribute to contain "
      "4 values, but got: ", strides.size());
  }

  const int32 stride_rows = GetTensorDim(strides, data_format, 'H');
  const int32 stride_cols = GetTensorDim(strides, data_format, 'W');
  const int32 dilation_rows = GetTensorDim(dilations, data_format, 'H');
  const int32 dilation_cols = GetTensorDim(dilations, data_format, 'W');

  DimensionHandle batch_size_dim = c->Dim(
    input_shape,
    GetTensorDimIndex(data_format, 'N'));
  DimensionHandle input_rows_dim = c->Dim(
    input_shape,
    GetTensorDimIndex(data_format, 'H'));
  DimensionHandle input_cols_dim = c->Dim(
    input_shape,
    GetTensorDimIndex(data_format, 'W'));
  DimensionHandle input_depth_dim = c->Dim(
    input_shape,
    GetTensorDimIndex(data_format, 'C'));

  DimensionHandle output_depth_dim = c->Dim(
    filter_shape,
    GetFilterDimIndex<num_spatial_dims>(filter_format, 'O'));
  DimensionHandle filter_input_depth_dim = c->Dim(
    filter_shape,
    GetFilterDimIndex<num_spatial_dims>(filter_format, 'I'));
  DimensionHandle filter_rows_dim = c->Dim(
    filter_shape,
    GetFilterDimIndex<num_spatial_dims>(filter_format, 'H'));
  DimensionHandle filter_cols_dim = c->Dim(
    filter_shape,
    GetFilterDimIndex<num_spatial_dims>(filter_format, 'W'));

  {
    DimensionHandle unused;
    TF_RETURN_IF_ERROR(c->Merge(input_depth_dim, filter_input_depth_dim, &unused));
  }

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

  DimensionHandle output_rows_dim, output_cols_dim;
  TF_RETURN_IF_ERROR(
    GetWindowedOutputSizeFromDimsV2(
      c, input_rows_dim, filter_rows_dim, dilation_rows, stride_rows,
      padding, &output_rows_dim));
  TF_RETURN_IF_ERROR(
    GetWindowedOutputSizeFromDimsV2(
      c, input_cols_dim, filter_cols_dim, dilation_cols, stride_cols,
      padding, &output_cols_dim));

  ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(
    shape_inference::MakeShapeFromFormat(
      data_format, batch_size_dim, {output_rows_dim, output_cols_dim},
      output_depth_dim, &output_shape, c));
  c->set_output(0, output_shape);

  return Status::OK();
}


Status SparseDepthwiseConv2DNativeShape(InferenceContext* c) {
  constexpr int num_spatial_dims = 2;
  const TensorFormat data_format = FORMAT_NHWC;
  const FilterTensorFormat filter_format = FORMAT_HWIO;

  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

  {
    ShapeHandle filter_indices_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &filter_indices_shape));
    DimensionHandle filter_indices_dim1 = c->Dim(filter_indices_shape, 1);
    TF_RETURN_IF_ERROR(c->WithValue(filter_indices_dim1, 4, &filter_indices_dim1));
  }

  {
    ShapeHandle unused;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused)); // filter_values
  }

  ShapeHandle filter_shape;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &filter_shape));
  TF_RETURN_IF_ERROR(c->WithRank(filter_shape, 4, &filter_shape));

  std::vector<int32> dilations;
  TF_RETURN_IF_ERROR(c->GetAttr("dilations", &dilations));

  if (dilations.size() != 4) {
    return errors::InvalidArgument(
      "SparseConv2D requires the dilations attribute to contain "
      "4 values, but got: ", dilations.size());
  }

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));

  if (strides.size() != 4) {
    return errors::InvalidArgument(
      "SparseConv2D requires the strides attribute to contain "
      "4 values, but got: ", strides.size());
  }

  const int32 stride_rows = GetTensorDim(strides, data_format, 'H');
  const int32 stride_cols = GetTensorDim(strides, data_format, 'W');
  const int32 dilation_rows = GetTensorDim(dilations, data_format, 'H');
  const int32 dilation_cols = GetTensorDim(dilations, data_format, 'W');

  DimensionHandle batch_size_dim = c->Dim(
    input_shape,
    GetTensorDimIndex(data_format, 'N'));
  DimensionHandle input_rows_dim = c->Dim(
    input_shape,
    GetTensorDimIndex(data_format, 'H'));
  DimensionHandle input_cols_dim = c->Dim(
    input_shape,
    GetTensorDimIndex(data_format, 'W'));
  DimensionHandle input_depth_dim = c->Dim(
    input_shape,
    GetTensorDimIndex(data_format, 'C'));

  DimensionHandle depth_multiplier_dim = c->Dim(
    filter_shape,
    GetFilterDimIndex<num_spatial_dims>(filter_format, 'O'));
  DimensionHandle filter_input_depth_dim = c->Dim(
    filter_shape,
    GetFilterDimIndex<num_spatial_dims>(filter_format, 'I'));
  DimensionHandle filter_rows_dim = c->Dim(
    filter_shape,
    GetFilterDimIndex<num_spatial_dims>(filter_format, 'H'));
  DimensionHandle filter_cols_dim = c->Dim(
    filter_shape,
    GetFilterDimIndex<num_spatial_dims>(filter_format, 'W'));

  DimensionHandle output_depth_dim;
  TF_RETURN_IF_ERROR(c->Multiply(input_depth_dim, depth_multiplier_dim, &output_depth_dim));

  {
    DimensionHandle unused;
    TF_RETURN_IF_ERROR(c->Merge(input_depth_dim, filter_input_depth_dim, &unused));
  }

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

  DimensionHandle output_rows_dim, output_cols_dim;
  TF_RETURN_IF_ERROR(
    GetWindowedOutputSizeFromDimsV2(
      c, input_rows_dim, filter_rows_dim, dilation_rows, stride_rows,
      padding, &output_rows_dim));
  TF_RETURN_IF_ERROR(
    GetWindowedOutputSizeFromDimsV2(
      c, input_cols_dim, filter_cols_dim, dilation_cols, stride_cols,
      padding, &output_cols_dim));

  ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(
    shape_inference::MakeShapeFromFormat(
      data_format, batch_size_dim, {output_rows_dim, output_cols_dim},
      output_depth_dim, &output_shape, c));
  c->set_output(0, output_shape);

  return Status::OK();
}

} // namespace


REGISTER_OP("SparseConv2D")
  .Input("input: T")
  .Input("filter_indices: int64")
  .Input("filter_values: T")
  .Input("filter_shape: int64")
  .Output("output: T")
  .Attr("T: {float, double}")
  .Attr("strides: list(int) = [1, 1, 1, 1]")
  .Attr(GetPaddingAttrString())
  .Attr(GetConvnetDataFormatAttrString())
  .Attr("dilations: list(int) = [1, 1, 1, 1]")
  .SetShapeFn(SparseConv2DShape);


REGISTER_OP("SparseDepthwiseConv2dNative")
  .Input("input: T")
  .Input("filter_indices: int64")
  .Input("filter_values: T")
  .Input("filter_shape: int64")
  .Output("output: T")
  .Attr("T: {float, double}")
  .Attr("strides: list(int) = [1, 1, 1, 1]")
  .Attr(GetPaddingAttrString())
  .Attr("dilations: list(int) = [1, 1, 1, 1]")
  .SetShapeFn(SparseDepthwiseConv2DNativeShape);

} // namespace tensorflow
