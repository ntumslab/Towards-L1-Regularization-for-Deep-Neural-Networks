import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


_sparse_conv2d_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_sparse_conv2d_ops.so'))

def sparse_conv2d(input,
                  sp_filter,
                  strides=None,
                  padding='SAME',
                  data_format='NHWC',
                  dilations=None,
                  name=None):
  """Computes a 2-D convolution given 4-D dense `input`
  and sparse `sp_filter` tensors.

  Args:
    input: A `Tensor`. Must be one of the following types:
      `float32`, `float64`.
      A 4-D tensor. The dimension order is interpreted according to the value
      of `data_format`, see below for details.
    sp_filter: A `SparseTensor`. Must have the same type as `input`.
      A 4-D sparse tensor of dense shape
      `[filter_height, filter_width, in_channels, out_channels]`
    strides: Am optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The stride of the sliding window for each
      dimension of `input`. The dimension order is determined by the value of
      `data_format`, see below for details.
    padding: A `string` from: `"SAME", "VALID"`. Defaults to `"SAME"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`.
      Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  if not isinstance(sp_filter, tf.SparseTensor):
    raise ValueError('sp_filter must be a SparseTensor')

  return _sparse_conv2d_ops.sparse_conv2d(
      input=input,
      filter_indices=sp_filter.indices,
      filter_values=sp_filter.values,
      filter_shape=sp_filter.dense_shape,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilations=dilations,
      name=name)


def sparse_depthwise_conv2d_native(input,
                                   sp_filter,
                                   strides=None,
                                   padding='SAME',
                                   data_format='NHWC',
                                   dilations=None,
                                   name=None):
  """Computes a depthwise 2-D convolution given 4-D dense `input`
  and sparse `sp_filter` tensors.

  Args:
    input: A Tensor. Must be one of the following types: `float32`, `float64`.
    sp_filter: A `SparseTensor`. Must have the same type as `input`.
      A 4-D sparse tensor of dense shape
      `[filter_height, filter_width, in_channels, channel_multiplier]`
    strides: A list of `int`s. 1-D of length 4. The stride of the sliding
      window for each dimension of `input`. Defaults to `[1, 1, 1, 1]`.
    padding: A `str` from: `"SAME"`, `"VALID"`. The type of padding algorithm to use.
      Defaults to `"SAME"`.
    data_format: An optional `str` from: `"NHWC"`, `"NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the default
      format `"NHWC"`, the data is stored in the order of:
      `[batch, height, width, channels]`. Alternatively, the format could be
      `"NCHW"`, the data storage order of: `[batch, channels, height, width]`.
      Current only support `"NHWC"`.
    dilations: An optional list of `int`s. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4. The dilation factor for each dimension of `input`.
      If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value
      of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for this operation (optional).
  Returns:
    A 4-D `Tensor` with shape according to `data_format`.  E.g., for
    "NHWC" format, shape is
    `[batch, out_height, out_width, in_channels * channel_multiplier].`
  """
  if data_format is None:
    data_format = 'NHWC'
  if data_format != 'NHWC':
    raise ValueError('Currently only support data_format="NHWC"')

  if not isinstance(sp_filter, tf.SparseTensor):
    raise ValueError('sp_filter must be a SparseTensor')

  if dilations is None:
    dilations = [1, 1, 1, 1]

  return _sparse_conv2d_ops.sparse_depthwise_conv2d_native(
      input=input,
      filter_indices=sp_filter.indices,
      filter_values=sp_filter.values,
      filter_shape=sp_filter.dense_shape,
      strides=strides,
      padding=padding,
      dilations=dilations,
      name=name)
