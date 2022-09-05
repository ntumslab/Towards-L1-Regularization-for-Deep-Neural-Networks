import numpy as np

import tensorflow as tf

from tensorflow_sparse_conv2d.python.ops import sparse_conv2d_ops as ops


internal_ops = ops._sparse_conv2d_ops


def _to_float32(arr):
  return np.array(arr, dtype=np.float32)


def _to_int64(arr):
  return np.array(arr, dtype=np.int64)


def _get_simple_input_and_filter():
  inp = _to_float32([[[[1]]]])
  sp_filter = tf.SparseTensor(
      _to_int64([[0, 0, 0, 0]]),
      _to_float32([1]),
      _to_float32([1, 1, 1, 1]))
  return inp, sp_filter


class SparseConv2DTest(tf.test.TestCase):
  def test_input_rank(self):
    sp_filter = tf.SparseTensor(
        _to_int64([[0, 0, 0, 0]]),
        _to_float32([1]),
        _to_float32([1, 1, 1, 1]))

    with self.assertRaisesRegex(ValueError, 'rank 4 but is rank 3'):
      ops.sparse_conv2d(
          input=_to_float32([[[1]]]),
          sp_filter=sp_filter)

    with self.assertRaisesRegex(ValueError, 'rank 4 but is rank 5'):
      ops.sparse_conv2d(
          input=_to_float32([[[[[1]]]]]),
          sp_filter=sp_filter)

  def test_filter_shape(self):
    inp = _to_float32([[[[1]]]])

    with self.assertRaisesRegex(ValueError, '4 but is 3'):
      internal_ops.sparse_conv2d(
          input=inp,
          filter_indices=_to_int64([[0, 0, 0]]),
          filter_values=_to_float32([1]),
          filter_shape=_to_int64([1, 1, 1, 1]),
          padding='SAME')

    with self.assertRaisesRegex(ValueError, '4 but is rank 3'):
      internal_ops.sparse_conv2d(
          input=inp,
          filter_indices=_to_int64([[0, 0, 0, 0]]),
          filter_values=_to_float32([1]),
          filter_shape=_to_int64([1, 1, 1]),
          padding='SAME')

    with self.assertRaisesRegex(ValueError, '1 but is rank 2'):
      internal_ops.sparse_conv2d(
          input=inp,
          filter_indices=_to_int64([[0, 0, 0, 0]]),
          filter_values=_to_float32([[]]),
          filter_shape=_to_int64([1, 1, 1, 1]),
          padding='SAME')

    with self.assertRaisesRegex(ValueError, ' but are 1 and 2'):
      internal_ops.sparse_conv2d(
          input=inp,
          filter_indices=_to_int64([[0, 0, 0, 0]]),
          filter_values=_to_float32([1]),
          filter_shape=_to_int64([1, 1, 2, 1]),
          padding='SAME')

  def test_filter_shape_with_session(self):
    inp = _to_float32([[[[1]]]])

    with self.test_session():
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError, 'got: 1 and 2'):
        internal_ops.sparse_conv2d(
            input=inp,
            filter_indices=_to_int64([[0, 0, 0, 0]]),
            filter_values=_to_float32([1, 1]),
            filter_shape=_to_int64([1, 1, 1, 1]),
            padding='SAME').eval()

  def test_filter_not_sparse(self):
    inp = _to_float32([[[[1]]]])

    with self.assertRaisesRegex(ValueError, 'SparseTensor'):
      ops.sparse_conv2d(
          input=inp,
          sp_filter=tf.constant(inp))

  def test_invalid_dilations(self):
    inp, sp_filter = _get_simple_input_and_filter()

    with self.assertRaisesRegex(ValueError, 'but got: 3'):
      ops.sparse_conv2d(
          input=inp,
          sp_filter=sp_filter,
          dilations=[1, 1, 1])

    with self.assertRaisesRegex(ValueError, 'but got 0'):
      ops.sparse_conv2d(
          input=inp,
          sp_filter=sp_filter,
          dilations=[1, 1, 0, 1])

    with self.test_session():
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError,
          'not yet support dilations in the batch'):
        ops.sparse_conv2d(
            input=inp,
            sp_filter=sp_filter,
            dilations=[1, 1, 1, 0]).eval()

      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError,
          'not yet support dilations in the height'):
        ops.sparse_conv2d(
            input=inp,
            sp_filter=sp_filter,
            dilations=[1, 1, 2, 1]).eval()

  def test_invalid_strides(self):
    inp, sp_filter = _get_simple_input_and_filter()

    with self.assertRaisesRegex(ValueError, 'but got: 3'):
      ops.sparse_conv2d(
          input=inp,
          sp_filter=sp_filter,
          strides=[1, 1, 1])

    with self.assertRaisesRegex(ValueError, 'but got 0'):
      ops.sparse_conv2d(
          input=inp,
          sp_filter=sp_filter,
          strides=[1, 1, 0, 1])

    with self.test_session():
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError,
          'not yet support strides in the batch'):
        ops.sparse_conv2d(
            input=inp,
            sp_filter=sp_filter,
            strides=[1, 1, 1, 0]).eval()

  def test_output_shape(self):
    inp = np.zeros([2, 9, 8, 5], dtype=np.float32)
    sp_filter = tf.SparseTensor(
        _to_int64([[0, 0, 0, 0]]),
        _to_float32([1]),
        _to_int64([3, 3, 5, 7]))

    self.assertAllEqual(
        ops.sparse_conv2d(
          inp,
          sp_filter,
          padding='SAME').shape.as_list(),
        [2, 9, 8, 7])

    self.assertAllEqual(
        ops.sparse_conv2d(
          inp,
          sp_filter,
          padding='VALID').shape.as_list(),
        [2, 7, 6, 7])

    self.assertAllEqual(
        ops.sparse_conv2d(
          inp,
          sp_filter,
          strides=[1, 2, 2, 1],
          padding='SAME').shape.as_list(),
        [2, 5, 4, 7])

    with self.test_session():
      self.assertAllEqual(
          ops.sparse_conv2d(
            inp,
            sp_filter,
            padding='SAME').eval().shape,
          [2, 9, 8, 7])

      self.assertAllEqual(
          ops.sparse_conv2d(
            inp,
            sp_filter,
            padding='VALID').eval().shape,
          [2, 7, 6, 7])

      self.assertAllEqual(
          ops.sparse_conv2d(
            inp,
            sp_filter,
            strides=[1, 2, 2, 1],
            padding='SAME').eval().shape,
          [2, 5, 4, 7])

  def test_simple(self):
    inp, sp_filter = _get_simple_input_and_filter()

    with self.test_session():
      self.assertAllClose(
          ops.sparse_conv2d(
            inp,
            sp_filter).eval(),
          [[[[1]]]])

  def test_3x3(self):
    inp = np.arange(9).reshape(1, 3, 3, 1).astype(np.float32)
    # dense = [
    #   [1, 0, -1],
    #   [0, 2, 0],
    #   [-3, 0, 0]
    # ]
    sp_filter = tf.SparseTensor(
        _to_int64([
          [0, 0, 0, 0],
          [2, 0, 0, 0],
          [1, 1, 0, 0],
          [0, 2, 0, 0]]),
        _to_float32([1, -3, 2, -1]),
        _to_int64([3, 3, 1, 1]))

    with self.test_session():
      expected = _to_float32([
        [0, -7, -8],
        [5, -12, -10],
        [8, 12, 20]
      ]).reshape(1, 3, 3, 1)
      self.assertAllClose(
          ops.sparse_conv2d(
            inp,
            sp_filter).eval(),
          expected)

  def test_packet(self):
    inp = np.ones([1, 19, 19, 1], dtype=np.float32)
    sp_filter = tf.SparseTensor(
        _to_int64([
          [0, 0, 0, 0],
          [2, 0, 0, 0],
          [1, 1, 0, 0],
          [0, 2, 0, 0]]),
        _to_float32([1, -3, 2, -1]),
        _to_int64([3, 3, 1, 1]))

    with self.test_session():
      expected = -1 * np.ones([1, 17, 17, 1], dtype=np.float32)
      self.assertAllClose(
          ops.sparse_conv2d(
            inp,
            sp_filter,
            padding='VALID').eval(),
          expected)


class SparseDepthwiseConv2DTest(tf.test.TestCase):
  def test_input_rank(self):
    sp_filter = tf.SparseTensor(
        _to_int64([[0, 0, 0, 0]]),
        _to_float32([1]),
        _to_float32([1, 1, 1, 1]))

    with self.assertRaisesRegex(ValueError, 'rank 4 but is rank 3'):
      ops.sparse_depthwise_conv2d_native(
          input=_to_float32([[[1]]]),
          sp_filter=sp_filter)

    with self.assertRaisesRegex(ValueError, 'rank 4 but is rank 5'):
      ops.sparse_depthwise_conv2d_native(
          input=_to_float32([[[[[1]]]]]),
          sp_filter=sp_filter)

  def test_filter_shape(self):
    inp = _to_float32([[[[1]]]])

    with self.assertRaisesRegex(ValueError, '4 but is 3'):
      internal_ops.sparse_depthwise_conv2d_native(
          input=inp,
          filter_indices=_to_int64([[0, 0, 0]]),
          filter_values=_to_float32([1]),
          filter_shape=_to_int64([1, 1, 1, 1]),
          padding='SAME')

    with self.assertRaisesRegex(ValueError, '4 but is rank 3'):
      internal_ops.sparse_depthwise_conv2d_native(
          input=inp,
          filter_indices=_to_int64([[0, 0, 0, 0]]),
          filter_values=_to_float32([1]),
          filter_shape=_to_int64([1, 1, 1]),
          padding='SAME')

    with self.assertRaisesRegex(ValueError, '1 but is rank 2'):
      internal_ops.sparse_depthwise_conv2d_native(
          input=inp,
          filter_indices=_to_int64([[0, 0, 0, 0]]),
          filter_values=_to_float32([[]]),
          filter_shape=_to_int64([1, 1, 1, 1]),
          padding='SAME')

    with self.assertRaisesRegex(ValueError, ' but are 1 and 2'):
      internal_ops.sparse_depthwise_conv2d_native(
          input=inp,
          filter_indices=_to_int64([[0, 0, 0, 0]]),
          filter_values=_to_float32([1]),
          filter_shape=_to_int64([1, 1, 2, 1]),
          padding='SAME')

  def test_filter_shape_with_session(self):
    inp = _to_float32([[[[1]]]])

    with self.test_session():
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError, 'got: 1 and 2'):
        internal_ops.sparse_depthwise_conv2d_native(
            input=inp,
            filter_indices=_to_int64([[0, 0, 0, 0]]),
            filter_values=_to_float32([1, 1]),
            filter_shape=_to_int64([1, 1, 1, 1]),
            padding='SAME').eval()

  def test_filter_not_sparse(self):
    inp = _to_float32([[[[1]]]])

    with self.assertRaisesRegex(ValueError, 'SparseTensor'):
      ops.sparse_depthwise_conv2d_native(
          input=inp,
          sp_filter=tf.constant(inp))

  def test_data_format(self):
    with self.assertRaisesRegex(ValueError, 'NHWC'):
      ops.sparse_depthwise_conv2d_native(
          input=None,
          sp_filter=None,
          data_format='NCHW')

  def test_invalid_dilations(self):
    inp, sp_filter = _get_simple_input_and_filter()

    with self.assertRaisesRegex(ValueError, 'but got: 3'):
      ops.sparse_depthwise_conv2d_native(
          input=inp,
          sp_filter=sp_filter,
          dilations=[1, 1, 1])

    with self.assertRaisesRegex(ValueError, 'but got 0'):
      ops.sparse_depthwise_conv2d_native(
          input=inp,
          sp_filter=sp_filter,
          dilations=[1, 1, 0, 1])

  def test_invalid_strides(self):
    inp, sp_filter = _get_simple_input_and_filter()

    with self.assertRaisesRegex(ValueError, 'but got: 3'):
      ops.sparse_depthwise_conv2d_native(
          input=inp,
          sp_filter=sp_filter,
          strides=[1, 1, 1])

    with self.assertRaisesRegex(ValueError, 'but got 0'):
      ops.sparse_depthwise_conv2d_native(
          input=inp,
          sp_filter=sp_filter,
          strides=[1, 1, 0, 1])

    with self.test_session():
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError,
          'not yet support strides in the batch'):
        ops.sparse_depthwise_conv2d_native(
            input=inp,
            sp_filter=sp_filter,
            strides=[1, 1, 1, 0]).eval()

  def test_output_shape(self):
    inp = np.zeros([2, 9, 8, 5], dtype=np.float32)
    sp_filter = tf.SparseTensor(
        _to_int64([[0, 0, 0, 0]]),
        _to_float32([1]),
        _to_int64([3, 3, 5, 2]))

    self.assertAllEqual(
        ops.sparse_depthwise_conv2d_native(
          inp,
          sp_filter,
          padding='SAME').shape.as_list(),
        [2, 9, 8, 10])

    self.assertAllEqual(
        ops.sparse_depthwise_conv2d_native(
          inp,
          sp_filter,
          padding='VALID').shape.as_list(),
        [2, 7, 6, 10])

    self.assertAllEqual(
        ops.sparse_depthwise_conv2d_native(
          inp,
          sp_filter,
          strides=[1, 2, 2, 1],
          padding='SAME').shape.as_list(),
        [2, 5, 4, 10])

    with self.test_session():
      self.assertAllEqual(
          ops.sparse_depthwise_conv2d_native(
            inp,
            sp_filter,
            padding='SAME').eval().shape,
          [2, 9, 8, 10])

      self.assertAllEqual(
          ops.sparse_depthwise_conv2d_native(
            inp,
            sp_filter,
            padding='VALID').eval().shape,
          [2, 7, 6, 10])

      self.assertAllEqual(
          ops.sparse_depthwise_conv2d_native(
            inp,
            sp_filter,
            strides=[1, 2, 2, 1],
            padding='SAME').eval().shape,
          [2, 5, 4, 10])

  def test_simple(self):
    inp, sp_filter = _get_simple_input_and_filter()

    with self.test_session():
      self.assertAllClose(
          ops.sparse_depthwise_conv2d_native(
            inp,
            sp_filter).eval(),
          [[[[1]]]])

  def test_3x3(self):
    inp = np.arange(9).reshape(1, 3, 3, 1).astype(np.float32)
    # dense = [
    #   [1, 0, -1],
    #   [0, 2, 0],
    #   [-3, 0, 0]
    # ]
    sp_filter = tf.SparseTensor(
        _to_int64([
          [0, 0, 0, 0],
          [0, 2, 0, 0],
          [1, 1, 0, 0],
          [2, 0, 0, 0]]),
        _to_float32([1, -1, 2, -3]),
        _to_int64([3, 3, 1, 1]))

    with self.test_session():
      expected = _to_float32([
        [0, -7, -8],
        [5, -12, -10],
        [8, 12, 20]
      ]).reshape(1, 3, 3, 1)
      self.assertAllClose(
          ops.sparse_depthwise_conv2d_native(
            inp,
            sp_filter).eval(),
          expected)

  def test_3x3_multipler_2(self):
    inp = np.arange(9).reshape(1, 3, 3, 1).astype(np.float32)
    # dense[:, :, 0, 0] = [
    #   [1, 0, -1],
    #   [0, 2, 0],
    #   [-3, 0, 0]
    # ]
    # dense[:, :, 0, 1] = [
    #   [0, -1, 0],
    #   [-2, 0, 1],
    #   [0, 3, 0]
    # ]
    sp_filter = tf.SparseTensor(
        _to_int64([
          [0, 0, 0, 0],
          [0, 1, 0, 1],
          [0, 2, 0, 0],
          [1, 0, 0, 1],
          [1, 1, 0, 0],
          [1, 2, 0, 1],
          [2, 0, 0, 0],
          [2, 1, 0, 1]]),
        _to_float32([1, -1, -1, -2, 2, 1, -3, 3]),
        _to_int64([3, 3, 1, 2]))

    with self.test_session():
      expected_0 = _to_float32([
        [0, -7, -8],
        [5, -12, -10],
        [8, 12, 20]
      ]).reshape(1, 3, 3, 1)
      expected_1 = _to_float32([
        [10, 14, 13],
        [22, 19, 14],
        [4, -8, -19]
      ]).reshape(1, 3, 3, 1)
      expected = np.concatenate([expected_0, expected_1], axis=3)
      self.assertAllClose(
          ops.sparse_depthwise_conv2d_native(
            inp,
            sp_filter).eval(),
          expected)
