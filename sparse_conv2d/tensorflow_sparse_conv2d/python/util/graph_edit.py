import numpy as np

import tensorflow as tf
from tensorflow.contrib import graph_editor as ge

from tensorflow_sparse_conv2d.python.ops import sparse_conv2d_ops


def replace_dense_conv2d_in_graph_and_freeze(
    graph,
    is_frozen=True,
    checkpoint_path=None,
    density_threshold=None):
  """Replace Conv2D with SparseConv2D. The weights will be
  converted to constants.

  Args:
    graph: A `tf.Graph`. The graph to edit.
    is_frozen: A bool. Whether the `graph` is frozen.
    checkpoint_path: A `str` points to the checkpoint containing
      the weights. If `is_frozen` is `True`, `checkpoint_path` is ignored.
    density_threshold: A number between 0 and 1 or None.
      The conv ops with weight non-zero ratio > `density_threshold`
      will be kept. If `None`, all conv ops will be replaced.
  """
  if not is_frozen and checkpoint_path is None:
    raise ValueError('Need checkpoint_path while graph is not frozen')

  op_names = []
  filters = []
  with graph.as_default():
    for op in graph.get_operations():
      if op.type == 'Conv2D':
        filter_index = 1
      else:
        continue
      op_names.append(op.name)
      filters.append(op.inputs[filter_index])

    with tf.Session() as sess:
      if not is_frozen:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
      filter_values = sess.run(filters)

    filter_values_map = dict(zip(op_names, filter_values))

    for op in graph.get_operations():
      if op.name not in filter_values_map:
        continue

      filter_value = filter_values_map[op.name]
      scope = op.name.rsplit('/', 1)[0]

      if op.type == 'Conv2D':
        op_input = op.inputs[0]
        op_attrs = _get_conv2d_op_attrs(op)
        op_attrs['name'] = scope + '/SparseConv2D'
        sparse_op_fn = sparse_conv2d_ops.sparse_conv2d
      else:
        continue

      replace_dense_op_and_freeze(
          op, op_input, filter_value,
          sparse_op_fn, op_attrs, scope,
          density_threshold=density_threshold)


def replace_dense_op_and_freeze(
    op,
    op_input,
    filters,
    sparse_op_fn,
    op_attrs,
    scope,
    density_threshold=None):
  # sort the indices in OIHW order
  filters_oihw = np.transpose(filters, [3, 2, 0, 1])
  nnzs_oihw = np.nonzero(filters_oihw)
  nnzs = (nnzs_oihw[2], nnzs_oihw[3], nnzs_oihw[1], nnzs_oihw[0])
  density = nnzs[0].size / filters.size
  if density_threshold is not None and density > density_threshold:
    return
  indices = np.stack(nnzs).T.astype(np.int64)
  values = filters[nnzs]
  dense_shape = np.array(filters.shape).astype(np.int64)

  with tf.name_scope(scope + '/sparse_weights'):
    sp_filter = tf.SparseTensor(
        indices,
        values,
        dense_shape)

  output = sparse_op_fn(
      op_input,
      sp_filter,
      **op_attrs)

  ge.swap_outputs(ge.sgv(op), ge.sgv(output.op))


def _get_conv2d_op_attrs(op):
  strides = op.get_attr('strides')
  padding = op.get_attr('padding').decode()
  dilations = op.get_attr('dilations')
  data_format = op.get_attr('data_format').decode()
  op_attrs = dict(
      strides=strides,
      padding=padding,
      dilations=dilations,
      data_format=data_format)
  return op_attrs
