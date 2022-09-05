from absl import flags

import tensorflow as tf
from tensorflow.python.framework import graph_util

from tensorflow_sparse_conv2d.python.util import graph_edit


flags.DEFINE_string(
    'in_graph', None, 'Input graph def')
flags.mark_flag_as_required('in_graph')

flags.DEFINE_string(
    'out_graph', None, 'Output graph def')
flags.mark_flag_as_required('out_graph')

flags.DEFINE_list(
    'output_node_names', None, 'The name of the output nodes, comma separated.')
flags.mark_flag_as_required('output_node_names')

flags.DEFINE_float(
    'density_threshold', None,
    'The threshold of density of weights to replace')

FLAGS = flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.gfile.GFile(FLAGS.in_graph, 'rb') as f:
    in_graph_def = tf.GraphDef()
    in_graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    tf.import_graph_def(in_graph_def, name='')
    graph_edit.replace_dense_conv2d_in_graph_and_freeze(
        graph,
        is_frozen=True,
        density_threshold=FLAGS.density_threshold)
    out_graph_def = graph_util.extract_sub_graph(
        graph.as_graph_def(),
        FLAGS.output_node_names)

  with tf.gfile.GFile(FLAGS.out_graph, 'wb') as f:
    f.write(out_graph_def.SerializeToString())


if __name__ == '__main__':
  tf.app.run()
