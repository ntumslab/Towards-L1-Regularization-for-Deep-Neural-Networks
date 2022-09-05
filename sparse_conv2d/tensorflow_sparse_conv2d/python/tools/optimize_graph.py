from absl import flags

import tensorflow as tf
from tensorflow.tools import graph_transforms


flags.DEFINE_string(
    'in_graph', '', 'Input graph def')

flags.DEFINE_string(
    'out_graph', '', 'Output graph def')

flags.DEFINE_list(
    'inputs', '', 'Input tensor names')

flags.DEFINE_list(
    'outputs', '', 'Output tensor names')

flags.DEFINE_string(
    'input_size', '1,224,224,3',
    'The input size to use, for example, "1,224,224,3"')

FLAGS = flags.FLAGS


def main(_):
  if not FLAGS.in_graph:
    raise ValueError('Missing --in_graph')
  if not FLAGS.out_graph:
    raise ValueError('Missing --out_graph')
  if not FLAGS.inputs:
    raise ValueError('Missing --inputs')
  if not FLAGS.outputs:
    raise ValueError('Missing --outputs')
  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.gfile.GFile(FLAGS.in_graph, 'rb') as f:
    in_graph_def = tf.GraphDef()
    in_graph_def.ParseFromString(f.read())

  transforms = [
      'strip_unused_nodes(type=float, shape="{size}")'.format(size=FLAGS.input_size),
      'remove_nodes(op=Identity, op=CheckNumerics)',
      'remove_attribute(attribute_name=_class)',
      'fold_constants(ignore_errors=true)',
      'fold_batch_norms',
      'fold_old_batch_norms',
  ]

  out_graph_def = graph_transforms.TransformGraph(
      in_graph_def,
      FLAGS.inputs,
      FLAGS.outputs,
      transforms)

  with tf.gfile.GFile(FLAGS.out_graph, 'wb') as f:
    f.write(out_graph_def.SerializeToString())


if __name__ == '__main__':
  tf.app.run()
