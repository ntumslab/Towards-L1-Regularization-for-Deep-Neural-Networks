import tensorflow as tf
import numpy as np
from tensorflow_sparse_conv2d.python.ops import sparse_conv2d_ops

frozen = './models/ssh_vgg_16_dense.pb'

image_size = [1, 800, 1200, 3]

with tf.gfile.GFile(frozen, 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())

g = tf.Graph()
with g.as_default():
  tf.import_graph_def(graph_def, name='')

inp = g.get_tensor_by_name('image_tensor:0')
output = [
  g.get_tensor_by_name('detection_boxes:0'),
  g.get_tensor_by_name('detection_classes:0'),
  g.get_tensor_by_name('detection_scores:0'),
  g.get_tensor_by_name('num_detections:0'),
]

config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

stat_summarizer = tf.contrib.stat_summarizer.StatSummarizer()

with tf.Session(graph=g, config=config) as sess:
  for i in range(10):
    x = np.random.randint(0, 256, size=image_size, dtype=np.uint8)
    sess.run(output, feed_dict={inp: x}, options=options, run_metadata=run_metadata)
    stat_summarizer.ProcessStepStatsStr(run_metadata.step_stats.SerializeToString())

stat_summarizer.PrintStepStats()
