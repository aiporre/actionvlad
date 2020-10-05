# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
"""A simple script for write out the first layer with 20 channels"""




import sys
import h5py
import numpy as np

import tensorflow as tf

FLAGS = tf.compat.v1.app.flags.FLAGS
tf.compat.v1.app.flags.DEFINE_string("file_name", "", "Checkpoint filename")
tf.compat.v1.app.flags.DEFINE_string("tensor_name", "", "Name of the tensor to modify")
tf.compat.v1.app.flags.DEFINE_string("output_file_name", "", "Path to write out the" 
                           "first layer weights")


def get_modified_weights(file_name, tensor_name):
  try:
    reader = tf.compat.v1.train.NewCheckpointReader(file_name)
    T = reader.get_tensor(tensor_name)
    return np.repeat(np.mean(T, axis=2, keepdims=True), 20, axis=2)
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")


def main(unused_argv):
  if not FLAGS.file_name:
    print("Usage: inspect_checkpoint --file_name=checkpoint_file_name "
          "[--tensor_name=tensor_to_modify]")
    sys.exit(1)
  else:
    W = get_modified_weights(FLAGS.file_name, FLAGS.tensor_name)
    with h5py.File(FLAGS.output_file_name, 'w') as fout:
      fout.create_dataset('feat', data=W, compression='gzip',
                          compression_opts=9)


if __name__ == "__main__":
  tf.compat.v1.app.run()
