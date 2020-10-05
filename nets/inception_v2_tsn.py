"""Contains the definition for inception v2 (TSN) classification network."""





import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.platform import tf_logging as logging

trunc_normal = lambda stddev: tf.compat.v1.truncated_normal_initializer(0.0, stddev)
random_normal = lambda stddev: tf.compat.v1.random_normal_initializer(0.0, stddev)

FLAGS = tf.compat.v1.app.flags.FLAGS

tf.compat.v1.app.flags.DEFINE_string(
  'inception_concat_layers', '', 'List of layers to combine with logits.')



def inception_v2_tsn(inputs,
                     num_classes=1000,
                     is_training=True,
                     dropout_keep_prob=0.2,
                     min_depth=16,
                     depth_multiplier=1.0,
                     prediction_fn="softmax",
                     spatial_squeeze=True,
                     reuse=None,
                     conv_only=None,
                     # conv_endpoint='inception_5b',
                     conv_endpoint='inception_5a',  # testing for now
                     scope='InceptionV2_TSN'):
  """Inception v2 model for video classification.

  """
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  return tf.keras.applications.InceptionResNetV2(
      include_top=True, weights=None, input_tensor=inputs, input_shape=inputs.shape,
      pooling=None, classes=num_classes, classifier_activation= prediction_fn
  )
inception_v2_tsn.default_image_size = 224


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.

  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.pack([tf.minimum(shape[1], kernel_size[0]),
                        tf.minimum(shape[2], kernel_size[1])])

  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


def inception_v2_tsn_arg_scope(weight_decay=0.00004):
  """Defines the default InceptionV2 arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.9997,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing update_ops.
      'updates_collections': tf.compat.v1.GraphKeys.UPDATE_OPS,
      # Allow a gamma variable
      'scale': True,
  }

  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=tf.keras.regularizers.l2(0.5 * (weight_decay))):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        activation_fn=None,  # manually added later, as I need to add BN after
                             # the convolution
        biases_initializer=init_ops.constant_initializer(value=0.2),
        normalizer_fn=None) as sc:
      return sc
