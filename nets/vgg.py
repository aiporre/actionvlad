# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""




import tensorflow as tf

layers = tf.compat.v1.layers


# def vgg_arg_scope(weight_decay=0.0005):
#   """Defines the VGG arg scope.
#
#   Args:
#     weight_decay: The l2 regularization coefficient.
#
#   Returns:
#     An arg_scope.
#   """
#   with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                       activation_fn=tf.nn.relu,
#                       weights_regularizer=tf.keras.regularizers.l2(0.5 * (weight_decay)),
#                       biases_initializer=tf.compat.v1.zeros_initializer):
#     with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
#       return arg_sc


def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a'):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.compat.v1.variable_scope(scope, 'vgg_a', [inputs]) as sc:
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    net = layers.conv2d(inputs, 1, 64, [3, 3], name='conv1a')
    net = layers.conv2d(net, 1, 64, [3, 3], name='conv1b')
    net = layers.conv2d(net, 1, 64, [3, 3], name='conv1c')
    net = layers.max_pooling2d(net, [2, 2], name='pool1')

    net = layers.conv2d(net, 1, 128, [3, 3], name='conv2a')
    net = layers.conv2d(net, 1, 128, [3, 3], name='conv2b')
    net = layers.conv2d(net, 1, 128, [3, 3], name='conv2c')
    net = layers.max_pooling2d(net, [2, 2], name='pool2')

    net = layers.conv2d(net, 1, 256, [3, 3], name='conv3a')
    net = layers.conv2d(net, 1, 256, [3, 3], name='conv3b')
    net = layers.conv2d(net, 1, 256, [3, 3], name='conv3c')
    net = layers.max_pooling2d(net, [2, 2], name='pool3')

    net = layers.conv2d(net, 1, 512, [3, 3], name='conv4a')
    net = layers.conv2d(net, 1, 512, [3, 3], name='conv4b')
    net = layers.conv2d(net, 1, 512, [3, 3], name='conv4c')
    net = layers.max_pooling2d(net, [2, 2], name='pool4')

    net = layers.conv2d(net, 1, 512, [3, 3], name='conv5a')
    net = layers.conv2d(net, 1, 512, [3, 3], name='conv5b')
    net = layers.conv2d(net, 1, 512, [3, 3], name='conv5c')
    net = layers.max_pooling2d(net, [2, 2], name='pool5')

    # Use conv2d instead of fully_connected layers.
    net = layers.conv2d(net, 4096, [7, 7], padding='VALID', name='fc6')
    net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
                       name='dropout6')
    net = layers.conv2d(net, 4096, [1, 1], name='fc7')
    #avg_fc7 = tf.reduce_mean(net, axis=0) # Added by brussell
    net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
                       name='dropout7')
    net = layers.conv2d(net, num_classes, [1, 1],
                      activation_fn=None,
                      normalizer_fn=None,
                      name='fc8')
    # Convert end_points_collection into a end_point dict.
    #end_points.update({'avg_fc7': avg_fc7}) # Added by brussell
    if spatial_squeeze:
      net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
    return net
vgg_a.default_image_size = 224


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           conv_only=False,
           conv_endpoint='conv5'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  return tf.keras.applications.VGG16(
    include_top=True, weights=None, input_tensor=inputs, input_shape=inputs.shape,
    pooling=None, classes=num_classes, classifier_activation='softmax'
  )
vgg_16.default_image_size = 224


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19'):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  return tf.keras.applications.VGG19(
    include_top=True, weights=None, input_tensor=inputs, input_shape=inputs.shape,
    pooling=None, classes=num_classes, classifier_activation='softmax'
  )
vgg_19.default_image_size = 224

# Alias
vgg_d = vgg_16
vgg_e = vgg_19
