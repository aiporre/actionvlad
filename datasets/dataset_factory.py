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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from datasets import cifar10
# from datasets import flowers
# from datasets import imagenet
# from datasets import mnist
# from datasets import ucf101
# from datasets import charades
# from datasets import places365
# from datasets import hmdb51
import tensorflow as tf
import tensorflow_datasets as tfds
# datasets_map = {
#     'cifar10': cifar,
#     'flowers': flowers,
#     'imagenet': imagenet,
#     'mnist': mnist,
#     'ucf101': ucf101,
#     'charades': charades,
#     'places365': places365,
#     'hmdb51': hmdb51,
# }
datasets_available = [
    'cifar10',
    'flowers',
    'imagenet',
    'mnist',
    'ucf101',
    'charades',
    'places365',
    'hmdb51'
]



def get_dataset(name, split_name, dataset_dir, dataset_list_dir='',
                file_pattern=None, reader=None, modality='rgb', num_samples=1,
                split_id=1):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    dataset_list_dir: The directory where train/test splits are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.
    modality: In case of video datasets, you could read RGB or Flow frames
    num_samples: In case of videos, number of frames per video

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  if name not in datasets_available:
    raise ValueError('Name of dataset unknown %s' % name)
  print('--------------->', dataset_dir)
  config = tfds.download.DownloadConfig(verify_ssl=False)

  ds = tfds.load(name, split=split_name, shuffle_files=True, data_dir=dataset_dir, download_and_prepare_kwargs={"download_config" : config})

  return ds.shuffle(1024).batch(1).take(num_samples).prefetch(tf.data.experimental.AUTOTUNE)
  # return datasets_map[name].get_split(
  #     split_name,
  #     dataset_dir,
  #     dataset_list_dir,
  #     file_pattern,
  #     reader,
  #     modality,
  #     num_samples,
  #     split_id=split_id)
