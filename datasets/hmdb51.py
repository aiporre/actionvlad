# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
"""Provides data for the HMDB51 dataset.
"""





from datasets.video_data_utils import gen_dataset


def get_split(split_name, dataset_dir, dataset_list_dir='', file_pattern=None,
              reader=None, modality='rgb', num_samples=1,
              split_id=1):
  
  _NUM_CLASSES = 51
  _LIST_FN = lambda split, id: \
      '%s/%s_split%d.txt' % (dataset_list_dir, split, id)

  return gen_dataset(split_name, dataset_dir, file_pattern,
                     reader, modality, num_samples, split_id,
                     _NUM_CLASSES, _LIST_FN)
