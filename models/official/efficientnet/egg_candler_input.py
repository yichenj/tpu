# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Egg candler input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tensorflow as tf

class EggCandlerInput(object):

  def __init__(self, 
               is_training,
               data_dir,
               train_batch_size,
               eval_batch_size,
               num_cores=8,
               image_size=224):
    self.is_training = is_training
    self.data_dir = data_dir
    self.train_batch_size = train_batch_size
    self.eval_batch_size = eval_batch_size
    self.num_cores = num_cores
    self.image_size = image_size

    batch_size = self.eval_batch_size
    horizontal_flip = False
    dataset_source = os.path.join(self.data_dir, 'Testing')
    if self.is_training:
      batch_size = self.train_batch_size
      horizontal_flip = True
      dataset_source = os.path.join(self.data_dir, 'Training')

    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                             rescale=1./255, horizontal_flip=horizontal_flip)

    self.iterator = image_data_generator.flow_from_directory(
                              dataset_source,
                              target_size=(self.image_size, self.image_size),
                              classes=['normal', 'dead', 'rotten', 'clear'],
                              shuffle=True, batch_size=batch_size,
                              class_mode='sparse')

  def set_shapes(self, batch_size, images, labels):
    """Statically set the batch_size dimension."""
    images.set_shape(images.get_shape().merge_with(
        tf.TensorShape([batch_size, None, None, None])))
    labels.set_shape(labels.get_shape().merge_with(
        tf.TensorShape([batch_size])))
    return images, tf.cast(labels, tf.int32)

  def input_fn(self, params):
    dataset = tf.data.Dataset.from_generator(
                   lambda: self.iterator,
                   (tf.float32, tf.float32))

    batch_size = self.train_batch_size if self.is_training else self.eval_batch_size
    dataset = dataset.map(functools.partial(self.set_shapes, batch_size))
    return dataset
