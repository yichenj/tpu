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
"""Train a EfficientNets for egg candler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

import efficientnet_builder
import egg_candler_input
import utils
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator

FLAGS = flags.FLAGS

# Model specific flags
flags.DEFINE_string(
    'data_dir', default='/mnt/projects/Data',
    help=('The directory where input data is stored. Please see'
          ' the README.md for the expected data format.'))

flags.DEFINE_string(
    'model_dir', default='model',
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

flags.DEFINE_string(
    'model_name',
    default='efficientnet-b0',
    help=('The model name among existing configurations.'))

flags.DEFINE_string(
    'mode', default='train_and_eval',
    help='One of {"train_and_eval", "train", "eval"}.')

flags.DEFINE_integer(
    'train_steps', default=4200,
    help=('The number of steps to use for training. Default is 21000 steps'
          ' which is approximately 200 epochs at batch size 2. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))

flags.DEFINE_integer(
    'input_image_size', default=None,
    help=('Input image size: it depends on specific model name.'))

flags.DEFINE_integer(
    'train_batch_size', default=2, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=10, help='Batch size for evaluation.')

flags.DEFINE_integer(
    'num_train_images', default=210, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=110, help='Size of evaluation data set.')

flags.DEFINE_integer(
    'steps_per_eval', default=105,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'save_checkpoints_steps', default=105,
    help=('Save checkpoints every this many steps.'))

flags.DEFINE_integer('log_step_count_steps', 105, 'The number of steps at '
                     'which the global step information is logged.')

flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help='Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_integer(
    'num_label_classes', default=4, help='Number of classes, at least 2')

flags.DEFINE_float(
    'batch_norm_momentum',
    default=None,
    help=('Batch normalization layer momentum of moving average to override.'))
flags.DEFINE_float(
    'batch_norm_epsilon',
    default=None,
    help=('Batch normalization layer epsilon to override..'))

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))

flags.DEFINE_float(
    'base_learning_rate',
    default=0.016,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('Momentum parameter used in the MomentumOptimizer.'))

flags.DEFINE_float(
    'moving_average_decay', default=0.9999,
    help=('Moving average decay rate.'))

flags.DEFINE_float(
    'weight_decay', default=1e-5,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing', default=0.1,
    help=('Label smoothing parameter used in the softmax_cross_entropy'))

flags.DEFINE_float(
    'dropout_rate', default=None,
    help=('Dropout rate for the final output layer.'))

flags.DEFINE_float(
    'drop_connect_rate', default=None,
    help=('Drop connect rate for the network.'))

flags.DEFINE_float(
    'depth_coefficient', default=None,
    help=('Depth coefficient for scaling number of layers.'))

flags.DEFINE_float(
    'width_coefficient', default=None,
    help=('Width coefficient for scaling channel size.'))

flags.DEFINE_integer(
    'keep_checkpoint_max', default=10,
    help=('The maximum number of recent checkpoint files to keep.'))


def model_fn(features, labels, mode, params):
  """The model_fn to be used with TPUEstimator.

  Args:
    features: `Tensor` of batched images.
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  """
  if isinstance(features, dict):
    features = features['feature']

  stats_shape = [1, 1, 3]

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  has_moving_average_decay = (FLAGS.moving_average_decay > 0)
  # This is essential, if using a keras-derived model.
  tf.keras.backend.set_learning_phase(is_training)
  tf.logging.info('Using open-source implementation.')
  override_params = {}
  if FLAGS.batch_norm_momentum is not None:
    override_params['batch_norm_momentum'] = FLAGS.batch_norm_momentum
  if FLAGS.batch_norm_epsilon is not None:
    override_params['batch_norm_epsilon'] = FLAGS.batch_norm_epsilon
  if FLAGS.dropout_rate is not None:
    override_params['dropout_rate'] = FLAGS.dropout_rate
  if FLAGS.drop_connect_rate is not None:
    override_params['drop_connect_rate'] = FLAGS.drop_connect_rate
  if FLAGS.num_label_classes:
    override_params['num_classes'] = FLAGS.num_label_classes
  if FLAGS.depth_coefficient:
    override_params['depth_coefficient'] = FLAGS.depth_coefficient
  if FLAGS.width_coefficient:
    override_params['width_coefficient'] = FLAGS.width_coefficient

  def normalize_features(features, mean_rgb, stddev_rgb):
    """Normalize the image given the means and stddevs."""
    features -= tf.constant(mean_rgb, shape=stats_shape, dtype=features.dtype)
    features /= tf.constant(stddev_rgb, shape=stats_shape, dtype=features.dtype)
    return features

  def build_model():
    """Build model using the model_name given through the command line."""
    model_builder = None
    if FLAGS.model_name.startswith('efficientnet'):
      model_builder = efficientnet_builder
    else:
      raise ValueError(
          'Model must be either efficientnet-b*')

    normalized_features = normalize_features(features, model_builder.MEAN_RGB,
                                             model_builder.STDDEV_RGB)
    logits, _ = model_builder.build_model(
        normalized_features,
        model_name=FLAGS.model_name,
        training=is_training,
        override_params=override_params,
        model_dir=FLAGS.model_dir)
    return logits

  logits = build_model()

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  one_hot_labels = tf.one_hot(labels, FLAGS.num_label_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=one_hot_labels,
      label_smoothing=FLAGS.label_smoothing)

  # Add weight decay to the loss for non-batch-normalization variables.
  loss = cross_entropy + FLAGS.weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])

  global_step = tf.train.get_global_step()
  if has_moving_average_decay:
    ema = tf.train.ExponentialMovingAverage(
        decay=FLAGS.moving_average_decay, num_updates=global_step)
    ema_vars = utils.get_ema_vars()

  train_op = None
  restore_vars_dict = None
  training_hooks = []
  if is_training:
    # Compute the current epoch and associated learning rate from global_step.
    current_epoch = (
        tf.cast(global_step, tf.float32) / params['steps_per_epoch'])

    scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)
    learning_rate = utils.build_learning_rate(scaled_lr, global_step,
                                              params['steps_per_epoch'])
    optimizer = utils.build_optimizer(learning_rate, optimizer_name='adam')

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

    if has_moving_average_decay:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(ema_vars)

    predictions = tf.argmax(logits, axis=1)
    top1_accuray = tf.metrics.accuracy(labels, predictions)
    logging_hook = tf.train.LoggingTensorHook({
            "loss": loss, 
            "accuracy": top1_accuray[1], 
            "step": global_step}, 
        every_n_iter=1)
    training_hooks.append(logging_hook)

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    predictions = tf.argmax(logits, axis=1)
    top1_accuray = tf.metrics.accuracy(labels, predictions)
    eval_metrics = {
      'val_accuracy': top1_accuray
    }

  num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
  tf.logging.info('number of trainable parameters: {}'.format(num_params))

  scaffold = None
  if has_moving_average_decay and not is_training:
    # Only apply scaffold for eval jobs.
    saver = tf.train.Saver(restore_vars_dict)
    scaffold = tf.train.Scaffold(saver=saver)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      training_hooks=training_hooks,
      eval_metric_ops=eval_metrics,
      scaffold=scaffold)


def export(est, export_dir, input_image_size=None):
  """Export graph to SavedModel and TensorFlow Lite.

  Args:
    est: estimator instance.
    export_dir: string, exporting directory.
    input_image_size: int, input image size.

  Raises:
    ValueError: the export directory path is not specified.
  """
  if not export_dir:
    raise ValueError('The export directory path is not specified.')

  if not input_image_size:
    input_image_size = FLAGS.input_image_size

  tf.logging.info('Starting to export model.')
  est.export_saved_model(
      export_dir_base=export_dir,
      serving_input_receiver_fn=
          tf.estimator.export.build_raw_serving_input_receiver_fn)


def main(unused_argv):
  input_image_size = FLAGS.input_image_size
  if not input_image_size:
    if FLAGS.model_name.startswith('efficientnet'):
      _, _, input_image_size, _ = efficientnet_builder.efficientnet_params(
          FLAGS.model_name)
    else:
      raise ValueError('input_image_size must be set except for EfficientNet')

  config = tf.estimator.RunConfig(
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      log_step_count_steps=FLAGS.log_step_count_steps,
      session_config=tf.ConfigProto(
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True))))
  # Initializes model parameters.
  params = dict(
      steps_per_epoch=FLAGS.num_train_images / FLAGS.train_batch_size)
  est = tf.estimator.Estimator(
      model_fn=model_fn,
      config=config,
      params=params)

  def build_input(is_training):
    """Input for training and eval."""
    tf.logging.info('Using dataset: %s', FLAGS.data_dir)
    return egg_candler_input.EggCandlerInput(
              is_training=is_training,
              data_dir=FLAGS.data_dir,
              train_batch_size=FLAGS.train_batch_size,
              eval_batch_size=FLAGS.eval_batch_size,
              image_size=input_image_size)

  image_train = build_input(is_training=True)
  image_eval = build_input(is_training=False)

  if FLAGS.mode == 'eval':
    eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size
    # Run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):
      tf.logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = est.evaluate(
            input_fn=image_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Eval results: %s. Elapsed seconds: %d',
                        eval_results, elapsed_time)
        utils.archive_ckpt(eval_results, eval_results['val_accuracy'], ckpt)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= FLAGS.train_steps:
          tf.logging.info(
              'Evaluation finished after training step %d', current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint', ckpt)
  else:   # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
    current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long

    tf.logging.info(
        'Training for %d steps (%.2f epochs in total). Current'
        ' step %d.', FLAGS.train_steps,
        FLAGS.train_steps / params['steps_per_epoch'], current_step)

    start_timestamp = time.time()  # This time will include compilation time

    if FLAGS.mode == 'train':
      est.train(
          input_fn=image_train.input_fn,
          max_steps=FLAGS.train_steps,
          hooks=[])
    else:
      assert FLAGS.mode == 'train_and_eval'
      while current_step < FLAGS.train_steps:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                              FLAGS.train_steps)
        est.train(input_fn=image_train.input_fn, max_steps=next_checkpoint, hooks=[])
        current_step = next_checkpoint

        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        next_checkpoint, int(time.time() - start_timestamp))

        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be excluded modulo the batch size. As long as the batch size is
        # consistent, the evaluated images are also consistent.
        tf.logging.info('Starting to evaluate.')
        eval_results = est.evaluate(
            input_fn=image_eval.input_fn,
            steps=FLAGS.num_eval_images // FLAGS.eval_batch_size)
        tf.logging.info('Eval results at step %d: %s',
                        next_checkpoint, eval_results)
        ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
        utils.archive_ckpt(eval_results, eval_results['val_accuracy'], ckpt)

      elapsed_time = int(time.time() - start_timestamp)
      tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                      FLAGS.train_steps, elapsed_time)
  if FLAGS.export_dir:
    export(est, FLAGS.export_dir, input_image_size)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
