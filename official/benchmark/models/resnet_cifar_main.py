# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a ResNet model on the Cifar-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
from official.benchmark.models import resnet_cifar_model
from official.modeling.hyperparams import params_dict
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.vision.image_classification import callbacks as custom_callbacks
from official.vision.image_classification.resnet import cifar_preprocessing
from official.vision.image_classification.resnet import common
from official.vision.image_classification.pruning import cprune_from_config
from official.vision.image_classification.pruning.resnet_cifar import resnet_cifar_pruning_config
from tensorflow_model_optimization.python.core.sparsity.keras import cpruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import cprune


LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (0.1, 91), (0.01, 136), (0.001, 182)
]
pp = pprint.PrettyPrinter()


def learning_rate_schedule(current_epoch,
                           current_batch,
                           batches_per_epoch,
                           batch_size):
  """Handles linear scaling rule and LR decay.

  Scale learning rate at epoch boundaries provided in LR_SCHEDULE by the
  provided scaling factor.

  Args:
    current_epoch: integer, current epoch indexed from 0.
    current_batch: integer, current batch in the current epoch, indexed from 0.
    batches_per_epoch: integer, number of steps in an epoch.
    batch_size: integer, total batch sized.

  Returns:
    Adjusted learning rate.
  """
  del current_batch, batches_per_epoch  # not used
  initial_learning_rate = common.BASE_LEARNING_RATE * batch_size / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if current_epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  return learning_rate


def resume_from_checkpoint(model: tf.keras.Model,
                           model_dir: str,
                           train_steps: int) -> int:
  """Resumes from the latest checkpoint, if possible.

  Loads the model weights and optimizer settings from a checkpoint.
  This function should be used in case of preemption recovery.

  Args:
    model: The model whose weights should be restored.
    model_dir: The directory where model weights were saved.
    train_steps: The number of steps to train.

  Returns:
    The epoch of the latest checkpoint, or 0 if not restoring.

  """
  logging.info('Load from checkpoint is enabled.')
  latest_checkpoint = tf.train.latest_checkpoint(model_dir)
  logging.info('latest_checkpoint: %s', latest_checkpoint)
  if not latest_checkpoint:
    logging.info('No checkpoint detected.')
    return 0

  logging.info('Checkpoint file %s found and restoring from '
               'checkpoint', latest_checkpoint)
  model.load_weights(latest_checkpoint)
  initial_epoch = model.optimizer.iterations // train_steps
  logging.info('Completed loading from checkpoint.')
  logging.info('Resuming from epoch %d', initial_epoch)
  return int(initial_epoch)


class LearningRateBatchScheduler(tf.keras.callbacks.Callback):
  """Callback to update learning rate on every batch (not epoch boundaries).

  N.B. Only support Keras optimizers, not TF optimizers.

  Attributes:
      schedule: a function that takes an epoch index and a batch index as input
          (both integer, indexed from 0) and returns a new learning rate as
          output (float).
  """

  def __init__(self, schedule, batch_size, steps_per_epoch):
    super(LearningRateBatchScheduler, self).__init__()
    self.schedule = schedule
    self.steps_per_epoch = steps_per_epoch
    self.batch_size = batch_size
    self.epochs = -1
    self.prev_lr = -1

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'learning_rate'):
      raise ValueError('Optimizer must have a "learning_rate" attribute.')
    self.epochs += 1

  def on_batch_begin(self, batch, logs=None):
    """Executes before step begins."""
    lr = self.schedule(self.epochs,
                       batch,
                       self.steps_per_epoch,
                       self.batch_size)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function should be float.')
    if lr != self.prev_lr:
      self.model.optimizer.learning_rate = lr  # lr should be a float here
      self.prev_lr = lr
      tf.compat.v1.logging.debug(
          'Epoch %05d Batch %05d: LearningRateBatchScheduler '
          'change learning rate to %s.', self.epochs, batch, lr)


def run(flags_obj):
  """Run ResNet Cifar-10 training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  keras_utils.set_session_config(
      enable_eager=flags_obj.enable_eager,
      enable_xla=flags_obj.enable_xla)

  # Execute flag override logic for better model performance
  if flags_obj.tf_gpu_thread_mode:
    keras_utils.set_gpu_thread_mode_and_count(
        per_gpu_thread_count=flags_obj.per_gpu_thread_count,
        gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
        num_gpus=flags_obj.num_gpus,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads)
  common.set_cudnn_batchnorm_mode()

  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == 'fp16':
    raise ValueError('dtype fp16 is not supported in Keras. Use the default '
                     'value(fp32).')

  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs)

  if strategy:
    # flags_obj.enable_get_next_as_optional controls whether enabling
    # get_next_as_optional behavior in DistributedIterator. If true, last
    # partial batch can be supported.
    strategy.extended.experimental_enable_get_next_as_optional = (
        flags_obj.enable_get_next_as_optional
    )

  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  if flags_obj.use_synthetic_data:
    distribution_utils.set_up_synthetic_data()
    input_fn = common.get_synth_input_fn(
        height=cifar_preprocessing.HEIGHT,
        width=cifar_preprocessing.WIDTH,
        num_channels=cifar_preprocessing.NUM_CHANNELS,
        num_classes=cifar_preprocessing.NUM_CLASSES,
        dtype=flags_core.get_tf_dtype(flags_obj),
        drop_remainder=True)
  else:
    distribution_utils.undo_set_up_synthetic_data()
    input_fn = cifar_preprocessing.input_fn

  train_input_dataset = input_fn(
      is_training=True,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      parse_record_fn=cifar_preprocessing.parse_record,
      datasets_num_private_threads=flags_obj.datasets_num_private_threads,
      dtype=dtype,
      # Setting drop_remainder to avoid the partial batch logic in normalization
      # layer, which triggers tf.where and leads to extra memory copy of input
      # sizes between host and GPU.
      drop_remainder=(not flags_obj.enable_get_next_as_optional))

  eval_input_dataset = None
  if not flags_obj.skip_eval:
    eval_input_dataset = input_fn(
        is_training=False,
        data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size,
        parse_record_fn=cifar_preprocessing.parse_record)

  steps_per_epoch = (
      cifar_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)
  lr_schedule = 0.1
  if flags_obj.use_tensor_lr:
    initial_learning_rate = common.BASE_LEARNING_RATE * flags_obj.batch_size / 128
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=list(p[1] * steps_per_epoch for p in LR_SCHEDULE),
        values=[initial_learning_rate] +
        list(p[0] * initial_learning_rate for p in LR_SCHEDULE))

  with strategy_scope:
    optimizer = common.get_optimizer(lr_schedule)
    model = resnet_cifar_model.resnet56(classes=cifar_preprocessing.NUM_CLASSES)

    if flags_obj.model_weights_path:
      if os.path.isdir(flags_obj.model_weights_path):
        checkpoint = tf.train.latest_checkpoint(flags_obj.model_weights_path)
      else:
        checkpoint = flags_obj.model_weights_path
      logging.info('Load weights from  %s', checkpoint)
      model.load_weights(checkpoint)

    if flags_obj.mode == 'sensitivity_analysis' or flags_obj.pruning_config_file:
      if flags_obj.mode == 'sensitivity_analysis':
        if flags_obj.pruning_config_file:
          raise ValueError

        layer_name = [
            layer.name for layer in model.layers if hasattr(layer, 'kernel')
        ][flags_obj.sensitivity_layer_count]

        pruning_params = cprune_from_config.generate_sensitivity_config(
            model_name='resnet56',
            layer_name=layer_name,
            weight_name='kernel',
            granularity=flags_obj.sensitivity_granularity,
            gamma=flags_obj.sensitivity_gamma,
            respect_submatrix=flags_obj.sensitivity_respect_submatrix,
            two_over_four_chin=flags_obj.sensitivity_two_over_four_chin)
      else:
        pruning_params = resnet_cifar_pruning_config.ResNet56PruningConfig()

        params_dict.override_params_dict(
            pruning_params, flags_obj.pruning_config_file, is_strict=False)
        logging.info('Specified pruning params: %s', pp.pformat(pruning_params.as_dict()))

      _pruning_params = cprune_from_config.predict_sparsity(model, pruning_params)
      logging.info('Understood pruning params: %s', pp.pformat(_pruning_params))

      model = cprune_from_config.cprune_from_config(model, pruning_params)

    else:
      weights_list = model.get_weights()
      model = tf.keras.models.clone_model(model)
      model.set_weights(weights_list)

    models = [model]

    if flags_obj.mode == 'prune_physically':
      smaller_model = cprune_from_config.prune_physically(model)
      models.append(smaller_model)

    for _model in models:
      _model.compile(
          loss='sparse_categorical_crossentropy',
          optimizer=optimizer,
          metrics=(['sparse_categorical_accuracy']
                   if flags_obj.report_accuracy_metrics else None),
          run_eagerly=flags_obj.run_eagerly)

    train_epochs = flags_obj.train_epochs

    initial_epoch = 0
    if flags_obj.resume_checkpoint:
      initial_epoch = resume_from_checkpoint(model=model,
                                             model_dir=flags_obj.model_dir,
                                             train_steps=steps_per_epoch)

  model_pruning_config = None
  if flags_obj.pruning_config_file:
    model_pruning_config = cprune_from_config._expand_model_pruning_config(
        model, pruning_params
    )
  callbacks = common.get_callbacks(steps_per_epoch,
                                   enable_checkpoint_and_export=True,
                                   model_dir=flags_obj.model_dir)
  for i, callback in enumerate(callbacks):
    if isinstance(callback, tf.keras.callbacks.TensorBoard):
      callbacks.pop(i)
      break
  callbacks.append(
    custom_callbacks.CustomTensorBoard(
      log_dir=flags_obj.model_dir,
      track_lr=True,
      model_pruning_config=model_pruning_config,
    )
  )
  if flags_obj.pruning_config_file:
    callbacks.append(cpruning_callbacks.UpdateCPruningStep())

  if not flags_obj.use_tensor_lr:
    lr_callback = LearningRateBatchScheduler(
        schedule=learning_rate_schedule,
        batch_size=flags_obj.batch_size,
        steps_per_epoch=steps_per_epoch)
    callbacks.append(lr_callback)

  # if mutliple epochs, ignore the train_steps flag.
  if train_epochs <= 1 and flags_obj.train_steps:
    steps_per_epoch = min(flags_obj.train_steps, steps_per_epoch)
    train_epochs = 1

  num_eval_steps = (cifar_preprocessing.NUM_IMAGES['validation'] //
                    flags_obj.batch_size)

  validation_data = eval_input_dataset
  if flags_obj.skip_eval:
    if flags_obj.set_learning_phase_to_train:
      # TODO(haoyuzhang): Understand slowdown of setting learning phase when
      # not using distribution strategy.
      tf.keras.backend.set_learning_phase(1)
    num_eval_steps = None
    validation_data = None

  if not strategy and flags_obj.explicit_gpu_placement:
    # TODO(b/135607227): Add device scope automatically in Keras training loop
    # when not using distribition strategy.
    no_dist_strat_device = tf.device('/device:GPU:0')
    no_dist_strat_device.__enter__()

  if flags_obj.mode == 'train_and_eval':
    history = model.fit(train_input_dataset,
                        epochs=train_epochs,
                        steps_per_epoch=steps_per_epoch,
                        initial_epoch=initial_epoch,
                        callbacks=callbacks,
                        validation_steps=num_eval_steps,
                        validation_data=validation_data,
                        validation_freq=flags_obj.epochs_between_evals,
                        verbose=2)
  elif flags_obj.mode == 'eval':
    callbacks = None
    history = cprune.apply_cpruning_masks(model)
  elif flags_obj.mode in ('sensitivity_analysis', 'prune_physically'):
    callbacks, history = None, None
  else:
    raise ValueError('{} is not a valid mode.'.format(flags.FLAGS.mode))

  export_path = os.path.join(flags_obj.model_dir, 'saved_model')
  model.save(export_path, include_optimizer=False)

  if flags.FLAGS.pruning_config_file:
    _pruning_params = cprune_from_config.predict_sparsity(model, pruning_params)
    logging.info('Pruning result: %s', pp.pformat(_pruning_params))

  eval_output = None
  if flags_obj.mode == 'sensitivity_analysis':
    file_writer = tf.summary.create_file_writer(flags_obj.model_dir + '/metrics')
    file_writer.set_as_default()
    for sparsity_x_16 in range(16):
      cprune.apply_cpruning_masks(model, step=sparsity_x_16)
      _eval_output = model.evaluate(
          eval_input_dataset, steps=num_eval_steps, verbose=2)
      _stats = common.build_stats(history, _eval_output, callbacks)
      prefix = 'pruning_sensitivity/' + layer_name + '/' + 'kernel' + '/'
      for key, value in _stats.items():
        tf.summary.scalar(prefix + key, data=value, step=sparsity_x_16)
      _pruning_params = cprune_from_config.predict_sparsity(model, pruning_params)
      sparsity = _pruning_params['pruning'][0]['pruning'][0]['current_sparsity']
      tf.summary.scalar(prefix + 'sparsity', data=sparsity, step=sparsity_x_16)
  elif flags_obj.mode == 'prune_physically':
    logging.info('Number of filters before and after physical pruning:')
    for layer, new_layer in zip(model.layers, smaller_model.layers):
      if type(layer) is tf.keras.layers.Conv2D:
        logging.info('    {}, {}, {}'.format(layer.name, layer.filters, new_layer.filters))
      if type(layer) is tf.keras.layers.Dense:
        logging.info('    {}, {}, {}'.format(layer.name, layer.units, new_layer.units))
    for i, _model in enumerate(models):
      situation = 'before' if i == 0 else 'after'
      logging.info('Model summary {} physical pruning:'.format(situation))
      _model.summary(print_fn=logging.info)
      _eval_output = _model.evaluate(
          eval_input_dataset, steps=num_eval_steps, verbose=2)
      _stats = common.build_stats(history, _eval_output, callbacks)
      logging.info('Evaluation {} physical pruning: {}'.format(situation, _stats))
    export_path = os.path.join(flags_obj.model_dir, 'saved_model_small')
    smaller_model.save(export_path, include_optimizer=False)
  elif not flags_obj.skip_eval:
    eval_output = model.evaluate(eval_input_dataset,
                                 steps=num_eval_steps,
                                 verbose=2)

  if not strategy and flags_obj.explicit_gpu_placement:
    no_dist_strat_device.__exit__()

  stats = common.build_stats(history, eval_output, callbacks)
  return stats


def define_cifar_flags():
  common.define_keras_flags(dynamic_loss_scale=False)

  flags_core.set_defaults(data_dir='/tmp/cifar10_data/cifar-10-batches-bin',
                          model_dir='/tmp/cifar10_model',
                          epochs_between_evals=10,
                          batch_size=128)

  flags.DEFINE_string('pruning_config_file', None,
                      'Path to a yaml file of model pruning configuration.')


  flags.DEFINE_string(
      'mode',
      default=None,
      help='Mode to run: `train_and_eval`, `eval`, `sensitivity_analysis`, or '
           '`prune_physically`.')
  flags.DEFINE_integer(
      'sensitivity_layer_count',
      default=0,
      help='The ordinal number representing a layer whose pruning sensitivity '
           'is to be analyzed. 0 for `"conv1"` (the first layer), 58 for '
           '`"fc10"` (the last layer) etc. Valid only if '
           '`mode=sensitivity_analysis`.')
  flags.DEFINE_string(
      'sensitivity_granularity',
      default='BlockSparsity',
      help='The granularity for analyzing pruning sensitivity. Valid only if '
           '`mode=sensitivity_analysis`.')
  flags.DEFINE_integer(
      'sensitivity_gamma',
      default=2,
      help='The gamma parameter for ArayaMag or QuasiCyclic granularity.'
           ' for analyzing pruning sensitivity. Valid only if '
           '`mode=sensitivity_analysis`.')
  flags.DEFINE_bool(
      'sensitivity_respect_submatrix',
      default=False,
      help='Whether or not to apply pruning masks submatrix-wise. Valid only '
           'for ArayaMag, QuasiCyclic, and TwoOutOfFour granularity.')
  flags.DEFINE_bool(
      'sensitivity_two_over_four_chin',
      default=False,
      help='Whether or not to realize two-out-of-four sparsity pattern along '
           'input channels. Defaults to `False`, in which case the sparsity '
           'pattern is achieved along the output channels.')
  flags.DEFINE_bool('resume_checkpoint', None,
                    'Whether or not to enable load checkpoint loading. Defaults '
                    'to None.')
  flags.DEFINE_string('model_weights_path', None,
                      'The path to the directory where model checkpoints are '
                      'saved.')


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    return run(flags.FLAGS)


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  define_cifar_flags()
  app.run(main)
