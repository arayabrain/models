# Lint as: python3
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
"""Runs an Image Classification model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pprint
from typing import Any, Tuple, Text, Optional, Mapping

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from official.modeling import performance
from official.modeling.hyperparams import params_dict
from official.utils import hyperparams_flags
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.vision.image_classification import callbacks as custom_callbacks
from official.vision.image_classification import dataset_factory
from official.vision.image_classification import optimizer_factory
from official.vision.image_classification.configs import base_configs
from official.vision.image_classification.configs import configs
from official.vision.image_classification.efficientnet import efficientnet_model
from official.vision.image_classification.resnet import common
from official.vision.image_classification.resnet import resnet_model
from official.vision.image_classification.mobilenet_v1 import mobilenet_v1_model
from official.vision.image_classification.pruning import cprune_from_config
from official.vision.image_classification.pruning import pruning_base_configs
#from official.vision.image_classification.pruning.efficientnet import efficientnet_pruning_config
from official.vision.image_classification.pruning.mobilenet_v1 import mobilenet_v1_pruning_config
from official.vision.image_classification.pruning.resnet_imagenet import resnet_imagenet_pruning_config
from tensorflow_model_optimization.python.core.sparsity.keras import cpruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import cprune
from tensorflow_model_optimization.python.core.sparsity.keras import cprune_registry


pp = pprint.PrettyPrinter()


def get_models() -> Mapping[str, tf.keras.Model]:
  """Returns the mapping from model type name to Keras model."""
  return {
      'efficientnet': efficientnet_model.EfficientNet.from_name,
      'resnet': resnet_model.resnet50,
      'mobilenet_v1': mobilenet_v1_model.mobilenet_v1
  }


def get_pruning() -> Mapping[str, pruning_base_configs.ModelPruningConfig]:
  """Returns the mapping from model type name to model pruning config."""
  return {
      #'efficientnet': efficientnet_pruning_config.EfficientNetPruningConfig(),
      'resnet': resnet_imagenet_pruning_config.ResNet50PruningConfig(),
      'mobilenet_v1': mobilenet_v1_pruning_config.MobileNetV1PruningConfig(),
  }

def get_dtype_map() -> Mapping[str, tf.dtypes.DType]:
  """Returns the mapping from dtype string representations to TF dtypes."""
  return {
      'float32': tf.float32,
      'bfloat16': tf.bfloat16,
      'float16': tf.float16,
      'fp32': tf.float32,
      'bf16': tf.bfloat16,
  }


def _get_metrics(one_hot: bool) -> Mapping[Text, Any]:
  """Get a dict of available metrics to track."""
  if one_hot:
    return {
        # (name, metric_fn)
        'acc': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        'accuracy': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        'top_1': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        'top_5': tf.keras.metrics.TopKCategoricalAccuracy(
            k=5,
            name='top_5_accuracy'),
    }
  else:
    return {
        # (name, metric_fn)
        'acc': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        'top_1': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        'top_5': tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5,
            name='top_5_accuracy'),
    }


def get_image_size_from_model(
    params: base_configs.ExperimentConfig) -> Optional[int]:
  """If the given model has a preferred image size, return it."""
  if params.model_name == 'efficientnet':
    efficientnet_name = params.model.model_params.model_name
    if efficientnet_name in efficientnet_model.MODEL_CONFIGS:
      return efficientnet_model.MODEL_CONFIGS[efficientnet_name].resolution
  return None


def _get_dataset_builders(params: base_configs.ExperimentConfig,
                          strategy: tf.distribute.Strategy,
                          one_hot: bool
                         ) -> Tuple[Any, Any]:
  """Create and return train and validation dataset builders."""
  if one_hot:
    logging.warning('label_smoothing > 0, so datasets will be one hot encoded.')
  else:
    logging.warning('label_smoothing not applied, so datasets will not be one '
                    'hot encoded.')

  num_devices = strategy.num_replicas_in_sync if strategy else 1

  image_size = get_image_size_from_model(params)

  dataset_configs = [
      params.train_dataset, params.validation_dataset
  ]
  builders = []

  for config in dataset_configs:
    if config is not None and config.has_data:
      builder = dataset_factory.DatasetBuilder(
          config,
          image_size=image_size or config.image_size,
          num_devices=num_devices,
          one_hot=one_hot)
    else:
      builder = None
    builders.append(builder)

  return builders


def get_loss_scale(params: base_configs.ExperimentConfig,
                   fp16_default: float = 128.) -> float:
  """Returns the loss scale for initializations."""
  loss_scale = params.runtime.loss_scale
  if loss_scale == 'dynamic':
    return loss_scale
  elif loss_scale is not None:
    return float(loss_scale)
  elif params.train_dataset.dtype == 'float32':
    return 1.
  else:
    assert params.train_dataset.dtype == 'float16'
    return fp16_default


def _get_params_from_flags(flags_obj: flags.FlagValues):
  """Get ParamsDict from flags."""
  model = flags_obj.model_type.lower()
  dataset = flags_obj.dataset.lower()
  params = configs.get_config(model=model, dataset=dataset)

  flags_overrides = {
      'model_dir': flags_obj.model_dir,
      'mode': flags_obj.mode,
      'model': {
          'name': model,
      },
      'runtime': {
          'run_eagerly': flags_obj.run_eagerly,
          'tpu': flags_obj.tpu,
      },
      'train_dataset': {
          'data_dir': flags_obj.data_dir,
      },
      'validation_dataset': {
          'data_dir': flags_obj.data_dir,
      },
      'train': {
          'time_history': {
              'log_steps': flags_obj.log_steps,
          },
      },
  }

  overriding_configs = (flags_obj.config_file,
                        flags_obj.params_override,
                        flags_overrides)

  logging.info('Base params: %s', pp.pformat(params.as_dict()))

  for param in overriding_configs:
    logging.info('Overriding params: %s', param)
    # Set is_strict to false because we can have dynamic dict parameters.
    params = params_dict.override_params_dict(params, param, is_strict=False)

  params.validate()
  params.lock()

  logging.info('Final model parameters: %s', pp.pformat(params.as_dict()))
  return params


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


def initialize(params: base_configs.ExperimentConfig,
               dataset_builder: dataset_factory.DatasetBuilder):
  """Initializes backend related initializations."""
  keras_utils.set_session_config(
      enable_eager=params.runtime.run_eagerly,
      enable_xla=params.runtime.enable_xla)
  if params.runtime.gpu_threads_enabled:
    keras_utils.set_gpu_thread_mode_and_count(
        per_gpu_thread_count=params.runtime.per_gpu_thread_count,
        gpu_thread_mode=params.runtime.gpu_thread_mode,
        num_gpus=params.runtime.num_gpus,
        datasets_num_private_threads=params.runtime.dataset_num_private_threads)

  performance.set_mixed_precision_policy(dataset_builder.dtype)
  if tf.config.list_physical_devices('GPU'):
    data_format = 'channels_first'
  else:
    data_format = 'channels_last'
  tf.keras.backend.set_image_data_format(data_format)
  distribution_utils.configure_cluster(
      params.runtime.worker_hosts,
      params.runtime.task_index)
  if params.runtime.run_eagerly:
    # Enable eager execution to allow step-by-step debugging
    tf.config.experimental_run_functions_eagerly(True)


def define_classifier_flags():
  """Defines common flags for image classification."""
  hyperparams_flags.initialize_common_flags()
  flags.DEFINE_string(
      'data_dir',
      default=None,
      help='The location of the input data.')
  flags.DEFINE_string(
      'mode',
      default=None,
      help='Mode to run: `train`, `eval`, `train_and_eval`, `export`, '
           '`sensitivity_analysis`, or `prune_physically`.')
  flags.DEFINE_bool(
      'run_eagerly',
      default=None,
      help='Use eager execution and disable autograph for debugging.')
  flags.DEFINE_string(
      'model_type',
      default=None,
      help='The type of the model, e.g. EfficientNet, etc.')
  flags.DEFINE_string(
      'dataset',
      default=None,
      help='The name of the dataset, e.g. ImageNet, etc.')
  flags.DEFINE_integer(
      'log_steps',
      default=100,
      help='The interval of steps between logging of batch level stats.')
  flags.DEFINE_integer(
      'verbose',
      default=1,
      help='0, 1, or 2. Verbosity mode.'
      '0 = silent, 1 = progress bar, 2 = one line per epoch.'
      'Note that the progress bar is not particularly useful when logged to a file, '
      'so verbose=2 is recommended when not running interactively '
      '(eg, in a production environment).')
  flags.DEFINE_string('pruning_config_file', None,
                      'Path to a yaml file of model pruning configuration.')
  flags.DEFINE_integer(
      'sensitivity_layer_count',
      default=0,
      help='The ordinal number representing a layer whose pruning sensitivity '
           'is to be analyzed. 0 for the first layer, 27 (MobileNet V1) and 53 '
           '(ResNet-50) for the last layer, etc. Valid only if '
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


def serialize_config(params: base_configs.ExperimentConfig,
                     model_dir: str):
  """Serializes and saves the experiment config."""
  params_save_path = os.path.join(model_dir, 'params.yaml')
  logging.info('Saving experiment configuration to %s', params_save_path)
  tf.io.gfile.makedirs(model_dir)
  params_dict.save_params_dict_to_yaml(params, params_save_path)


def train_and_eval(
    params: base_configs.ExperimentConfig,
    strategy_override: tf.distribute.Strategy) -> Mapping[str, Any]:
  """Runs the train and eval path using compile/fit."""
  logging.info('Running train and eval.')

  # Note: for TPUs, strategy and scope should be created before the dataset
  strategy = strategy_override or distribution_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)

  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  logging.info('Detected %d devices.',
               strategy.num_replicas_in_sync if strategy else 1)

  label_smoothing = params.model.loss.label_smoothing
  one_hot = label_smoothing and label_smoothing > 0

  builders = _get_dataset_builders(params, strategy, one_hot)
  datasets = [builder.build() if builder else None for builder in builders]

  # Unpack datasets and builders based on train/val/test splits
  train_builder, validation_builder = builders  # pylint: disable=unbalanced-tuple-unpacking
  train_dataset, validation_dataset = datasets

  train_epochs = params.train.epochs
  train_steps = params.train.steps or train_builder.num_steps
  validation_steps = params.evaluation.steps or validation_builder.num_steps

  initialize(params, train_builder)

  logging.info('Global batch size: %d', train_builder.global_batch_size)

  with strategy_scope:
    model_params = params.model.model_params.as_dict()
    model = get_models()[params.model.name](**model_params)

    if params.model.model_weights_path:
      if os.path.isdir(params.model.model_weights_path):
        checkpoint = tf.train.latest_checkpoint(params.model.model_weights_path)
      else:
        checkpoint = params.model.model_weights_path
      logging.info('Load weights from  %s', checkpoint)
      model.load_weights(checkpoint)

    if flags.FLAGS.mode == 'sensitivity_analysis' or flags.FLAGS.pruning_config_file:
      if flags.FLAGS.mode == 'sensitivity_analysis':
        if flags.FLAGS.pruning_config_file:
          raise ValueError

        layer = [
            layer for layer in model.layers
            if hasattr(layer, 'kernel') or hasattr(layer, 'depthwise_kernel')
        ][flags.FLAGS.sensitivity_layer_count]
        layer_name = layer.name
        weight_name = 'kernel' if hasattr(layer, 'kernel') else 'depthwise_kernel'

        pruning_params = cprune_from_config.generate_sensitivity_config(
            model_name=model.name,
            layer_name=layer_name,
            weight_name=weight_name,
            granularity=flags.FLAGS.sensitivity_granularity,
            gamma=flags.FLAGS.sensitivity_gamma,
            respect_submatrix=flags.FLAGS.sensitivity_respect_submatrix,
            two_over_four_chin=flags.FLAGS.sensitivity_two_over_four_chin)
      else:
        pruning_params = get_pruning()[params.model.name]

        params_dict.override_params_dict(
            pruning_params, flags.FLAGS.pruning_config_file, is_strict=False)
        logging.info('Specified pruning params: %s', pp.pformat(pruning_params.as_dict()))

      _pruning_params = cprune_from_config.predict_sparsity(model, pruning_params)
      logging.info('Understood pruning params: %s', pp.pformat(_pruning_params))

      model = cprune_from_config.cprune_from_config(model, pruning_params)

    else:
      weights_list = model.get_weights()
      model = tf.keras.models.clone_model(model)
      model.set_weights(weights_list)

    models = [model]

    if flags.FLAGS.mode == 'prune_physically':
      smaller_model = cprune_from_config.prune_physically(model)
      models.append(smaller_model)

    for _model in models:
      learning_rate = optimizer_factory.build_learning_rate(
          params=params.model.learning_rate,
          batch_size=train_builder.global_batch_size,
          train_steps=train_steps)
      optimizer = optimizer_factory.build_optimizer(
          optimizer_name=params.model.optimizer.name,
          base_learning_rate=learning_rate,
          params=params.model.optimizer.as_dict())

      metrics_map = _get_metrics(one_hot)
      metrics = [metrics_map[metric] for metric in params.train.metrics]

      if one_hot:
        loss_obj = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=params.model.loss.label_smoothing)
      else:
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

      _model.compile(optimizer=optimizer,
                     loss=loss_obj,
                     metrics=metrics)

    initial_epoch = 0
    if params.train.resume_checkpoint:
      initial_epoch = resume_from_checkpoint(model=model,
                                             model_dir=params.model_dir,
                                             train_steps=train_steps)

  callbacks = None
  if params.mode == 'train_and_eval':
    serialize_config(params=params, model_dir=params.model_dir)
    # TODO(dankondratyuk): callbacks significantly slow down training
    model_pruning_config = None
    if flags.FLAGS.pruning_config_file:
      model_pruning_config = cprune_from_config._expand_model_pruning_config(
        model, pruning_params
      )
    callbacks = custom_callbacks.get_callbacks(
        model_checkpoint=params.train.callbacks.enable_checkpoint_and_export,
        include_tensorboard=params.train.callbacks.enable_tensorboard,
        time_history=params.train.callbacks.enable_time_history,
        track_lr=params.train.tensorboard.track_lr,
        model_pruning_config=model_pruning_config,
        write_model_weights=params.train.tensorboard.write_model_weights,
        batch_size=train_builder.global_batch_size,
        log_steps=params.train.time_history.log_steps,
        model_dir=params.model_dir)
    if flags.FLAGS.pruning_config_file:
      callbacks += [
        cpruning_callbacks.UpdateCPruningStep(),
        # cpruning_callbacks.CPruningSummaries(log_dir=params.model_dir),
      ]

  if params.evaluation.skip_eval:
    validation_kwargs = {}
  else:
    validation_kwargs = {
        'validation_data': validation_dataset,
        'validation_steps': validation_steps,
        'validation_freq': params.evaluation.epochs_between_evals,
    }

  history = None
  if params.mode == 'train_and_eval':
    history = model.fit(
        train_dataset,
        epochs=train_epochs,
        steps_per_epoch=train_steps,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=flags.FLAGS.verbose,
        **validation_kwargs)
  elif params.mode == 'eval':
    cprune.apply_cpruning_masks(model)

  if flags.FLAGS.pruning_config_file:
    _pruning_params = cprune_from_config.predict_sparsity(model, pruning_params)
    logging.info('Pruning result: %s', pp.pformat(_pruning_params))

  validation_output = None
  if params.evaluation.eval_data == 'train':
    eval_dataset = train_dataset
    eval_steps = train_steps
  elif params.evaluation.eval_data == 'validation':
    eval_dataset = validation_dataset
    eval_steps = validation_steps

  if params.mode == 'sensitivity_analysis':
    file_writer = tf.summary.create_file_writer(flags.FLAGS.model_dir + '/metrics')
    file_writer.set_as_default()
    cprune_registry.ConstraintRegistry.add_weight_constraint_pair(
        'depthwise_kernel', 'depthwise_constraint')

    for sparsity_x_16 in range(16):
      cprune.apply_cpruning_masks(model, step=sparsity_x_16)
      _validation_output = model.evaluate(
          eval_dataset, steps=eval_steps, verbose=2, return_dict=True)
      _validation_output = [_validation_output['loss'],
                            _validation_output['accuracy'],
                            _validation_output['top_5_accuracy']]
      _stats = common.build_stats(history, _validation_output, callbacks)
      prefix = 'pruning_sensitivity/' + layer_name + '/' + weight_name + '/'
      for key, value in _stats.items():
        tf.summary.scalar(prefix + key, data=value, step=sparsity_x_16)
      _pruning_params = cprune_from_config.predict_sparsity(model, pruning_params)
      sparsity = _pruning_params['pruning'][0]['pruning'][0]['current_sparsity']
      tf.summary.scalar(prefix + 'sparsity', data=sparsity, step=sparsity_x_16)

  elif flags.FLAGS.mode == 'prune_physically':
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
      _validation_output = _model.evaluate(
          eval_dataset, steps=eval_steps, verbose=2, return_dict=True)
      _validation_output = [_validation_output['loss'],
                            _validation_output['accuracy'],
                            _validation_output['top_5_accuracy']]
      _stats = common.build_stats(history, _validation_output, callbacks)
      logging.info('Evaluation {} physical pruning: {}'.format(situation, _stats))

      postfix = '' if i == 0 else '_small'
      export_path = os.path.join(flags.FLAGS.model_dir, 'saved_model' + postfix)
      _model.save(export_path, include_optimizer=False)

  elif not params.evaluation.skip_eval or params.mode == 'eval':
    logging.info('Evaluate %s data', params.evaluation.eval_data)
    validation_output = model.evaluate(
        eval_dataset, steps=eval_steps, verbose=2, return_dict=True)

  if validation_output:
    validation_output = [validation_output['loss'],
                         validation_output['accuracy'],
                         validation_output['top_5_accuracy']]

  # TODO(dankondratyuk): eval and save final test accuracy
  stats = common.build_stats(history, validation_output, callbacks)
  return stats


def export(params: base_configs.ExperimentConfig):
  """Runs the model export functionality."""
  if flags.FLAGS.pruning_config_file:
    raise ValueError
  logging.info('Exporting model.')
  model_params = params.model.model_params.as_dict()
  model = get_models()[params.model.name](**model_params)
  checkpoint = params.export.checkpoint
  if checkpoint is None:
    logging.info('No export checkpoint was provided. Using the latest '
                 'checkpoint from model_dir.')
    checkpoint = tf.train.latest_checkpoint(params.model_dir)

  model.load_weights(checkpoint)
  model.save(params.export.destination)


def run(flags_obj: flags.FlagValues,
        strategy_override: tf.distribute.Strategy = None) -> Mapping[str, Any]:
  """Runs Image Classification model using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.
    strategy_override: A `tf.distribute.Strategy` object to use for model.

  Returns:
    Dictionary of training/eval stats
  """
  params = _get_params_from_flags(flags_obj)
  if params.mode in ['train_and_eval', 'eval', 'sensitivity_analysis', 'prune_physically']:
    return train_and_eval(params, strategy_override)
  elif params.mode == 'export_only':
    export(params)
  else:
    raise ValueError('{} is not a valid mode.'.format(params.mode))


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    stats = run(flags.FLAGS)
  if stats:
    logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_classifier_flags()
  flags.mark_flag_as_required('data_dir')
  flags.mark_flag_as_required('mode')
  flags.mark_flag_as_required('model_type')
  flags.mark_flag_as_required('dataset')

  app.run(main)
