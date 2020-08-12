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
"""Runs a simple model on the MNIST dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from official.modeling.hyperparams import params_dict
from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
from official.vision.image_classification.resnet import common
from official.vision.image_classification import callbacks as custom_callbacks
from official.vision.image_classification.pruning import cprune_from_config
from official.vision.image_classification.pruning.mnist import mnist_pruning_config
from tensorflow_model_optimization.python.core.sparsity.keras import cpruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import cprune

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()


def build_model():
  """Constructs the ML model used to predict handwritten digits."""

  image = tf.keras.layers.Input(shape=(28, 28, 1))

  y = tf.keras.layers.Conv2D(filters=32,
                             kernel_size=5,
                             padding='same',
                             use_bias=False,
                             activation='relu')(image)
  y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='same')(y)
  y = tf.keras.layers.Conv2D(filters=32,
                             kernel_size=5,
                             padding='same',
                             use_bias=False,
                             activation='relu')(y)
  y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='same')(y)
  y = tf.keras.layers.Flatten()(y)
  y = tf.keras.layers.Dense(1024, activation='relu', use_bias=False)(y)
  y = tf.keras.layers.Dropout(0.4)(y)

  probs = tf.keras.layers.Dense(10, activation='softmax')(y)

  model = tf.keras.models.Model(image, probs, name='mnist')

  return model


@tfds.decode.make_decoder(output_dtype=tf.float32)
def decode_image(example, feature):
  """Convert image to float32 and normalize from [0, 255] to [0.0, 1.0]."""
  return tf.cast(feature.decode_example(example), dtype=tf.float32) / 255


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


def run(flags_obj, datasets_override=None, strategy_override=None):
  """Run MNIST model training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.
    datasets_override: A pair of `tf.data.Dataset` objects to train the model,
                       representing the train and test sets.
    strategy_override: A `tf.distribute.Strategy` object to use for model.

  Returns:
    Dictionary of training and eval stats.
  """
  strategy = strategy_override or distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      tpu_address=flags_obj.tpu)

  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  mnist = tfds.builder('mnist', data_dir=flags_obj.data_dir)
  if flags_obj.download:
    mnist.download_and_prepare()

  mnist_train, mnist_test = datasets_override or mnist.as_dataset(
      split=['train', 'test'],
      decoders={'image': decode_image()},  # pylint: disable=no-value-for-parameter
      as_supervised=True)
  train_input_dataset = mnist_train.cache().repeat().shuffle(
      buffer_size=50000).batch(flags_obj.batch_size)
  eval_input_dataset = mnist_test.cache().repeat().batch(flags_obj.batch_size)

  with strategy_scope:
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.05, decay_steps=100000, decay_rate=0.96)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    model = build_model()

    if flags_obj.mode == 'sensitivity_analysis' or flags_obj.pruning_config_file:
      if flags_obj.mode == 'sensitivity_analysis':
        if flags_obj.pruning_config_file:
          raise ValueError

        layer_name = [
            layer.name for layer in model.layers if hasattr(layer, 'kernel')
        ][flags_obj.sensitivity_layer_count]

        pruning_params = cprune_from_config.generate_sensitivity_config(
            model_name='mnist',
            layer_name=layer_name,
            weight_name='kernel',
            granularity=flags_obj.sensitivity_granularity,
            gamma=flags_obj.sensitivity_gamma)
      else:
        pruning_params = mnist_pruning_config.MNISTPruningConfig()
        params_dict.override_params_dict(
            pruning_params, flags_obj.pruning_config_file, is_strict=False)
        logging.info('Specified pruning params: %s', pp.pformat(pruning_params.as_dict()))

      _pruning_params = cprune_from_config.predict_sparsity(model, pruning_params)
      logging.info('Understood pruning params: %s', pp.pformat(_pruning_params))

      model = cprune_from_config.cprune_from_config(model, pruning_params)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])

  num_train_examples = mnist.info.splits['train'].num_examples
  train_steps = num_train_examples // flags_obj.batch_size
  train_epochs = flags_obj.train_epochs

  initial_epoch = 0
  if flags_obj.resume_checkpoint:
    initial_epoch = resume_from_checkpoint(model=model,
                                           model_dir=flags_obj.model_dir,
                                           train_steps=train_steps)

  ckpt_full_path = os.path.join(flags_obj.model_dir, 'model.ckpt-{epoch:04d}')
  model_pruning_config = None
  if flags_obj.pruning_config_file:
    model_pruning_config = cprune_from_config._expand_model_pruning_config(
        model, pruning_params
    )

  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(
          ckpt_full_path, save_weights_only=True),
      custom_callbacks.CustomTensorBoard(
          log_dir=flags_obj.model_dir,
          track_lr=True,
          model_pruning_config=model_pruning_config,
      ),
  ]
  if flags_obj.pruning_config_file:
    callbacks.append(cpruning_callbacks.UpdateCPruningStep())

  num_eval_examples = mnist.info.splits['test'].num_examples
  num_eval_steps = num_eval_examples // flags_obj.batch_size

  if flags_obj.mode == 'train_and_eval':
    history = model.fit(
        train_input_dataset,
        epochs=train_epochs,
        steps_per_epoch=train_steps,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        validation_steps=num_eval_steps,
        validation_data=eval_input_dataset,
        validation_freq=flags_obj.epochs_between_evals)
  elif flags_obj.mode == 'eval':
    callbacks = None
    history = cprune.apply_cpruning_masks(model)
  elif flags_obj.mode == 'sensitivity_analysis':
    callbacks, history = None, None
  else:
    raise ValueError('{} is not a valid mode.'.format(flags.FLAGS.mode))

  export_path = os.path.join(flags_obj.model_dir, 'saved_model')
  model.save(export_path, include_optimizer=False)

  if flags_obj.pruning_config_file:
    _pruning_params = cprune_from_config.predict_sparsity(model, pruning_params)
    logging.info('Pruning result: %s', pp.pformat(_pruning_params))

  if flags_obj.mode == 'sensitivity_analysis':
    eval_output = None
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
  else:
    eval_output = model.evaluate(
        eval_input_dataset, steps=num_eval_steps, verbose=2)

  stats = common.build_stats(history, eval_output, callbacks)
  return stats


def define_mnist_flags():
  """Define command line flags for MNIST model."""
  flags_core.define_base(
      clean=True,
      num_gpu=True,
      train_epochs=True,
      epochs_between_evals=True,
      distribution_strategy=True)
  flags_core.define_device()
  flags_core.define_distribution()
  flags.DEFINE_bool('download', False,
                    'Whether to download data to `--data_dir`.')
  FLAGS.set_default('batch_size', 1024)
  flags.DEFINE_string('pruning_config_file', None,
                      'Path to a yaml file of model pruning configuration.')
  flags.DEFINE_string(
      'mode',
      default=None,
      help='Mode to run: `train_and_eval`, `eval`, or `sensitivity_analysis.')
  flags.DEFINE_integer(
      'sensitivity_layer_count',
      default=0,
      help='The ordinal number representing a layer whose pruning sensitivity '
           'is to be analyzed. 0 for `"conv2d"` (the first layer), 3 for '
           '`"dense_1"` (the last layer) etc. Valid only if '
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
  flags.DEFINE_bool('resume_checkpoint', None,
                    'Whether or not to enable load checkpoint loading. Defaults '
                    'to None.')


def main(_):
  model_helpers.apply_clean(FLAGS)
  stats = run(flags.FLAGS)
  logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_mnist_flags()
  app.run(main)
