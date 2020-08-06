# Lint as: python3
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
"""Common modules for callbacks."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import functools
import os
from absl import logging

import numpy as np
import tensorflow as tf
from typing import Any, List, Optional, MutableMapping

from official.utils.misc import keras_utils
from official.vision.image_classification.pruning.pruning_base_configs import ModelPruningConfig
from tensorflow_model_optimization.python.core.keras import compat
from tensorflow_model_optimization.python.core.sparsity.keras.cprune_registry import ConstraintRegistry



def get_callbacks(model_checkpoint: bool = True,
                  include_tensorboard: bool = True,
                  time_history: bool = True,
                  track_lr: bool = True,
                  model_pruning_config: Optional[ModelPruningConfig] = None,
                  write_model_weights: bool = True,
                  batch_size: int = 0,
                  log_steps: int = 0,
                  model_dir: str = None) -> List[tf.keras.callbacks.Callback]:
  """Get all callbacks."""
  model_dir = model_dir or ''
  callbacks = []
  if model_checkpoint:
    ckpt_full_path = os.path.join(model_dir, 'model.ckpt-{epoch:04d}')
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_full_path, save_weights_only=True, verbose=1))
  if include_tensorboard:
    callbacks.append(
        CustomTensorBoard(
            log_dir=model_dir,
            track_lr=track_lr,
            model_pruning_config=model_pruning_config,
            write_images=write_model_weights))
  if time_history:
    callbacks.append(
        keras_utils.TimeHistory(
            batch_size,
            log_steps,
            logdir=model_dir if include_tensorboard else None))
  return callbacks


def get_scalar_from_tensor(t: tf.Tensor) -> int:
  """Utility function to convert a Tensor to a scalar."""
  t = tf.keras.backend.get_value(t)
  if callable(t):
    return t()
  else:
    return t


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
  """A customized TensorBoard callback that tracks additional datapoints.

  Metrics tracked:
  - Global learning rate

  Attributes:
    log_dir: the path of the directory where to save the log files to be parsed
      by TensorBoard.
    track_lr: `bool`, whether or not to track the global learning rate.
    **kwargs: Additional arguments for backwards compatibility. Possible key is
      `period`.
  """

  # TODO(b/146499062): track params, flops, log lr, l2 loss,
  # classification loss

  def __init__(self,
               log_dir: str,
               track_lr: bool = False,
               model_pruning_config: Optional[ModelPruningConfig] = None,
               **kwargs):
    super(CustomTensorBoard, self).__init__(log_dir=log_dir, **kwargs)
    self._track_lr = track_lr
    self._model_pruning_config = model_pruning_config

  def _collect_learning_rate(self, logs):
    logs = logs or {}
    lr_schedule = getattr(self.model.optimizer, "lr", None)
    if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
        logs["learning_rate"] = tf.keras.backend.get_value(
            lr_schedule(self.model.optimizer.iterations)
        )
        if isinstance(logs["learning_rate"], functools.partial):
          logs["learning_rate"] = logs["learning_rate"]()
    return logs

  def _log_metrics(self, logs, prefix, step):
    if self._track_lr:
      super()._log_metrics(self._collect_learning_rate(logs), prefix, step)

  def _log_pruning_metrics(self, logs, prefix, step):
    if compat.is_v1_apis():
      # Safely depend on TF 1.X private API given
      # no more 1.X releases.
      self._write_custom_summaries(step, logs)
    else:  # TF 2.X
      log_dir = self.log_dir + '/metrics'

      file_writer = tf.summary.create_file_writer(log_dir)
      file_writer.set_as_default()

      for name, value in logs.items():
        tf.summary.scalar(name, value, step=step)

      file_writer.flush()

  def on_epoch_begin(self, epoch, logs=None):
    if logs is not None:
      super(CustomTensorBoard, self).on_epoch_begin(epoch, logs)

    if self._model_pruning_config:
      pruning_logs = {}
      params = []
      postfixes = []

      for layer_pruning_config in self._model_pruning_config.pruning:
        layer_name = layer_pruning_config.layer_name
        layer = self.model.get_layer(layer_name)
        for weight_pruning_config in layer_pruning_config.pruning:
          weight_name = weight_pruning_config.weight_name
          constraint_name = ConstraintRegistry.get_constraint_from_weight(weight_name)
          constraint = getattr(layer, constraint_name)
          params.append(constraint.mask)
          params.append(constraint.threshold)
          postfixes.append('/' + layer_name + '/' + weight_name)

      params.append(self.model.optimizer.iterations)

      values = tf.keras.backend.batch_get_value(params)
      iteration = values[-1]
      del values[-1]
      del params[-1]

      param_value_pairs = list(zip(params, values))

      for (mask, mask_value), postfix in zip(param_value_pairs[::2], postfixes):
        pruning_logs.update({
            'mask_sparsity' + postfix: 1 - np.mean(mask_value)
        })

      for (threshold, threshold_value), postfix in zip(param_value_pairs[1::2], postfixes):
        pruning_logs.update({'threshold' + postfix: threshold_value})

      self._log_pruning_metrics(pruning_logs, '', iteration)
