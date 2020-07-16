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
"""Configuration definitions for MobileNetV1 losses, learning rates, and optimizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Mapping

import dataclasses

from official.vision.image_classification.configs import base_configs


_MBNETV1_LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]
_MBNETV1_LR_BOUNDARIES = list(p[1] for p in _MBNETV1_LR_SCHEDULE[1:])
_MBNETV1_LR_MULTIPLIERS = list(p[0] for p in _MBNETV1_LR_SCHEDULE)
_MBNETV1_LR_WARMUP_EPOCHS = _MBNETV1_LR_SCHEDULE[0][1]


@dataclasses.dataclass
class MobileNetV1ModelConfig(base_configs.ModelConfig):
  """Configuration for the MobileNetV1 model."""
  name: str = 'MobileNetV1'
  num_classes: int = 1000
  model_params: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {
      'num_classes': 1000,
      'batch_size': None,
      # 'use_l2_regularizer': True,
      # 'rescale_inputs': False,
  })
  loss: base_configs.LossConfig = base_configs.LossConfig(
      name='sparse_categorical_crossentropy')
  optimizer: base_configs.OptimizerConfig = base_configs.OptimizerConfig(
      name='momentum',
      decay=0.9,
      epsilon=0.001,
      momentum=0.9,
      moving_average_decay=None)
  learning_rate: base_configs.LearningRateConfig = (
      base_configs.LearningRateConfig(
          name='piecewise_constant_with_warmup',
          examples_per_epoch=1281167,
          warmup_epochs=_MBNETV1_LR_WARMUP_EPOCHS,
          boundaries=_MBNETV1_LR_BOUNDARIES,
          multipliers=_MBNETV1_LR_MULTIPLIERS))
