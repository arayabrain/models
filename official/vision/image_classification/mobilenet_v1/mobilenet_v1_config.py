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


@dataclasses.dataclass
class MobileNetV1ModelConfig(base_configs.ModelConfig):
  """Configuration for the MobileNetV1 model."""
  name: str = 'MobileNetV1'
  num_classes: int = 1000
  model_params: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {
      'num_classes': 1000,
      'batch_size': None,
  })
  loss: base_configs.LossConfig = base_configs.LossConfig(
      name='sparse_categorical_crossentropy')
  optimizer: base_configs.OptimizerConfig = base_configs.OptimizerConfig(
      name='rmsprop',
      decay=0.9,
      epsilon=0.001,
      momentum=0.9,
      moving_average_decay=None)
  learning_rate: base_configs.LearningRateConfig = base_configs.LearningRateConfig(  # pylint: disable=line-too-long
      name='exponential',
      initial_lr=0.045,
      decay_epochs=2.5,
      decay_rate=0.94)
