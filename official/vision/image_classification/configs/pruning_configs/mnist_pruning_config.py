"""Configuration definitions for MNIST pruning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional

import dataclasses

from official.vision.image_classification.configs import pruning_configs


ModelPruningConfig = pruning_configs.pruning_base_configs.ModelPruningConfig
LayerPruningConfig = pruning_configs.pruning_base_configs.LayerPruningConfig
WeightPruningConfig = pruning_configs.pruning_base_configs.WeightPruningConfig
PruningConfig = pruning_configs.pruning_base_configs.PruningConfig


@dataclasses.dataclass
class OnlyKernelPruningConfig(LayerPruningConfig):
  """Configuration for pruning a Dense or Conv2D layer."""
  layer_name: Optional[str] = None
  pruning: List[WeightPruningConfig] = dataclasses.field(
      default_factory=lambda: [WeightPruningConfig(
          weight_name='kernel',
          pruning=None,
      )]
  )


@dataclasses.dataclass
class MNISTPruningConfig(ModelPruningConfig):
  """Configuration for pruning the model for MNIST."""
  model_name: str = 'mnist'
  pruning: List[LayerPruningConfig] = dataclasses.field(
      default_factory=lambda: [
          OnlyKernelPruningConfig(layer_name='conv2d'),
          OnlyKernelPruningConfig(layer_name='conv2d_1'),
          OnlyKernelPruningConfig(layer_name='dense'),
          OnlyKernelPruningConfig(layer_name='dense_1'),
      ]
  )
  share_mask: None = None

