"""Configuration definitions for MNIST pruning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional

import dataclasses

from official.vision.image_classification.pruning import pruning_base_configs


@dataclasses.dataclass
class OnlyKernelPruningConfig(pruning_base_configs.LayerPruningConfig):
  """Configuration for pruning a Dense or Conv2D layer."""
  layer_name: Optional[str] = None
  pruning: List[pruning_base_configs.WeightPruningConfig] = dataclasses.field(
      default_factory=lambda: [pruning_base_configs.WeightPruningConfig(
          weight_name='kernel',
          pruning=None,
      )]
  )


@dataclasses.dataclass
class MNISTPruningConfig(pruning_base_configs.ModelPruningConfig):
  """Configuration for pruning the model for MNIST."""
  model_name: str = 'mnist'
  pruning: List[pruning_base_configs.LayerPruningConfig] = dataclasses.field(
      default_factory=lambda: [
          OnlyKernelPruningConfig(layer_name='conv2d'),
          OnlyKernelPruningConfig(layer_name='conv2d_1'),
          OnlyKernelPruningConfig(layer_name='dense'),
          OnlyKernelPruningConfig(layer_name='dense_1'),
      ]
  )
  share_mask: None = None
