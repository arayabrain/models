"""Configuration definitions for pruning ResNet-50."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional

import dataclasses

from official.vision.image_classification.pruning import pruning_base_configs


_PRUNABLE_LAYERS = [
    'conv1',
    'res2a_branch2a',
    'res2a_branch2b',
    'res2a_branch2c',
    'res2a_branch1',
    'res2b_branch2a',
    'res2b_branch2b',
    'res2b_branch2c',
    'res2c_branch2a',
    'res2c_branch2b',
    'res2c_branch2c',
    'res3a_branch2a',
    'res3a_branch2b',
    'res3a_branch2c',
    'res3a_branch1',
    'res3b_branch2a',
    'res3b_branch2b',
    'res3b_branch2c',
    'res3c_branch2a',
    'res3c_branch2b',
    'res3c_branch2c',
    'res3d_branch2a',
    'res3d_branch2b',
    'res3d_branch2c',
    'res4a_branch2a',
    'res4a_branch2b',
    'res4a_branch2c',
    'res4a_branch1',
    'res4b_branch2a',
    'res4b_branch2b',
    'res4b_branch2c',
    'res4c_branch2a',
    'res4c_branch2b',
    'res4c_branch2c',
    'res4d_branch2a',
    'res4d_branch2b',
    'res4d_branch2c',
    'res4e_branch2a',
    'res4e_branch2b',
    'res4e_branch2c',
    'res4f_branch2a',
    'res4f_branch2b',
    'res4f_branch2c',
    'res5a_branch2a',
    'res5a_branch2b',
    'res5a_branch2c',
    'res5a_branch1',
    'res5b_branch2a',
    'res5b_branch2b',
    'res5b_branch2c',
    'res5c_branch2a',
    'res5c_branch2b',
    'res5c_branch2c',
    'fc1000',
]


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
class ResNet50PruningConfig(pruning_base_configs.ModelPruningConfig):
  """Configuration for pruning ResNet-50."""
  model_name: str = 'resnet50'
  pruning: List[pruning_base_configs.LayerPruningConfig] = dataclasses.field(
      default_factory=lambda: [
          OnlyKernelPruningConfig(layer_name=layer_name)
          for layer_name in _PRUNABLE_LAYERS
      ]
  )
  share_mask: None = None
