"""Configuration definitions for pruning ResNet-56."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional

import dataclasses

from official.vision.image_classification.pruning import pruning_base_configs


_PRUNABLE_LAYERS = [
    'conv1',
    'res2block_0_branch2a',
    'res2block_0_branch2b',
    'res2block_0_branch1',
    'res2block_1_branch2a',
    'res2block_1_branch2b',
    'res2block_2_branch2a',
    'res2block_2_branch2b',
    'res2block_3_branch2a',
    'res2block_3_branch2b',
    'res2block_4_branch2a',
    'res2block_4_branch2b',
    'res2block_5_branch2a',
    'res2block_5_branch2b',
    'res2block_6_branch2a',
    'res2block_6_branch2b',
    'res2block_7_branch2a',
    'res2block_7_branch2b',
    'res2block_8_branch2a',
    'res2block_8_branch2b',
    'res3block_0_branch2a',
    'res3block_0_branch2b',
    'res3block_0_branch1',
    'res3block_1_branch2a',
    'res3block_1_branch2b',
    'res3block_2_branch2a',
    'res3block_2_branch2b',
    'res3block_3_branch2a',
    'res3block_3_branch2b',
    'res3block_4_branch2a',
    'res3block_4_branch2b',
    'res3block_5_branch2a',
    'res3block_5_branch2b',
    'res3block_6_branch2a',
    'res3block_6_branch2b',
    'res3block_7_branch2a',
    'res3block_7_branch2b',
    'res3block_8_branch2a',
    'res3block_8_branch2b',
    'res4block_0_branch2a',
    'res4block_0_branch2b',
    'res4block_0_branch1',
    'res4block_1_branch2a',
    'res4block_1_branch2b',
    'res4block_2_branch2a',
    'res4block_2_branch2b',
    'res4block_3_branch2a',
    'res4block_3_branch2b',
    'res4block_4_branch2a',
    'res4block_4_branch2b',
    'res4block_5_branch2a',
    'res4block_5_branch2b',
    'res4block_6_branch2a',
    'res4block_6_branch2b',
    'res4block_7_branch2a',
    'res4block_7_branch2b',
    'res4block_8_branch2a',
    'res4block_8_branch2b',
    'fc10',
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
class ResNet56PruningConfig(pruning_base_configs.ModelPruningConfig):
  """Configuration for pruning ResNet-56."""
  model_name: str = 'resnet56'
  pruning: List[pruning_base_configs.LayerPruningConfig] = dataclasses.field(
      default_factory=lambda: [
          OnlyKernelPruningConfig(layer_name=layer_name)
          for layer_name in _PRUNABLE_LAYERS
      ]
  )
  share_mask: None = None
