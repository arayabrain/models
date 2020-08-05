"""Configuration definitions for pruning MobileNetV1."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional

import dataclasses

from official.vision.image_classification.pruning import pruning_base_configs


_PRUNABLE_LAYERS = [
    'Conv2d_0',
    'Conv2d_1_depthwise',
    'Conv2d_1_pointwise',
    'Conv2d_2_depthwise',
    'Conv2d_2_pointwise',
    'Conv2d_3_depthwise',
    'Conv2d_3_pointwise',
    'Conv2d_4_depthwise',
    'Conv2d_4_pointwise',
    'Conv2d_5_depthwise',
    'Conv2d_5_pointwise',
    'Conv2d_6_depthwise',
    'Conv2d_6_pointwise',
    'Conv2d_7_depthwise',
    'Conv2d_7_pointwise',
    'Conv2d_8_depthwise',
    'Conv2d_8_pointwise',
    'Conv2d_9_depthwise',
    'Conv2d_9_pointwise',
    'Conv2d_10_depthwise',
    'Conv2d_10_pointwise',
    'Conv2d_11_depthwise',
    'Conv2d_11_pointwise',
    'Conv2d_12_depthwise',
    'Conv2d_12_pointwise',
    'Conv2d_13_depthwise',
    'Conv2d_13_pointwise',
    'Conv2d_1c_1x1',
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
class MobileNetV1PruningConfig(pruning_base_configs.ModelPruningConfig):
  """Configuration for pruning MobileNetV1."""
  model_name: str = 'mobilenetV1'
  pruning: List[pruning_base_configs.LayerPruningConfig] = dataclasses.field(
      default_factory=lambda: [
          OnlyKernelPruningConfig(layer_name=layer_name)
          for layer_name in _PRUNABLE_LAYERS
      ]
  )
  share_mask: None = None
