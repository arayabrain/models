"""Definitions for pruning configurations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from typing import Any, List, Mapping, Optional, Union

import dataclasses

from official.modeling.hyperparams import base_config


@dataclasses.dataclass
class PruningScheduleConfig(base_config.Config):
  """Configuration for a pruning schedule.
  Attributes:
    class_name: E.g. 'ConstantSparsity' or 'PolynomialDecay'.
    config: A dict of configuration for that pruning schedule class.
  """
  class_name: str = 'ConstantSparsity'
  config: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {
      'target_sparsity': 0.5,
      'begin_step': 0,
      'end_step': -1,
      'frequency': 100,
  })


@dataclasses.dataclass
class PruningGranularityConfig(base_config.Config):
  """Configuration for pruning granularity.
  Attributes:
    class_name: E.g. 'BlockSparsity', 'ChannelPruning', 'KernelLevel', or
      'QuasiCyclic'.
    config: A dict of configuration for that pruning granularity class.
  """
  class_name: str = 'BlockSparsity'
  config: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {
      'block_size': (1, 1),
      'block_pooling_type': 'AVG',
  })


@dataclasses.dataclass
class PruningConfig(base_config.Config):
  """Configuration for pruning schedule and granularity.
  Attributes:
    pruning_schedule: A PruningScheduleConfig that controls pruning rate throughout
      training.
    pruning_granularity: A PruningGranularityConfig that controls pruning granularity
      throughout training.
  """
  pruning_schedule: PruningScheduleConfig = PruningScheduleConfig()
  pruning_granularity: PruningGranularityConfig = PruningGranularityConfig()


@dataclasses.dataclass
class WeightPruningConfig(base_config.Config):
  """Configuration for pruning of a weight tensor.
  Attributes:
    weight_name: The name of a weight to be pruned.
    pruning: How to prune that weight. Defaults to None (no pruning). Otherwise
      a PruningConfig.
  """
  weight_name: Optional[str] = None
  pruning: Optional[PruningConfig] = None


@dataclasses.dataclass
class LayerPruningConfig(base_config.Config):
  """Configuration for pruning of a Keras layer.
  Attributes:
    layer_name: The name of a layer to be pruned.
    pruning: How to prune that layer. Defaults to None (no pruning).  Otherwise
      a PruningConfig or a list of WeightPruningConfig's.
  """
  layer_name: Optional[str] = None
  pruning: Optional[Union[PruningConfig, List[WeightPruningConfig]]] = None


@dataclasses.dataclass
class MaskSharingConfig(base_config.Config):
  """Configuration for mask sharing among layers.
  This config is dedicated to channel pruning.
  Attributes:
    layer_names: The names of layers for which pruning masks are shared.
      Defaults to an empty list (no mask sharing).
  """
  layer_names: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ModelPruningConfig(base_config.Config):
  """Configuration for pruning a Keras model.
  Attributes:
    model_name: The name of a model to be pruned.
    prune_func: The pruning function to be used. Defaults to
      'cprune_low_magnitude'.
    pruning: How to prune that model. Defaults to None (no pruning). Otherwise
      a PruningConfig or a list of LayerPruningConfig's.
    share_mask: Specifies how pruning masks are shared across layers. Defaults
      to None (no mask sharing). Otherwise a list of MaskSharingConfig's.
  """
  model_name: Optional[str] = None
  prune_func: str = 'cprune_low_magnitude'
  pruning: Optional[Union[PruningConfig, List[LayerPruningConfig]]] = None
  share_mask: Optional[List[MaskSharingConfig]] = None
