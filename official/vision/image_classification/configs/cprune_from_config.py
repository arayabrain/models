"""Enables pruning with constraints from a model-level configuration dict."""

import copy
import re

import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from tensorflow_model_optimization.python.core.sparsity.keras import cpruning_granularity as pruning_granu
from tensorflow_model_optimization.python.core.sparsity.keras import cprune_registry

from official.vision.image_classification.configs import pruning_configs


deserialize_keras_object = tf.keras.utils.deserialize_keras_object


def _expand_layer_pruning_config(layer, layer_pruning_config):
  """Expand a LayerPruningConfig.

  Arguments:
    layer: A tf.keras layer.
    layer_pruning_config: A LayerPruningConfig instance.

  Returns:
    layer_pruning_config: A new LayerPruningConfig instance.
  """
  layer_pruning_config = copy.deepcopy(layer_pruning_config)

  if layer_pruning_config.pruning is None:
    layer_pruning_config.pruning = []
  elif isinstance(layer_pruning_config.pruning, pruning_configs.PruningConfig):
    pruning_config = layer_pruning_config.pruning
    layer_pruning_config.pruning = []
    weight_constraint_names = cprune_registry.CPruneRegistry.get_pruning_weight_constraint_names(layer)
    for (weight_name, _) in weight_constraint_names:
      weight_pruning_config = pruning_configs.WeightPruningConfig(
          weight_name=weight_name, pruning=pruning_config)
      layer_pruning_config.pruning.append(weight_pruning_config)

  return layer_pruning_config


def _expand_model_pruning_config(model, model_pruning_config):
  """Expand a ModelPruningConfig.

  Arguments:
    model: A tf.keras model.
    model_pruning_config: A ModelPruningConfig instance.

  Returns:
    model_pruning_config: A new ModelPruningConfig instance.
  """
  model_pruning_config = copy.deepcopy(model_pruning_config)

  if model_pruning_config.pruning is None:
    model_pruning_config.pruning = []
  elif isinstance(model_pruning_config.pruning, pruning_configs.PruningConfig):
    pruning_config = model_pruning_config.pruning
    model_pruning_config.pruning = []
    for layer in model.layers:
      if layer not in model._input_layers:
        layer_pruning_config = pruning_configs.LayerPruningConfig(
            layer_name=layer.name, pruning=pruning_config)
        model_pruning_config.pruning.append(
            _expand_layer_pruning_config(layer, layer_pruning_config))
  else:
    layer_pruning_configs = model_pruning_config.pruning
    model_pruning_config.pruning = []
    all_layer_names = [layer.name for layer in model.layers]
    # A dict of str (layer name): LayerPruningConfig.
    _config_dict = dict()
    for layer_pruning_config in layer_pruning_configs:
      _layer_name = layer_pruning_config.layer_name
      if _layer_name in all_layer_names:
        layer = model.get_layer(_layer_name)
        layer_pruning_config = _expand_layer_pruning_config(layer, layer_pruning_config)
        _config_dict[_layer_name] = layer_pruning_config
      else:
        for layer_name in all_layer_names:
          if re.search(_layer_name, layer_name):
            layer = model.get_layer(layer_name)
            layer_pruning_config = _expand_layer_pruning_config(layer, layer_pruning_config)
            _config_dict[layer_name] = layer_pruning_config
    for layer_name, layer_pruning_config in _config_dict.items():
      layer_pruning_config.name = layer_name
      model_pruning_config.pruning.append(layer_pruning_config)

  return model_pruning_config


def _convert_config(model, model_pruning_config):
  """Convert a ModelPruningConfig.

  Arguments:
    model: A tf.keras model.
    model_pruning_config: A ModelPruningConfig instance.

  Returns:
    model_pruning_config: A new ModelPruningConfig instance.
  """
  model_pruning_config = _expand_model_pruning_config(model, model_pruning_config)
  assert model_pruning_config.model_name == model.name
  assert model_pruning_config.prune_func == 'cprune_low_magnitude'

  if model_pruning_config.share_mask is None:
    model_pruning_config.share_mask = []
  if model_pruning_config.share_mask:
    for mask_sharing_config in model_pruning_config.share_mask:
      layer_names = mask_sharing_config.layer_names
      layer_pruning_configs = []
      for layer_name in layer_names:
        for i, layer_pruning_config in enumerate(model_pruning_config.pruning):
          if layer_pruning_config.layer_name == layer_name:
            layer_pruning_configs.append(layer_pruning_config)
            model_pruning_config.pruning.pop(i)
      assert len(layer_names) == len(layer_pruning_configs)
      assert all(layer_pruning_config.pruning == layer_pruning_configs[0].pruning
                 for layer_pruning_config in layer_pruning_configs)
      mask_sharing_config.pruning = layer_pruning_configs[0].pruning
  for i, layer_pruning_config in enumerate(model_pruning_config.pruning):
    mask_sharing_config = pruning_configs.MaskSharingConfig(
        layer_names=[layer_pruning_config.layer_name])
    mask_sharing_config.pruning = layer_pruning_config.pruning
    model_pruning_config.pruning.pop(i)
    model_pruning_config.share_mask.append(mask_sharing_config)

  assert len(model_pruning_config.pruning) == 0
  return model_pruning_config


def _deserialize_config(model, model_pruning_config):
  """Deserialize pruning schedules and granularities of a ModelPruningConfig.

  Arguments:
    model: A tf.keras model.
    model_pruning_config: A ModelPruningConfig instance.

  Returns:
    model_pruning_config: A new ModelPruningConfig instance.
  """

  def get_weight_constraint(pruning_schedule, pruning_granularity):
    return pruning_granularity.constraint_type(
        pruning_schedule=pruning_schedule,
        **pruning_granularity.get_config()['config'])

  model_pruning_config = _convert_config(model, model_pruning_config)
  for mask_sharing_config in model_pruning_config.share_mask:
    for layer_pruning_config in mask_sharing_config.pruning:
      for weight_pruning_config in layer_pruning_config.pruning:
        custom_objects = {
          'ConstantSparsity': pruning_sched.ConstantSparsity,
          'PolynomialDecay': pruning_sched.PolynomialDecay,
          'BlockSparsity': pruning_granu.BlockSparsity,
          'ChannelPruning': pruning_granu.ChannelPruning,
          'KernelLevel': pruning_granu.KernelLevel,
          'QuasiCyclic': pruning_granu.QuasiCyclic,
        }
        pruning_schedule = deserialize_keras_object(
            weight_pruning_config.pruning_schedule.as_dict(),
            module_objects=globals(),
            custom_objects=custom_objects)
        pruning_granularity = deserialize_keras_object(
            weight_pruning_config.pruning_granularity.as_dict(),
            module_objects=globals(),
            custom_objects=custom_objects)
        weight_pruning_config.pruning_schedule = pruning_schedule
        weight_pruning_config.pruning_granularity = pruning_granularity
        weight_pruning_config.constraint = get_weight_constraint(
            pruning_schedule, pruning_granularity)

  return model_pruning_config


def report_target_sparsity(model, model_pruning_config):
  pass


def cprune_low_magnitude_from_config(model, model_pruning_config):
  """Modify a tf.keras model to be pruned during training.

  Arguments:
    model: A tf.keras model.
    model_pruning_config: A ModelPruningConfig instance.

  Returns:
    model: A new tf.keras model.
    model_pruning_config: A new ModelPruningConfig instance.
  """
  model_pruning_config = _deserialize_config(model, model_pruning_config)
  weights = model.get_weights()

  def clone_function(layer):
    layer_config = layer.get_config()
    for mask_sharing_config in model_pruning_config:
      if layer.name in mask_sharing_config.layer_names:
        for layer_pruning_config in mask_sharing_config.pruning:
          for weight_pruning_config in layer_pruning_config.pruning:
            weight_name = weight_pruning_config.weight_name
            constraint_name = cprune_registry.ConstraintRegistry._WEIGHTS_CONSTRAINS_MAP[weight_name]
            constraint = weight_pruning_config.constraint
            layer_config[constraint_name] = constraint
    return layer.__class__.from_config(layer_config)

  model = tf.keras.models.clone_model(
      model, input_tensors=None, clone_function=clone_function)
  model.set_weights(weights)
  return model
