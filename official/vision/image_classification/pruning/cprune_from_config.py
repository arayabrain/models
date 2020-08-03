"""Enables pruning with constraints from a model-level configuration dict."""

import copy
import re

import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from tensorflow_model_optimization.python.core.sparsity.keras import cpruning_granularity as pruning_granu
from tensorflow_model_optimization.python.core.sparsity.keras import cprune_registry

from official.vision.image_classification.pruning import  pruning_base_configs


ModelPruningConfig = pruning_base_configs.ModelPruningConfig
MaskSharingConfig = pruning_base_configs.MaskSharingConfig
LayerPruningConfig = pruning_base_configs.LayerPruningConfig
WeightPruningConfig = pruning_base_configs.WeightPruningConfig
PruningConfig = pruning_base_configs.PruningConfig


K = tf.keras.backend
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
  elif isinstance(layer_pruning_config.pruning, PruningConfig):
    pruning_config = layer_pruning_config.pruning
    layer_pruning_config.pruning = []
    weight_constraint_names = cprune_registry.CPruneRegistry.get_pruning_weight_constraint_names(layer)
    for (weight_name, _) in weight_constraint_names:
      weight_pruning_config = WeightPruningConfig(
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
  assert model_pruning_config.model_name == model.name

  if model_pruning_config.pruning is None:
    model_pruning_config.pruning = []
  elif isinstance(model_pruning_config.pruning, PruningConfig):
    pruning_config = model_pruning_config.pruning
    model_pruning_config.pruning = []
    for layer in model.layers:
      if layer.trainable_weights:
        layer_pruning_config = LayerPruningConfig(
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
    model_pruning_dict: A dict.
  """
  model_pruning_config = _expand_model_pruning_config(model, model_pruning_config)
  if model_pruning_config.share_mask is None:
    model_pruning_config.share_mask = []
  model_pruning_dict = dict()

  model_pruning_dict['share_mask'] = []
  for mask_sharing_config in model_pruning_config.share_mask:
    # TODO: debug this for block.
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
    mask_sharing_config._pruning = layer_pruning_configs[0].pruning
  for layer_pruning_config in model_pruning_config.pruning:
    mask_sharing_dict = MaskSharingConfig(
        layer_names=[layer_pruning_config.layer_name]).as_dict()
    mask_sharing_dict['pruning'] = layer_pruning_config.as_dict()['pruning']
    model_pruning_dict['share_mask'].append(mask_sharing_dict)

  return model_pruning_dict


def _deserialize_config(model, model_pruning_config):
  """Deserialize pruning schedules and granularities of a ModelPruningConfig.

  Arguments:
    model: A tf.keras model.
    model_pruning_config: A ModelPruningConfig instance.

  Returns:
    model_pruning_dict: A dict
  """

  def _init_constraint(weight_pruning_dict):
    custom_objects = {
      'ConstantSparsity': pruning_sched.ConstantSparsity,
      'PolynomialDecay': pruning_sched.PolynomialDecay,
      'ArayaMag': pruning_granu.ArayaMag,
      'BlockSparsity': pruning_granu.BlockSparsity,
      'ChannelPruning': pruning_granu.ChannelPruning,
      'KernelLevel': pruning_granu.KernelLevel,
      'QuasiCyclic': pruning_granu.QuasiCyclic,
    }
    schedule = deserialize_keras_object(
      weight_pruning_dict['pruning']['pruning_schedule'],
      module_objects=globals(),
      custom_objects=custom_objects)
    granularity = deserialize_keras_object(
      weight_pruning_dict['pruning']['pruning_granularity'],
      module_objects=globals(),
      custom_objects=custom_objects)
    weight_pruning_dict['pruning']['constraint'] = granularity.get_constraint(schedule)
    return weight_pruning_dict

  model_pruning_dict = _convert_config(model, model_pruning_config)
  for i, mask_sharing_dict in enumerate(model_pruning_dict['share_mask']):
    for j, weight_pruning_dict in enumerate(mask_sharing_dict['pruning']):
      model_pruning_dict['share_mask'][i]['pruning'][j] = _init_constraint(weight_pruning_dict)

  return model_pruning_dict


def predict_sparsity(model, model_pruning_config):
  """Predict sparsity after pruning out of a model and a ModelPruningConfig.

  Arguments:
    model: A tf.keras model.
    model_pruning_config: A ModelPruningConfig instance.

  Returns:
    model_sparsity_dict: A dict.
  """
  model_sparsity_dict = _expand_model_pruning_config(model, model_pruning_config).as_dict()
  for layer_pruning_dict in model_sparsity_dict['pruning']:
    for weight_pruning_dict in layer_pruning_dict['pruning']:
      custom_objects = {
        'ConstantSparsity': pruning_sched.ConstantSparsity,
        'PolynomialDecay': pruning_sched.PolynomialDecay,
      }
      pruning_schedule = deserialize_keras_object(
          weight_pruning_dict['pruning']['pruning_schedule'],
          module_objects=globals(),
          custom_objects=custom_objects)
      should_prune, target_sparsity = pruning_schedule(
          pruning_schedule.get_final_update_step())
      assert bool(should_prune.numpy())
      weight_pruning_dict['predicted_sparsity'] = float(target_sparsity.numpy())
  return model_sparsity_dict


def cprune_from_config(model, model_pruning_config):
  """Modify a tf.keras model to be pruned during training.

  Arguments:
    model: A tf.keras model.
    model_pruning_config: A ModelPruningConfig instance.

  Returns:
    model: A new tf.keras model.
  """
  model_pruning_dict = _deserialize_config(model, model_pruning_config)
  weights = model.get_weights()

  def clone_function(layer):
    layer_config = layer.get_config()
    for mask_sharing_dict in model_pruning_dict['share_mask']:
      if layer.name in mask_sharing_dict['layer_names']:
        for weight_pruning_dict in mask_sharing_dict['pruning']:
          weight_name = weight_pruning_dict['weight_name']
          constraint_name = cprune_registry.ConstraintRegistry._WEIGHTS_CONSTRAINS_MAP[weight_name]
          constraint = weight_pruning_dict['pruning']['constraint']
          layer_config[constraint_name] = constraint
    return layer.__class__.from_config(layer_config)

  model = tf.keras.models.clone_model(
      model, input_tensors=None, clone_function=clone_function)
  model.set_weights(weights)
  return model
