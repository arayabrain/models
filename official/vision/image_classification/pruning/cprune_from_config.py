"""Enables pruning with constraints from a model-level configuration dict."""

import copy
import re
import string

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from tensorflow_model_optimization.python.core.sparsity.keras import cpruning_granularity as pruning_granu
from tensorflow_model_optimization.python.core.sparsity.keras import cprune_registry

from official.modeling.hyperparams import params_dict
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
            layer_pruning_config.layer_name = layer_name
            _config_dict[layer_name] = layer_pruning_config
    for layer_name, layer_pruning_config in _config_dict.items():
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
    layer_names = mask_sharing_config.layer_names
    mask_sharing_dict = mask_sharing_config.as_dict()
    layer_pruning_configs = []
    for layer_name in layer_names:
      for layer_pruning_config in model_pruning_config.pruning:
        if layer_pruning_config.layer_name == layer_name:
          layer_pruning_configs.append(layer_pruning_config)
    assert len(layer_names) == len(layer_pruning_configs)
    assert all(layer_pruning_config.pruning == layer_pruning_configs[0].pruning
               for layer_pruning_config in layer_pruning_configs)
    model_pruning_config.pruning = [
        x for x in model_pruning_config.pruning if x.layer_name not in layer_names
    ]
    mask_sharing_dict['pruning'] = layer_pruning_configs[0].as_dict()['pruning']
    model_pruning_dict['share_mask'].append(mask_sharing_dict)

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

  def _get_sparsity(weights):
    weights = K.get_value(weights)
    return 1.0 - np.count_nonzero(weights) / float(weights.size)

  def _get_shared_sparsity(weights_list, ch_axis=-1):
    weight_shape = weights_list[0].shape.as_list()
    norm_axis = list(range(len(weight_shape)))
    norm_axis.pop(ch_axis)

    abs_weights_list = [tf.abs(weights) for weights in weights_list]
    saliences_list = [math_ops.reduce_sum(abs_weights, axis=norm_axis, keepdims=True)
                      for abs_weights in abs_weights_list]
    weights = tf.add_n(saliences_list)
    return _get_sparsity(weights)

  model_sparsity_dict = _expand_model_pruning_config(model, model_pruning_config).as_dict()

  for layer_pruning_dict in model_sparsity_dict['pruning']:
    layer_name = layer_pruning_dict['layer_name']
    for weight_pruning_dict in layer_pruning_dict['pruning']:
      weight_name = weight_pruning_dict['weight_name']
      weights = getattr(model.get_layer(layer_name), weight_name)
      weight_pruning_dict['current_sparsity'] = _get_sparsity(weights)

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

  if model_sparsity_dict['share_mask']:
    for mask_sharing_dict in model_sparsity_dict['share_mask']:
      layer_names = mask_sharing_dict['layer_names']
      mask_sharing_dict['pruning'] = []
      for _layer_pruning_dict in model_sparsity_dict['pruning']:
        if _layer_pruning_dict['layer_name'] == layer_names[0]:
          layer_pruning_dict = _layer_pruning_dict
          break
      sharing_layers = [model.get_layer(layer_name) for layer_name in layer_names]
      for weight_pruning_dict in layer_pruning_dict['pruning']:
        weight_name = weight_pruning_dict['weight_name']
        assert weight_pruning_dict['pruning']['pruning_granularity']['class_name'] == 'ChannelPruning'
        ch_axis = weight_pruning_dict['pruning']['pruning_granularity']['config']['ch_axis']
        weights_list = [getattr(layer, weight_name) for layer in sharing_layers]
        mask_sharing_dict['pruning'].append(dict(
            weight_name=weight_name,
            current_shared_sparsity=_get_shared_sparsity(weights_list, ch_axis=ch_axis)
        ))

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


def _get_resnet_share_mask(model_name='resnet56'):
  """Returns a list of MaskSharingConfig's.

  Arguments:
    model_name: 'resnet56' (CIFAR-10) or 'resnet50' (ImageNet).

  ResNet-56;
    share_mask:
    - layer_names:
      - 'res2block_0_branch2b'
      - 'res2block_0_branch1'
      - 'res2block_1_branch2b'
      - 'res2block_2_branch2b'
      - 'res2block_3_branch2b'
      - 'res2block_4_branch2b'
      - 'res2block_5_branch2b'
      - 'res2block_6_branch2b'
      - 'res2block_7_branch2b'
      - 'res2block_8_branch2b'
    - layer_names:
      - 'res3block_0_branch2b'
      - 'res3block_0_branch1'
      - 'res3block_1_branch2b'
      - 'res3block_2_branch2b'
      - 'res3block_3_branch2b'
      - 'res3block_4_branch2b'
      - 'res3block_5_branch2b'
      - 'res3block_6_branch2b'
      - 'res3block_7_branch2b'
      - 'res3block_8_branch2b'
    - layer_names:
      - 'res4block_0_branch2b'
      - 'res4block_0_branch1'
      - 'res4block_1_branch2b'
      - 'res4block_2_branch2b'
      - 'res4block_3_branch2b'
      - 'res4block_4_branch2b'
      - 'res4block_5_branch2b'
      - 'res4block_6_branch2b'
      - 'res4block_7_branch2b'
      - 'res4block_8_branch2b'

  ResNet-50:
    share_mask:
    - layer_names:
      - 'res2a_branch2c'
      - 'res2a_branch1'
      - 'res2b_branch2c'
      - 'res2c_branch2c'
    - layer_names:
      - 'res3a_branch2c'
      - 'res3a_branch1'
      - 'res3b_branch2c'
      - 'res3c_branch2c'
      - 'res3d_branch2c'
    - layer_names:
      - 'res4a_branch2c'
      - 'res4a_branch1'
      - 'res4b_branch2c'
      - 'res4c_branch2c'
      - 'res4d_branch2c'
      - 'res4e_branch2c'
      - 'res4f_branch2c'
    - layer_names:
      - 'res5a_branch2c'
      - 'res5a_branch1'
      - 'res5b_branch2c'
      - 'res5c_branch2c'
  """
  share_mask = []

  if model_name == 'resnet56':

    for stage in range(2, 5):
      mask_sharing_config = pruning_base_configs.MaskSharingConfig()
      prefix = 'res' + str(stage) + 'block_'
      for block in range(9):
        mask_sharing_config.layer_names.append(prefix + str(block) + '_branch2b')
        if block == 0:
          mask_sharing_config.layer_names.append(prefix + str(block) + '_branch1')
      share_mask.append(mask_sharing_config)
  elif model_name == 'resnet50':
    stage_block_pairs = [(2, 3), (3, 4), (5, 6), (6, 3)]
    for stage, blocks in stage_block_pairs:
      mask_sharing_config = pruning_base_configs.MaskSharingConfig()
      prefix = 'res' + str(stage)
      for block in range(blocks):
        block_str = string.ascii_lowercase[block]
        mask_sharing_config.layer_names.append(prefix + block_str + '_branch2c')
        if block == 0:
          mask_sharing_config.layer_names.append(prefix + block_str + '_branch1')
      share_mask.append(mask_sharing_config)
  else:
    raise ValueError


def generate_pruning_config(model_name,
                            sparsity,
                            end_step,
                            schedule='ConstantSparsity',
                            granularity='BlockSparsity',
                            path=None):
  """Generate a model pruning config out of sparsity configuration.

  Arguments:
    model_name: A `str`. 'mnist', 'resnet56' (CIFAR-10), 'resnet50' (ImageNet),
      or 'mobilenetV1'.
    sparsity: A `dict`. Keys are `str` representing layer names (or possibly a
      regexp pattern), and values are sparsity (must be convertible to float).
    schedule: 'ConstantSparsity' or 'PolynomialDecay'.
    granularity: 'ArayaMag', 'BlockSparsity', 'ChannelPruning', 'KernelLevel',
      or 'QuasiCyclic'.
    path: `None` or a `str`. If `str`, saves the model pruning config as YAML
      file.

  Returns:
    A ModelPruningConfig instance.
  """

  def get_pruning_schedule_config(_sparsity):
    _sparsity = float(_sparsity)
    config = dict(begin_step=0, end_step=end_step, frequency=100)
    if schedule == 'ConstantSparsity':
      config['target_sparsity'] = _sparsity
    elif schedule == 'PolynomialDecay':
      config['initial_sparsity'] = 0.
      config['final_sparsity'] = _sparsity
      config['power'] = 3
    else:
      raise ValueError
    return pruning_base_configs.PruningScheduleConfig(
      class_name=schedule,
      config=config
    )

  def get_pruning_granularity_config(_sparsity):
    _sparsity = float(_sparsity)
    config = dict()
    if granularity in ('ArayaMag', 'QuasiCyclic'):
      config['gamma'] = int(1/_sparsity)
    elif granularity == 'BlockSparsity':
      config['block_size'] = (1, 1)
      config['block_pooling_type'] = 'AVG'
    elif granularity == 'ChannelPruning':
      config['ch_axis'] = -1
    elif granularity == 'KernelLevel':
      config['ker_axis'] =(0, 1)
    else:
      raise ValueError
    return pruning_base_configs.PruningGranularityConfig(
      class_name=granularity,
      config=config,
    )

  def get_pruning_config(_sparsity):
    return pruning_base_configs.PruningConfig(
      pruning_schedule=get_pruning_schedule_config(_sparsity),
      pruning_granularity=get_pruning_granularity_config(_sparsity),
    )

  model_pruning_config = pruning_base_configs.ModelPruningConfig(
      model_name=model_name,
      pruning=[]
  )

  for layer_name, _sparsity in sparsity.items():
    layer_pruning_config = pruning_base_configs.LayerPruningConfig(
        layer_name=layer_name,
        pruning = [
            pruning_base_configs.WeightPruningConfig(
                weight_name='kernel',
                pruning=get_pruning_config(_sparsity),
            )
        ]
    )
    model_pruning_config.pruning.append(layer_pruning_config)

  if granularity == 'ChannelPruning':
    if model_name.startswith('resnet'):
      model_pruning_config.share_mask = _get_resnet_share_mask(model_name)

  if path:
    params_dict.save_params_dict_to_yaml(model_pruning_config, path)

  return model_pruning_config
