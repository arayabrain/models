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
import yaml

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
        count = 0
        for layer_name in all_layer_names:
          if re.search(_layer_name, layer_name):
            layer = model.get_layer(layer_name)
            layer_pruning_config = _expand_layer_pruning_config(layer, layer_pruning_config)
            layer_pruning_config.layer_name = layer_name
            _config_dict[layer_name] = layer_pruning_config
            count += 1
        if not count:
          raise ValueError('The specified layer name {} does not exist, so we '
                           'tried to interpret it as a regexp search pattern. '
                           'However, `re.search` did not hit any existing layer '
                           'names.'.format(_layer_name))
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
      'TwoOutOfFour': pruning_granu.TwoOutOfFour,
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
          weights_constraints_map = dict(
            kernel='kernel_constraint',
            depthwise_kernel='depthwise_constraint',
          )
          constraint_name = weights_constraints_map[weight_name]
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
    stage_block_pairs = [(2, 3), (3, 4), (4, 6), (5, 3)]
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
  return share_mask


def generate_pruning_config(model_name,
                            sparsity,
                            begin_step=0,
                            end_step=-1,
                            schedule='ConstantSparsity',
                            granularity='BlockSparsity',
                            respect_submatrix=False,
                            two_over_four_chin=False,
                            ch_share=True,
                            path=None):
  """Generate a model pruning config out of sparsity configuration.

  Arguments:
    model_name: A `str`. 'mnist', 'resnet56' (CIFAR-10), 'resnet50' (ImageNet),
      or 'mobilenetV1'.
    sparsity: A `dict`. Keys are `str` representing layer names (or possibly a
      regexp pattern), and values are sparsity (must be convertible to float).
    begin_step: Step at which to begin pruning. `0` by default.
    end_step:  Step at which to end pruning. `-1` by default. `-1` implies
        continuing to prune till the end of training (available only for
        'ConstantSparsity' schedule).
    schedule: 'ConstantSparsity' or 'PolynomialDecay'.
    granularity: 'ArayaMag', 'BlockSparsity', 'ChannelPruning', 'KernelLevel',
      or 'QuasiCyclic'.
    respect_submatrix: A `bool`. Whether or not to mask weight tensors
      submatrix-wise.
    two_over_four_chin: A `bool`. Whether or not to realize two-out-of-four
      sparsity pattern along input channels. Defaults to `False`, in which case
      the sparsity pattern is achieved along the output channels.
    ch_share: A `bool`. Whether or not to share masks ac
    path: `None` or a `str`. If `str`, saves the model pruning config as YAML
      file.

  Returns:
    A ModelPruningConfig instance.
  """

  def get_pruning_schedule_config(_sparsity):
    _sparsity = float(_sparsity)
    config = dict(begin_step=begin_step, end_step=end_step, frequency=100)
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
      config['gamma'] = int(1/(1.0 - _sparsity))
      if respect_submatrix:
        config['respect_submatrix'] = True
    elif granularity == 'BlockSparsity':
      config['block_size'] = [1, 1]
      config['block_pooling_type'] = 'AVG'
    elif granularity == 'ChannelPruning':
      config['ch_axis'] = -1
    elif granularity == 'KernelLevel':
      config['ker_axis'] = [0, 1]
    elif granularity == 'TwoOutOfFour':
      block_axis = -2 if two_over_four_chin else -1
      config['block_axis'] = block_axis
      if respect_submatrix:
        config['respect_submatrix'] = True
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

  if granularity == 'ChannelPruning' and ch_share:
    if model_name.startswith('resnet'):
      model_pruning_config.share_mask = _get_resnet_share_mask(model_name)

  if path:
    def save_params_dict_to_yaml(params, file_path):
      """Saves the input ParamsDict to a YAML file.

      Taken from params_dict.save_params_dict_to_yaml.
      """
      with tf.io.gfile.GFile(file_path, 'w') as f:
        #def _my_list_rep(dumper, data):
        #  # u'tag:yaml.org,2002:seq' is the YAML internal tag for sequence.
        #  return dumper.represent_sequence(
        #    u'tag:yaml.org,2002:seq', data, flow_style=True)
        #
        #yaml.add_representer(list, _my_list_rep)
        yaml.dump(params.as_dict(), f, default_flow_style=False)
    save_params_dict_to_yaml(model_pruning_config, path)

  return model_pruning_config


def generate_sensitivity_config(model_name,
                                layer_name,
                                weight_name,
                                granularity='BlockSparsity',
                                gamma=2,
                                respect_submatrix=False,
                                two_over_four_chin=False):
  """Generate a model pruning config for pruning sensitivity analysis."""

  def get_pruning_schedule_config():
    config = dict(
        initial_sparsity=0.0,
        final_sparsity=15/16,
        power=1,
        begin_step=0,
        end_step=15,
        frequency=1,
    )

    return pruning_base_configs.PruningScheduleConfig(
        class_name='PolynomialDecay',
        config=config,
    )

  def get_pruning_granularity_config():
    config = dict()
    if granularity in ('ArayaMag', 'QuasiCyclic'):
      config['gamma'] = gamma
      if respect_submatrix:
        config['respect_submatrix'] = True
    elif granularity == 'BlockSparsity':
      config['block_size'] = (1, 1)
      config['block_pooling_type'] = 'AVG'
    elif granularity == 'ChannelPruning':
      config['ch_axis'] = -1
    elif granularity == 'KernelLevel':
      config['ker_axis'] = (0, 1)
    elif granularity == 'TwoOutOfFour':
      block_axis = -2 if two_over_four_chin else -1
      config['block_axis'] = block_axis
      if respect_submatrix:
        config['respect_submatrix'] = True
    else:
      raise ValueError
    return pruning_base_configs.PruningGranularityConfig(
      class_name=granularity,
      config=config,
    )

  def get_pruning_config():
    return pruning_base_configs.PruningConfig(
      pruning_schedule=get_pruning_schedule_config(),
      pruning_granularity=get_pruning_granularity_config(),
    )

  model_pruning_config = pruning_base_configs.ModelPruningConfig(
      model_name=model_name,
      pruning=[]
  )

  layer_pruning_config = pruning_base_configs.LayerPruningConfig(
    layer_name=layer_name,
    pruning=[
      pruning_base_configs.WeightPruningConfig(
        weight_name=weight_name,
        pruning=get_pruning_config(),
      )
    ]
  )
  model_pruning_config.pruning.append(layer_pruning_config)

  return model_pruning_config


def _get_nonvanishing_channels(weights, ch_axis=-1):
  """Returns a list of indices along ch_axis with non-vanishing weights.

  Arguments:
    weights: A tensor, e.g. the kernel of a Conv2D layer.
    ch_axis: An int which specifies the axis for channel pruning.

  Returns:
    indices: A integer vector which specifies the non-vanishing channels.
  """
  weights_shape = weights.shape.as_list()
  norm_axis = list(range(len(weights_shape)))
  norm_axis.pop(ch_axis)
  saliences = math_ops.reduce_sum(tf.abs(weights), axis=norm_axis)
  where = tf.greater(saliences, 0)
  return tf.reshape(tf.where(where), [-1])


def _get_chin_controller(model):
  chin_controller = dict()

  def get_layer_names():
    return [layer.name for layer in model.layers if type(layer) in (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.Dense,
        tf.keras.layers.BatchNormalization)]

  if model.name == 'mnist':
    chin_controller['conv2d'] = None
    chin_controller['conv2d_1'] = 'conv2d'
    chin_controller['dense'] = 'conv2d_1'
    chin_controller['dense_1'] = 'dense'

  elif model.name == 'resnet56':
    for layer_name in get_layer_names():
      if layer_name == 'conv1':
        prev_layer_name = None
      elif layer_name == 'bn_conv1':
        prev_layer_name = 'conv1'
      elif layer_name.startswith('bn'):
        prev_layer_name = 'res' + layer_name[2:]
      elif layer_name == 'fc10':
        prev_layer_name = 'res4block_8_branch2b'
      else:
        words = layer_name.split('_')
        assert len(words) == 3
        if words[2] == 'branch2b':
          words[2] = 'branch2a'
        elif words[1] == '0':
          if words[0] == 'res2block':
            words = ['conv1']
          elif words[0] == 'res3block':
            words = ['res2block', '8', 'branch2b']
          elif words[0] == 'res4block':
            words = ['res3block', '8', 'branch2b']
        else:
          words[1] = str(int(words[1]) - 1)
          words[2] = 'branch2b'
        prev_layer_name = '_'.join(words)
      chin_controller[layer_name] = prev_layer_name

  elif model.name == 'resnet50':
    for layer_name in get_layer_names():
      if layer_name == 'conv1':
        prev_layer_name = None
      elif layer_name == 'bn_conv1':
        prev_layer_name = 'conv1'
      elif layer_name.startswith('bn'):
        prev_layer_name = 'res' + layer_name[2:]
      elif layer_name == 'fc1000':
        prev_layer_name = 'res5c_branch2c'
      else:
        words = layer_name.split('_')
        assert len(words) == 2
        if words[1] == 'branch2c':
          words[1] = 'branch2b'
        elif words[1] == 'branch2b':
          words[1] = 'branch2a'
        elif words[0][-1] == 'a':
          if words[0][-2] == '2':
            words = ['conv1']
          elif words[0][-2] == '3':
            words = ['res2c_branch2c']
          elif words[0][-2] == '4':
            words = ['res3d_branch2c']
          elif words[0][-2] == '5':
            words = ['res4f_branch2c']
        else:
          prev_chr = chr(ord(words[0][-1]) - 1)
          words[0] = words[0][:-1] + prev_chr
          words[1] = 'branch2c'
        prev_layer_name = '_'.join(words)
      chin_controller[layer_name] = prev_layer_name

  elif model.name == 'mobilenetV1':
    for layer_name in get_layer_names():
      if layer_name == 'Conv2d_0':
        prev_layer_name = None
      elif layer_name == 'Conv2d_1_depthwise':
        prev_layer_name = 'Conv2d_0'
      elif layer_name == 'Conv2d_1c_1x1':
        prev_layer_name = 'Conv2d_13_pointwise'
      elif layer_name.endswith('/BN'):
        prev_layer_name = layer_name[:-3]
      else:
        words = layer_name.split('_')
        assert len(words) == 3
        if words[2] == 'depthwise':
          words[1] = str(int(words[1]) - 1)
          words[2] = 'pointwise'
        elif words[2] == 'pointwise':
          words[2] = 'depthwise'
        else:
          raise ValueError
        prev_layer_name = '_'.join(words)

      if prev_layer_name:
        if prev_layer_name.endswith('depthwise'):
          prev_layer_name = chin_controller[prev_layer_name]

      chin_controller[layer_name] = prev_layer_name

  else:
    raise ValueError('Unknown model name: {}'.format(model.name))

  return chin_controller


def prune_physically(model):
  """Physically reduces output channels of Conv2D layers after channel pruning.

  Arguments:
    model: A Keras model.

  Returns:
    new_model: Another model with reduced number of output channels.
  """
  chout_axis = -1
  chin_axis = -2

  def clone_function(layer):
    layer_config = layer.get_config()
    if type(layer) is tf.keras.layers.Conv2D:
      layer_config['filters'] = len(_get_nonvanishing_channels(layer.kernel))
    elif type(layer) is tf.keras.layers.Dense:
      layer_config['units'] = len(_get_nonvanishing_channels(layer.kernel))
    return layer.__class__.from_config(layer_config)

  # Defines the new model architecture.
  new_model = tf.keras.models.clone_model(
      model, input_tensors=None, clone_function=clone_function)

  # Copy relevant weights from the original.
  assert len(model.layers) == len(new_model.layers)
  chin_controller = _get_chin_controller(model)
  for layer, new_layer in zip(model.layers, new_model.layers):
    weights = layer.get_weights()
    assert type(layer) is type(new_layer)
    if type(layer) in (tf.keras.layers.Conv2D,
                       tf.keras.layers.DepthwiseConv2D,
                       tf.keras.layers.Dense):
      kernel = layer.depthwise_kernel if type(layer) is tf.keras.layers.DepthwiseConv2D else layer.kernel

      # Gather slices along the output channel dimension.
      chout_indices = _get_nonvanishing_channels(kernel, ch_axis=chout_axis)
      kernel = tf.gather(kernel, indices=chout_indices, axis=chout_axis)

      # Gather slices along the input channel dimension.
      prev_layer_name = chin_controller[layer.name]
      if prev_layer_name is not None:
        prev_layer = model.get_layer(prev_layer_name)
        assert prev_layer.bias is None
        prev_layer_kernel = prev_layer.depthwise_kernel \
            if type(prev_layer) is tf.keras.layers.DepthwiseConv2D else prev_layer.kernel
        chin_indices = _get_nonvanishing_channels(prev_layer_kernel, ch_axis=chout_axis)

        if type(prev_layer) is tf.keras.layers.Conv2D:
          if type(layer) is tf.keras.layers.Dense:
            num_repeats, remainder = divmod(layer.input_shape[-1], prev_layer.filters)
            if remainder:
              raise ValueError
            if tf.keras.backend.image_data_format() == 'channels_first':
              staircase = tf.repeat(chin_indices * num_repeats, repeats=num_repeats)
              wave = tf.tile(tf.range(num_repeats), multiples=[len(chin_indices)] * num_repeats)
              chin_indices = staircase + tf.cast(wave, tf.int64)
            else:
              tensor_list = [chin_indices + i * prev_layer.filters for i in range(num_repeats)]
              chin_indices = tf.concat(tensor_list, axis=0)

        kernel = tf.gather(kernel, indices=chin_indices, axis=chin_axis)
      weights[0] = kernel
    elif type(layer) is tf.keras.layers.BatchNormalization:
      prev_layer_name = chin_controller[layer.name]
      if prev_layer_name is not None:
        prev_layer = model.get_layer(prev_layer_name)
        assert prev_layer.bias is None
        chout_indices = _get_nonvanishing_channels(prev_layer.kernel, ch_axis=chout_axis)
        for i, weight in enumerate(weights):
          weights[i] = tf.gather(weight, indices=chout_indices, axis=chout_axis)

    new_layer.set_weights(weights)

  return new_model
