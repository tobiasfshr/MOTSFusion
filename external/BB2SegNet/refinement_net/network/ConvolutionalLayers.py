import tensorflow as tf
import numpy as np

from external.BB2SegNet.refinement_net.network.Layer import Layer
from external.BB2SegNet.refinement_net.network.Util import conv2d, conv2d_dilated, get_activation, prepare_input, apply_dropout, max_pool, conv2d_transpose


class Conv(Layer):
  output_layer = False

  def __init__(self, name, inputs, n_features, tower_setup, filter_size=(3, 3), old_order=False,
               strides=(1, 1), dilation=None, pool_size=(1, 1), pool_strides=None, activation="relu", dropout=0.0,
               batch_norm=False, bias=False, batch_norm_decay=Layer.BATCH_NORM_DECAY_DEFAULT, l2=Layer.L2_DEFAULT,
               padding="SAME"):
    super(Conv, self).__init__()
    # mind the order of dropout, conv, activation and batchnorm!
    # batchnorm -> activation -> dropout -> conv -> pool
    # if old_order: dropout -> conv -> batchnorm -> activation -> pool (used for example in tensorpack)
    curr, n_features_inp = prepare_input(inputs)
    filter_size = list(filter_size)
    strides = list(strides)
    pool_size = list(pool_size)
    if pool_strides is None:
      pool_strides = pool_size

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", filter_size + [n_features_inp, n_features], l2, tower_setup)
      b = None
      if bias:
        b = self.create_bias_variable("b", [n_features], tower_setup)

      if old_order:
        curr = apply_dropout(curr, dropout)
        if dilation is None:
          curr = conv2d(curr, W, strides, padding=padding)
        else:
          curr = conv2d_dilated(curr, W, dilation, padding=padding)
        if bias:
          curr += b
        if batch_norm:
          curr = self.create_and_apply_batch_norm(curr, n_features, batch_norm_decay, tower_setup)
        curr = get_activation(activation)(curr)
      else:
        if batch_norm:
          curr = self.create_and_apply_batch_norm(curr, n_features_inp, batch_norm_decay, tower_setup)
        curr = get_activation(activation)(curr)
        curr = apply_dropout(curr, dropout)
        if dilation is None:
          curr = conv2d(curr, W, strides, padding=padding)
        else:
          curr = conv2d_dilated(curr, W, dilation, padding=padding)
        if bias:
          curr += b

      if pool_size != [1, 1]:
        curr = max_pool(curr, pool_size, pool_strides)
    self.outputs = [curr]


class ConvTranspose(Layer):
  output_layer = False

  def __init__(self, name, inputs, n_features, tower_setup, filter_size=(3, 3),
               strides=(1, 1), activation="relu",  # dropout=0.0,
               batch_norm=False, bias=False, batch_norm_decay=Layer.BATCH_NORM_DECAY_DEFAULT, l2=Layer.L2_DEFAULT,
               padding="SAME"):
    super(ConvTranspose, self).__init__()
    # TODO this uses the tensorpack order of operations
    curr, n_features_inp = prepare_input(inputs)
    filter_size = list(filter_size)
    strides = list(strides)

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", filter_size + [n_features, n_features_inp], l2, tower_setup)
      b = None
      if bias:
        b = self.create_bias_variable("b", [n_features], tower_setup)

      # curr = apply_dropout(curr, dropout)
      curr = conv2d_transpose(curr, W, strides, padding=padding)
      if bias:
        curr = tf.nn.bias_add(curr, b)
      if batch_norm:
        curr = self.create_and_apply_batch_norm(curr, n_features, batch_norm_decay, tower_setup)
      curr = get_activation(activation)(curr)
    self.outputs = [curr]


class ConvForOutput(Layer):
  output_layer = False

  def __init__(self, name, inputs, dataset, n_features, tower_setup, filter_size=(1, 1),
               input_activation=None, dilation=None, l2=Layer.L2_DEFAULT, dropout=0.0):
    super().__init__()
    if n_features == -1:
      n_features = dataset.num_classes()
    filter_size = list(filter_size)
    inp, n_features_inp = prepare_input(inputs)
    if input_activation is not None:
      inp = get_activation(input_activation)(inp)
    inp = apply_dropout(inp, dropout)

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", filter_size + [n_features_inp, n_features], l2, tower_setup)
      b = self.create_bias_variable("b", [n_features], tower_setup)
      if dilation is None:
        output = conv2d(inp, W) + b
      else:
        output = conv2d_dilated(inp, W, dilation) + b
      self.outputs = [output]


class ResidualUnit(Layer):
  output_layer = False

  def __init__(self, name, inputs, tower_setup, n_convs=2, n_features=None, dilations=None, strides=None,
               filter_size=None, activation="relu", dropout=0.0, batch_norm_decay=Layer.BATCH_NORM_DECAY_DEFAULT,
               l2=Layer.L2_DEFAULT):
    super().__init__()
    curr, n_features_inp = prepare_input(inputs)
    res = curr
    assert n_convs >= 1, n_convs

    if dilations is not None:
      assert strides is None
    elif strides is None:
      strides = [[1, 1]] * n_convs
    if filter_size is None:
      filter_size = [[3, 3]] * n_convs
    if n_features is None:
      n_features = n_features_inp
    if not isinstance(n_features, list):
      n_features = [n_features] * n_convs

    with tf.variable_scope(name):
      curr = self.create_and_apply_batch_norm(curr, n_features_inp, batch_norm_decay, tower_setup, "bn0")
      curr = get_activation(activation)(curr)
      if tower_setup.is_training:
        curr = apply_dropout(curr, dropout)

      if strides is None:
        strides_res = [1, 1]
      else:
        strides_res = np.prod(strides, axis=0).tolist()
      if (n_features[-1] != n_features_inp) or (strides_res != [1, 1]):
        W0 = self.create_weight_variable("W0", [1, 1] + [n_features_inp, n_features[-1]], l2, tower_setup)
        if dilations is None:
          res = conv2d(curr, W0, strides_res)
        else:
          res = conv2d(curr, W0)

      W1 = self.create_weight_variable("W1", filter_size[0] + [n_features_inp, n_features[0]], l2, tower_setup)
      if dilations is None:
        curr = conv2d(curr, W1, strides[0])
      else:
        curr = conv2d_dilated(curr, W1, dilations[0])
      for idx in range(1, n_convs):
        curr = self.create_and_apply_batch_norm(curr, n_features[idx - 1], batch_norm_decay,
                                                tower_setup, "bn" + str(idx + 1))
        curr = get_activation(activation)(curr)
        Wi = self.create_weight_variable("W" + str(idx + 1), filter_size[idx] + [n_features[idx - 1], n_features[idx]],
                                         l2, tower_setup)
        if dilations is None:
          curr = conv2d(curr, Wi, strides[idx])
        else:
          curr = conv2d_dilated(curr, Wi, dilations[idx])

    curr += res
    self.outputs = [curr]


class Upsampling(Layer):
  def __init__(self, name, inputs, tower_setup, n_features, concat, activation="relu", filter_size=(3, 3),
               l2=Layer.L2_DEFAULT):
    super(Upsampling, self).__init__()
    filter_size = list(filter_size)
    assert isinstance(concat, list)
    assert len(concat) > 0
    curr, n_features_inp = prepare_input(inputs)
    concat_inp, n_features_concat = prepare_input(concat)

    curr = tf.image.resize_nearest_neighbor(curr, tf.shape(concat_inp)[1:3])
    curr = tf.concat([curr, concat_inp], axis=3)
    n_features_curr = n_features_inp + n_features_concat

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", filter_size + [n_features_curr, n_features], l2, tower_setup)
      b = self.create_bias_variable("b", [n_features], tower_setup)
      curr = conv2d(curr, W) + b
      curr = get_activation(activation)(curr)

    self.outputs = [curr]
