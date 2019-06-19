from enum import Enum, unique
from functools import partial
import tensorflow as tf
import numpy as np

from external.BB2SegNet.refinement_net.datasets import DataKeys
from external.BB2SegNet.refinement_net.datasets.util.Util import resize_image, random_crop_image, resize_coords
from external.BB2SegNet.refinement_net.core.Log import log


@unique
class ResizeMode(Enum):
  FIXED_SIZE = "fixed_size"
  UNCHANGED = "unchanged"
  RANDOM_RESIZE_AND_CROP = "random_resize_and_crop"
  BBOX_CROP_AND_RESIZE_FIXED_SIZE = "bbox_crop_and_resize_fixed_size"
  RESIZE_MIN_SHORT_EDGE_MAX_LONG_EDGE = "resize_min_short_edge_max_long_edge"
  FIXED_RESIZE_AND_CROP = "fixed_resize_and_crop"


def resize(tensors, resize_mode, size):
  if resize_mode == ResizeMode.FIXED_SIZE:
    return resize_fixed_size(tensors, size)
  elif resize_mode == ResizeMode.UNCHANGED:
    return resize_unchanged(tensors)
  elif resize_mode == ResizeMode.RANDOM_RESIZE_AND_CROP:
    return random_resize_and_crop(tensors, size)
  elif resize_mode == ResizeMode.FIXED_RESIZE_AND_CROP:
    return fixed_resize_and_crop(tensors, size)
  elif resize_mode == ResizeMode.BBOX_CROP_AND_RESIZE_FIXED_SIZE:
    return bbox_crop_and_resize_fixed_size(tensors, size)
  elif resize_mode == ResizeMode.RESIZE_MIN_SHORT_EDGE_MAX_LONG_EDGE:
    return resize_min_short_edge_max_long_edge(tensors, size)
  else:
    assert False, ("resize mode not implemented yet", resize_mode)


def resize_fixed_size(tensors, size):
  tensors_resized = tensors.copy()

  class Resizing_:
    Bilinear, NN, Coords = range(3)

  def resize_(key_, resizing_):
    if key_ in tensors:
      val = tensors[key_]
      if resizing_ == Resizing_.Bilinear:
        resized = resize_image(val, size, True)
      elif resizing_ == Resizing_.NN:
        resized = resize_image(val, size, False)
      elif resizing_ == Resizing_.Coords:
        resized = resize_coords(val, size, tf.shape(tensors[DataKeys.IMAGES])[:2])
      tensors_resized[key_] = resized

  keys_to_resize_bilinear = [DataKeys.IMAGES]
  keys_to_resize_nn = [DataKeys.SEGMENTATION_LABELS, DataKeys.BBOX_GUIDANCE, DataKeys.RAW_SEGMENTATION_LABELS, DataKeys.SEGMENTATION_MASK]
  keys_to_resize_coords = [DataKeys.BBOXES_y0x0y1x1]
  for key in keys_to_resize_bilinear:
    resize_(key, Resizing_.Bilinear)
  for key in keys_to_resize_nn:
    resize_(key, Resizing_.NN)
  for key in keys_to_resize_coords:
    resize_(key, Resizing_.Coords)
  return tensors_resized


def resize_min_short_edge_max_long_edge(tensors, size):
  # Reference: CustomResize._get_augment_params in
  # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/FasterRCNN/common.py
  assert len(size) == 2
  min_short_edge = float(size[0])
  max_long_edge = float(size[1])
  h = tf.cast(tf.shape(tensors[DataKeys.IMAGES])[0], tf.float32)
  w = tf.cast(tf.shape(tensors[DataKeys.IMAGES])[1], tf.float32)

  scale = min_short_edge * 1.0 / tf.minimum(h, w)
  less_val = tf.less(h, w)
  newh = tf.where(less_val, min_short_edge, tf.multiply(scale, h))
  neww = tf.where(less_val, tf.multiply(scale, w), min_short_edge)

  scale = max_long_edge * 1.0 / tf.maximum(newh, neww)
  greater_val = tf.greater(tf.maximum(newh, neww), max_long_edge)
  newh = tf.where(greater_val, tf.multiply(scale, newh), newh)
  neww = tf.where(greater_val, tf.multiply(scale, neww), neww)

  newh = tf.cast(tf.round(newh), tf.int32)
  neww = tf.cast(tf.round(neww), tf.int32)

  return resize_fixed_size(tensors, tf.stack([newh, neww], axis=0))


def resize_unchanged(tensors):
  return tensors


def random_resize_and_crop(tensors, size):
  assert len(size) in (1, 2)
  if len(size) == 2:
    assert size[0] == size[1]
    crop_size = size
  else:
    crop_size = [size, size]
  tensors = resize_random_scale_with_min_size(tensors, min_size=crop_size)
  tensors = random_crop_tensors(tensors, crop_size)
  return tensors


def fixed_resize_and_crop(tensors, size):
  assert len(size) in (1, 2)
  if len(size) == 2:
    assert size[0] == size[1]
    crop_size = size
  else:
    crop_size = [size, size]
  tensors = scale_with_min_size(tensors, min_size=crop_size)
  tensors = object_crop_fixed_offset(tensors, crop_size)

  return tensors


def random_crop_tensors(tensors, size, offset=None):
  tensors_cropped = tensors.copy()

  keys_to_crop = [DataKeys.IMAGES, DataKeys.SEGMENTATION_LABELS, DataKeys.SEGMENTATION_LABELS_ORIGINAL_SIZE,
                  DataKeys.BBOX_GUIDANCE, DataKeys.RAW_SEGMENTATION_LABELS]
  # offset = None

  def _crop(key_, offset_):
    if key_ in tensors:
      val = tensors[key_]
      cropped, offset_ = random_crop_image(val, size, offset_)
      tensors_cropped[key_] = cropped
    return offset_

  for key in keys_to_crop:
    offset = _crop(key, offset)
  return tensors_cropped


def object_crop_fixed_offset(tensors, size):
  label = tensors[DataKeys.SEGMENTATION_LABELS]
  object_locations = tf.cast(tf.where(tf.not_equal(label, 0))[:, :2], tf.int32)
  min_val = tf.maximum(tf.constant([0, 0]),
                       tf.reduce_max(object_locations, axis=0) - size)
  offset = tf.concat([min_val, [0]], axis=0)

  return random_crop_tensors(tensors, size, offset)


def bbox_crop_and_resize_fixed_size(tensors, size):
  MARGIN = 50
  tensors_cropped = tensors.copy()

  assert DataKeys.BBOXES_y0x0y1x1 in tensors
  bbox = tensors[DataKeys.BBOXES_y0x0y1x1]
  bbox_rounded = tf.cast(tf.round(bbox), tf.int32)
  y0, x0, y1, x1 = tf.unstack(bbox_rounded)

  # add margin and clip to bounds
  shape = tf.shape(tensors[DataKeys.IMAGES])
  y0 = tf.maximum(y0 - MARGIN, 0)
  x0 = tf.maximum(x0 - MARGIN, 0)
  y1 = tf.minimum(y1 + MARGIN, shape[0])
  x1 = tf.minimum(x1 + MARGIN, shape[1])

  def crop_and_resize(key_, bilinear):
    if key_ in tensors:
      val = tensors[key_]
      res = val[y0:y1, x0:x1]
      res = resize_image(res, size, bilinear)
      tensors_cropped[key_] = res

  keys_to_resize_bilinear = [DataKeys.IMAGES]
  keys_to_resize_nn = [DataKeys.SEGMENTATION_LABELS, DataKeys.BBOX_GUIDANCE, DataKeys.RAW_SEGMENTATION_LABELS]

  for key in keys_to_resize_bilinear:
    crop_and_resize(key, True)

  for key in keys_to_resize_nn:
    crop_and_resize(key, False)

  if DataKeys.LASER_GUIDANCE in tensors:
    laser = tensors[DataKeys.LASER_GUIDANCE][y0: y1, x0: x1]
    laser = resize_laser_to_fixed_size(laser, size)
    tensors_cropped[DataKeys.LASER_GUIDANCE] = laser

  if DataKeys.SEGMENTATION_LABELS in tensors:
    tensors_cropped[DataKeys.SEGMENTATION_LABELS_ORIGINAL_SIZE] = \
      tensors[DataKeys.SEGMENTATION_LABELS]

  tensors_cropped[DataKeys.CROP_BOXES_y0x0y1x1] = tf.stack([y0, x0, y1, x1])

  return tensors_cropped


def resize_random_scale_with_min_size(tensors, min_size, min_scale=0.7, max_scale=1.3):
  assert min_size is not None
  img = tensors[DataKeys.IMAGES]

  h = tf.shape(img)[0]
  w = tf.shape(img)[1]
  shorter_side = tf.minimum(h, w)
  min_scale_factor = tf.cast(min_size, tf.float32) / tf.cast(shorter_side, tf.float32)
  min_scale = tf.maximum(min_scale, min_scale_factor)
  max_scale = tf.maximum(max_scale, min_scale_factor)
  scale_factor = tf.random_uniform(shape=[], minval=min_scale, maxval=max_scale, dtype=tf.float32)
  scaled_size = tf.cast(tf.round(tf.cast(tf.shape(img)[:2], tf.float32) * scale_factor), tf.int32)
  tensors_out = resize_fixed_size(tensors, scaled_size)
  return tensors_out


def scale_with_min_size(tensors, min_size, min_scale=0.7, max_scale=1.3):
  assert min_size is not None
  img = tensors[DataKeys.IMAGES]

  h = tf.shape(img)[0]
  w = tf.shape(img)[1]
  shorter_side = tf.minimum(h, w)
  min_scale_factor = tf.cast(min_size, tf.float32) / tf.cast(shorter_side, tf.float32)
  scaled_size = tf.cast(tf.round(tf.cast(tf.shape(img)[:2], tf.float32) * min_scale_factor), tf.int32)
  tensors_out = resize_fixed_size(tensors, scaled_size)
  return tensors_out


def resize_laser_to_fixed_size(laser, size):
  f = partial(resize_laser_to_fixed_size_np, size=size)
  laser = tf.py_func(f, [laser], tf.float32, name="resize_laser_to_fixed_size")
  laser.set_shape(size + [1])
  return laser


def resize_laser_to_fixed_size_np(laser, size):
  # here we assume, that 1 is used for foreground, -1 for background and 0 for "no reading"
  fg_y, fg_x, _ = (laser == 1).nonzero()
  bg_y, bg_x, _ = (laser == -1).nonzero()

  def scale_indices(ind, size_in, size_out):
    return (np.round(((ind + 0.5) * size_out / size_in) - 0.5)).astype(np.int)

  fg_y = scale_indices(fg_y, laser.shape[0], size[0])
  fg_x = scale_indices(fg_x, laser.shape[1], size[1])
  bg_y = scale_indices(bg_y, laser.shape[0], size[0])
  bg_x = scale_indices(bg_x, laser.shape[1], size[1])

  laser_out = np.zeros(size + [1], dtype=np.float32)
  laser_out[fg_y, fg_x, 0] = 1
  laser_out[bg_y, bg_x, 0] = -1

  return laser_out
