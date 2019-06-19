import tensorflow as tf
import math
from abc import ABC

from external.BB2SegNet.refinement_net.datasets import DataKeys
from external.BB2SegNet.refinement_net.core.Log import log
from external.BB2SegNet.refinement_net.datasets.util.Util import flip_coords_horizontal


class Augmentor(ABC):
  def apply_before_resize(self, tensors):
    return tensors

  def apply_after_resize(self, tensors):
    return tensors


class GammaAugmentor(Augmentor):
  def __init__(self, gamma_range=(-0.1, 0.1)):
    self.gamma_range = gamma_range

  def apply_after_resize(self, tensors, factor=None):
    """
    Augments the images. Expects it to be in the [0, 1] range
    """
    with tf.name_scope('gamma_augmentor'):
      img = tensors[DataKeys.IMAGES]

      # Sample a gamma factor
      if factor is None:
        factor = tf.random_uniform(shape=[], minval=self.gamma_range[0], maxval=self.gamma_range[1], dtype=tf.float32)
      gamma = tf.log(0.5 + 1 / math.sqrt(2) * factor) / tf.log(0.5 - 1 / math.sqrt(2) * factor)

      # Perform the gamma correction
      aug_image = img ** gamma

      aug_tensors = tensors.copy()
      aug_tensors[DataKeys.IMAGES] = aug_image
    return aug_tensors


class FlipAugmentor(Augmentor):
  def __init__(self, p=0.5):
    """
    :param p: The probability that the image will be flipped.
    """
    self.p = p

  def apply_after_resize(self, tensors, doit=None):
    with tf.name_scope("flip_augmentor"):
      aug_tensors = tensors.copy()

      if doit is None:
        doit = tf.random_uniform([]) > self.p

      def maybe_flip(key_, image_flip):
        if key_ in tensors:
          val = tensors[key_]
          if image_flip:
            flipped = tf.image.flip_left_right(val)
          else:
            flipped = flip_coords_horizontal(val, tf.shape(tensors[DataKeys.IMAGES])[1])
          aug = tf.cond(doit, lambda: flipped, lambda: val)
          aug_tensors[key_] = aug

      keys_to_flip = [DataKeys.IMAGES, DataKeys.SEGMENTATION_LABELS, DataKeys.BBOX_GUIDANCE,
                      DataKeys.SEGMENTATION_LABELS_ORIGINAL_SIZE, DataKeys.LASER_GUIDANCE, DataKeys.SEGMENTATION_MASK]
      coords_to_flip = [DataKeys.BBOXES_y0x0y1x1]
      for key in keys_to_flip:
        maybe_flip(key, image_flip=True)
      for key in coords_to_flip:
        maybe_flip(key, image_flip=False)

      return aug_tensors


class BBoxJitterAugmentor(Augmentor):
  def __init__(self, v=0.15):
    self.v = v

  def apply_before_resize(self, tensors):
    if DataKeys.BBOXES_y0x0y1x1 in tensors:
      y0, x0, y1, x1 = tf.unstack(tf.cast(tensors[DataKeys.BBOXES_y0x0y1x1], tf.float32))
      g = tf.random_normal((4,))
      # avoid outliers by clipping. Otherwise the resulting bounding box might even become empty
      g = tf.clip_by_value(g, -2.5, 2.5)
      h = y1 - y0
      w = x1 - x0
      y0 += self.v * g[0] * h
      x0 += self.v * g[1] * w
      y1 += self.v * g[2] * h
      x1 += self.v * g[3] * w

      # clip to image size
      shape = tf.shape(tensors[DataKeys.IMAGES])
      y0 = tf.maximum(y0, 0)
      x0 = tf.maximum(x0, 0)
      y1 = tf.minimum(y1, tf.cast(shape[0], tf.float32))
      x1 = tf.minimum(x1, tf.cast(shape[1], tf.float32))

      bbox_jittered = tf.stack([y0, x0, y1, x1])
      tensors[DataKeys.BBOXES_y0x0y1x1] = bbox_jittered
    return tensors


def parse_augmentors(strs, config):
  augmentors = []
  for s in strs:
    if s == "gamma":
      augmentor = GammaAugmentor(gamma_range=(-0.05, 0.05))
    elif s == "flip":
      augmentor = FlipAugmentor()
    elif s == "bbox_jitter":
      v = config.float("bbox_jitter_factor", 0.15)
      print("using bbox_jitter_factor=", v, file=log.v5, sep="")
      augmentor = BBoxJitterAugmentor(v)
    else:
      assert False, "unknown augmentor" + s
    augmentors.append(augmentor)
  return augmentors
