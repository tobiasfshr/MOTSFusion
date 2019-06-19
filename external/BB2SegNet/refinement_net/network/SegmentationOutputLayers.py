import tensorflow as tf

from external.BB2SegNet.refinement_net.network.Layer import Layer
from external.BB2SegNet.refinement_net.network.Util import prepare_input, get_activation, apply_dropout, conv2d, conv2d_dilated, bootstrapped_ce_loss, \
  class_balanced_ce_loss
from external.BB2SegNet.refinement_net.datasets import DataKeys
from external.BB2SegNet.refinement_net.core.Measures import compute_measures_for_binary_segmentation_tf
from external.BB2SegNet.refinement_net.core import Measures, Extractions
from external.BB2SegNet.refinement_net.core.Log import log
from external.BB2SegNet.refinement_net.core.Util import smart_shape

VOID_LABEL = 255


class SegmentationSoftmax(Layer):
  output_layer = True

  def __init__(self, name, inputs, dataset, network_input_dict, tower_setup, resize_targets=False,
               resize_logits=False, loss="ce", fraction=None):
    super().__init__()
    self.n_classes = dataset.num_classes()
    targets = network_input_dict[DataKeys.SEGMENTATION_LABELS]
    assert targets.get_shape().ndims == 4, targets.get_shape()
    # only one of these options can be true
    assert not (resize_targets and resize_logits)

    assert len(inputs) == 1, len(inputs)
    logits = inputs[0]
    assert logits.get_shape()[-1] == self.n_classes

    if resize_targets:
      print("warning, using resize_targets=True, so the resulting scores will not be computed at the initial "
            "resolution", file=log.v1)
      targets = tf.image.resize_nearest_neighbor(targets, tf.shape(logits)[1:3])
    if resize_logits:
      logits = tf.image.resize_images(logits, tf.shape(targets)[1:3])

    output = tf.nn.softmax(logits, -1, 'softmax')
    self.outputs = [output]
    if self.n_classes == 2:
      # the foreground prediction is sufficient for binary segmentation
      self.extractions[Extractions.SEGMENTATION_POSTERIORS] = output[..., 1]

    class_pred = tf.argmax(logits, axis=3)
    targets = tf.cast(targets, tf.int64)
    targets = tf.squeeze(targets, axis=3)
    self.loss = self._create_loss(loss, fraction, logits, targets)
    self.losses.append(self.loss)

    batch_size = smart_shape(targets)[0]
    if (not tower_setup.is_training) and batch_size == 1 and DataKeys.SEGMENTATION_LABELS_ORIGINAL_SIZE in network_input_dict:
      print(tower_setup.network_name, name, ": Using SEGMENTATION_LABELS_ORIGINAL_SIZE for calculating IoU")
      targets_for_measures = network_input_dict[DataKeys.SEGMENTATION_LABELS_ORIGINAL_SIZE]
      targets_for_measures = tf.cast(targets_for_measures, tf.int64)
      targets_for_measures = tf.squeeze(targets_for_measures, axis=3)
      class_pred_for_measures = self._resize_predictions_to_original_size(class_pred, network_input_dict,
                                                                          targets_for_measures, bilinear=False)
      self.extractions[Extractions.SEGMENTATION_MASK_ORIGINAL_SIZE] = class_pred_for_measures
      posteriors_orig_size = self._resize_predictions_to_original_size(output[..., 1], network_input_dict,
                                                                       targets_for_measures, bilinear=True)
      self.extractions[Extractions.SEGMENTATION_POSTERIORS_ORIGINAL_SIZE] = posteriors_orig_size
    else:
      print(tower_setup.network_name, name, ": Using SEGMENTATION_LABELS for calculating IoU", file=log.v1)
      targets_for_measures = targets
      class_pred_for_measures = class_pred
      self.extractions[Extractions.SEGMENTATION_MASK_INPUT_SIZE] = class_pred_for_measures

    self.measures = self._create_measures(class_pred_for_measures, targets_for_measures)
    self.add_image_summary(tf.cast(tf.expand_dims(class_pred, axis=3), tf.float32), "predicted labels")
    self.add_scalar_summary(self.loss, "loss")

  def _create_loss(self, loss_str, fraction, logits, targets):
    raw_ce = None
    n_valid_pixels_per_im = None
    if "ce" in loss_str:
      # we need to replace the void label to avoid nan
      no_void_label_mask = tf.not_equal(targets, VOID_LABEL)
      targets_no_void = tf.where(no_void_label_mask, targets, tf.zeros_like(targets))
      raw_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets_no_void, name="ce")
      # set the loss to 0 for the void label pixels
      raw_ce *= tf.cast(no_void_label_mask, tf.float32)
      n_valid_pixels_per_im = tf.reduce_sum(tf.cast(no_void_label_mask, tf.int32), axis=[1, 2])

    if loss_str == "ce":
      ce_per_im = tf.reduce_sum(raw_ce, axis=[1, 2])
      ce_per_im /= tf.cast(tf.maximum(n_valid_pixels_per_im, 1), tf.float32)
      ce_total = tf.reduce_mean(ce_per_im, axis=0)
      loss = ce_total
    elif loss_str == "bootstrapped_ce":
      loss = bootstrapped_ce_loss(raw_ce, fraction, n_valid_pixels_per_im)
    elif loss_str == "class_balanced_ce":
      loss = class_balanced_ce_loss(raw_ce, targets, self.n_classes)
    else:
      assert False, ("unknown loss", loss_str)
    return loss

  def _create_measures(self, pred, targets):
    n_examples = tf.shape(targets)[0]
    measures = {Measures.LOSS: self.loss * tf.cast(n_examples, tf.float32), Measures.N_EXAMPLES: n_examples}
    if self.n_classes == 2:
      binary_measures = compute_measures_for_binary_segmentation_tf(pred, targets)
      measures.update(binary_measures)
    return measures

  @staticmethod
  def _resize_predictions_to_original_size(class_pred, network_input_dict, targets_for_measures, bilinear):

    # here we assume that the inputs are optionally first cropped and then resized
    if DataKeys.CROP_BOXES_y0x0y1x1 in network_input_dict:
      crop_box = tf.squeeze(network_input_dict[DataKeys.CROP_BOXES_y0x0y1x1], axis=0)
      y0, x0, y1, x1 = tf.unstack(crop_box)
      height_before_resize = y1 - y0
      width_before_resize = x1 - x0
    else:
      height_before_resize, width_before_resize = tf.shape(targets_for_measures)[1:3]
      y0, x0, y1, x1 = None, None, None, None

    # resize to original size (might be the crop size)
    if bilinear:
      class_pred_original_size = tf.squeeze(tf.image.resize_bilinear(class_pred[..., tf.newaxis],
                                                                     [height_before_resize,width_before_resize]), axis=-1)
    else:
      class_pred_original_size = tf.squeeze(tf.image.resize_nearest_neighbor(class_pred[..., tf.newaxis],
                                                                             [height_before_resize, width_before_resize]),
                                            axis=-1)
    # revert cropping (if applicable)
    if DataKeys.CROP_BOXES_y0x0y1x1 in network_input_dict:
      pad_y_l = y0
      pad_y_r = tf.shape(targets_for_measures)[1] - y1
      pad_x_l = x0
      pad_x_r = tf.shape(targets_for_measures)[2] - x1
      class_pred_for_measures = tf.pad(class_pred_original_size, [[0, 0], [pad_y_l, pad_y_r], [pad_x_l, pad_x_r]])
    else:
      class_pred_for_measures = class_pred_original_size
    return class_pred_for_measures
