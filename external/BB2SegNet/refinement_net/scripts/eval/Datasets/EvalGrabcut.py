import random

import tensorflow as tf
from scipy.misc import imsave
import numpy as np

from refinement_net.core import Extractions
from refinement_net.datasets import DataKeys
from refinement_net.datasets.Grabcut.Grabcut import GrabcutDataset
from refinement_net.datasets.PascalVOC.PascalVOC_masked_dios import PREV_NEG_CLICKS, PREV_POS_CLICKS, PREV_MASK
from refinement_net.datasets.util import Util
from refinement_net.datasets.util.Normalization import unnormalize
from refinement_net.datasets.util.Util import visualise_clicks, encodeMask
from refinement_net.scripts.eval.Datasets.EvalPascalMasked import EvalPascalMaskedDataset


class EvalGrabcutDataset(GrabcutDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset)
    self.iterative_training = config.bool("iterative_training", True)
    self.eval_pascal_dataset = EvalPascalMaskedDataset(config, subset)
    self.previous_epoch_data = self.eval_pascal_dataset.previous_epoch_data
    self.save_images = config.bool("save_images", False)
    self.img_dir = config.string("img_dir", str(random.randrange(1, 10000)))

  def get_extraction_keys(self):
    return self.eval_pascal_dataset.get_extraction_keys()

  def postproc_example_before_assembly(self, tensors):
    return self.eval_pascal_dataset.postproc_example_before_assembly(tensors)

  def postproc_annotation(self, ann_filename, ann):
    mask = super().postproc_annotation(ann_filename, ann)
    return {DataKeys.SEGMENTATION_LABELS: mask, DataKeys.RAW_SEGMENTATION_LABELS: mask,
            DataKeys.IMAGE_FILENAMES: ann_filename}

  def create_summaries(self, data):
    if DataKeys.IMAGES in data:
      images = unnormalize(data[DataKeys.IMAGES])
      if DataKeys.NEG_CLICKS in data:
        images = tf.py_func(visualise_clicks, [images, data[DataKeys.NEG_CLICKS][:, :, :, 0:1], "r"], tf.float32)
        self.summaries.append(tf.summary.image(self.subset + "data/neg_clicks",
                                               tf.cast(data[DataKeys.NEG_CLICKS][:, :, :, 0:1], tf.float32)))
      if DataKeys.POS_CLICKS in data:
        images = tf.py_func(visualise_clicks, [images, data[DataKeys.POS_CLICKS][:, :, :, 0:1], "g"], tf.float32)
        self.summaries.append(tf.summary.image(self.subset + "data/pos_clicks",
                                               tf.cast(data[DataKeys.POS_CLICKS][:, :, :, 0:1], tf.float32)))
      self.summaries.append(tf.summary.image(self.subset + "data/images", images))

    if DataKeys.SEGMENTATION_LABELS in data:
      self.summaries.append(tf.summary.image(self.subset + "data/ground truth segmentation labels",
                                             tf.cast(data[DataKeys.SEGMENTATION_LABELS], tf.float32)))
    if DataKeys.BBOX_GUIDANCE in data:
      self.summaries.append(tf.summary.image(self.subset + "data/bbox guidance",
                                             tf.cast(data[DataKeys.BBOX_GUIDANCE], tf.float32)))
    if DataKeys.SIGNED_DISTANCE_TRANSFORM_GUIDANCE in data:
      self.summaries.append(tf.summary.image(self.subset + "data/signed_distance_transform_guidance",
                                             data[DataKeys.SIGNED_DISTANCE_TRANSFORM_GUIDANCE]))
    if DataKeys.UNSIGNED_DISTANCE_TRANSFORM_GUIDANCE in data:
      self.summaries.append(tf.summary.image(self.subset + "data/unsigned_distance_transform_guidance",
                                             data[DataKeys.UNSIGNED_DISTANCE_TRANSFORM_GUIDANCE]))

  def use_segmentation_mask(self, res):
    extractions = res[Extractions.EXTRACTIONS]

    # if self.subset == "train" and self.iterative_training:
    if self.iterative_training:
      assert DataKeys.IMAGE_FILENAMES in extractions
      assert Extractions.SEGMENTATION_MASK_INPUT_SIZE in extractions
      batch_size = len(extractions[DataKeys.IMAGE_FILENAMES][0])

      for id in range(batch_size):
        filename = extractions[DataKeys.IMAGE_FILENAMES][0][id]
        self.previous_epoch_data[filename][PREV_MASK] = \
          encodeMask(extractions[Extractions.SEGMENTATION_MASK_INPUT_SIZE][0][id])

    extractions = res[Extractions.EXTRACTIONS]
    filename = extractions[DataKeys.IMAGE_FILENAMES][0][0]
    inputs = extractions[DataKeys.INPUTS][0][0]
    prediction = extractions[Extractions.SEGMENTATION_MASK_INPUT_SIZE][0][0]
    prev_neg_clicks = self.previous_epoch_data[filename][PREV_NEG_CLICKS]
    prev_pos_clicks = self.previous_epoch_data[filename][PREV_POS_CLICKS]

    if self.save_images:
      main_folder = "scripts/eval/" + self.img_dir + "/"
      tf.gfile.MakeDirs(main_folder)
      num_clicks = len(prev_neg_clicks) + \
                   len(prev_pos_clicks)
      filename = filename.decode('UTF-8')
      fn = main_folder + filename.split("/")[-1].split(".")[0] + "_" + str(num_clicks) + ".png"
      img = unnormalize(inputs[:, :, :3])
      img = Util.get_masked_image(img, prediction)
      click_map = np.ones_like(inputs[:, :, 3:4])
      previous_clicks = [a for a in zip(*(val for val in prev_neg_clicks))]
      click_map[previous_clicks] = 0
      img = visualise_clicks([img], np.expand_dims(click_map, axis=0), 'r')

      click_map = np.ones_like(inputs[:, :, 3:4])
      previous_clicks = [a for a in zip(*(val for val in prev_pos_clicks))]
      click_map[previous_clicks] = 0
      img = visualise_clicks(img, np.expand_dims(click_map, axis=0), 'g')
      imsave(fn, img[0])