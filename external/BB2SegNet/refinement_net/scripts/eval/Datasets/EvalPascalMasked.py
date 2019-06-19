import random

import numpy as np
import skimage.measure as skimage_measure
import tensorflow as tf
from scipy.misc import imsave
from scipy.ndimage import distance_transform_edt

from refinement_net.core import Extractions
from refinement_net.core.Log import log
from refinement_net.datasets import DataKeys
from refinement_net.datasets.PascalVOC.PascalVOC_masked_dios import PascalVOCMaskedDiosDataset, PREV_MASK, PREV_NEG_CLICKS, \
  PREV_POS_CLICKS
from refinement_net.datasets.util import Util
from refinement_net.datasets.util.DistanceTransform import get_distance_transform
from refinement_net.datasets.util.Normalization import unnormalize
from refinement_net.datasets.util.Util import decodeMask, visualise_clicks

CURRENT_CLICK = "curr_click"


def generate_click_for_correction(label, prediction, previous_clicks, void_label, n_clicks=1, d_step=5, random_cluster=False):
  prediction = np.copy(prediction).astype(np.uint8)
  label = np.copy(label).astype(np.uint8)

  # Perform opening to separate clusters connected by small structures.
  valid_mask = label != void_label
  misclassified = np.where(label != prediction, 1, 0)
  misclassified *= valid_mask
  # opened = opening(misclassified, disk(2))
  misclassified = skimage_measure.label(misclassified, background=0)
  previous_clicks = [a for a in zip(*(val for val in previous_clicks))]
  misclassified[previous_clicks] = 0

  clicks = []
  clusters = np.setdiff1d(np.unique(misclassified), [0])
  if len(clusters) == 0:
    return clicks

  cluster_counts = np.delete(np.bincount(misclassified.flatten()), 0, axis=0)
  if random_cluster:
    print("Using mislabelled clusters sampled with probability corresponding to its size.", file=log.v1)
    largest_cluster = np.random.choice(cluster_counts, p=cluster_counts / np.sum(cluster_counts))
    largest_cluster, _ = np.where(cluster_counts == largest_cluster)
    largest_cluster += 1
  else:
    largest_cluster = np.argmax(np.delete(np.bincount(misclassified.flatten()), 0, axis=0)) + 1

  dt = np.where(misclassified == largest_cluster, 1, 0)
  dt=misclassified
  # Set the border pixels of the image to 0, so that the click is centred on the required mask.
  dt[[0, dt.shape[0] - 1], :] = 0
  dt[:, [0, dt.shape[1] - 1]] = 0

  dt = distance_transform_edt(dt)

  for i in range(n_clicks):
    row = None
    col = None

    if np.max(dt) > 0:
      # get points that are farthest from the object boundary.
      # farthest_pts = np.where(dt > np.max(dt) / 2.0)
      farthest_pts = np.where(dt == np.max(dt))
      farthest_pts = [x for x in zip(farthest_pts[0], farthest_pts[1])]
      # sample from the list since there could be more that one such points.
      row, col = random.sample(farthest_pts, 1)[0]
      x_min = max(0, row - d_step)
      x_max = min(row + d_step, dt.shape[0])
      y_min = max(0, col - d_step)
      y_max = min(col + d_step, dt.shape[1])
      dt[x_min:x_max, y_min:y_max] = 0

    if row is not None and col is not None:
      clicks.append((row, col))
      dt[row, col] = 0

  return clicks


class EvalPascalMaskedDataset(PascalVOCMaskedDiosDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset)
    self.use_gaussian = config.bool("use_gaussian", False)
    self.save_images = config.bool("save_images", False)
    self.img_dir = config.string("img_dir", str(random.randrange(1,10000)))
    self.random_cluster = config.bool("random_cluster", False)
    # self.dt_method = config.unicode(DataKeys.DT_METHOD, "edt")

  def dios_distance_transform(self, label, raw_label, ignore_classes, img_filenames):
    if len(label.shape) > 2:
      label = label.copy()
      label = label[:, :, 0]

    if img_filenames not in self.previous_epoch_data:
      print(img_filenames, "not in previous epoch data")
      self.previous_epoch_data[img_filenames] = {}
      self.previous_epoch_data[img_filenames][PREV_NEG_CLICKS] = []
      self.previous_epoch_data[img_filenames][PREV_POS_CLICKS] = []
      prediction = np.zeros_like(label)
      pos_clicks=[]
      neg_clicks=[]
    else:
      if PREV_MASK not in self.previous_epoch_data[img_filenames]:
        print("No previous mask",img_filenames, self.previous_epoch_data[img_filenames])
      prediction = decodeMask(self.previous_epoch_data[img_filenames][PREV_MASK])
      neg_clicks = self.previous_epoch_data[img_filenames][PREV_NEG_CLICKS]
      pos_clicks = self.previous_epoch_data[img_filenames][PREV_POS_CLICKS]
    clicks = generate_click_for_correction(label, prediction, neg_clicks + pos_clicks, void_label=255, random_cluster=self.random_cluster)
    self.previous_epoch_data[img_filenames][CURRENT_CLICK] = clicks

    if len(clicks) > 0:
      if label[clicks[0][0], clicks[0][1]] == 1:
        pos_clicks += clicks
      else:
        neg_clicks += clicks

    u0 = self.normalise(get_distance_transform(neg_clicks, label))
    u1 = self.normalise(get_distance_transform(pos_clicks, label))
    u0 = u0[:, :, np.newaxis].astype(np.float32)
    u1 = u1[:, :, np.newaxis].astype(np.float32)
    self.previous_epoch_data[img_filenames][PREV_NEG_CLICKS] = neg_clicks
    self.previous_epoch_data[img_filenames][PREV_POS_CLICKS] = pos_clicks

    # Sanity checks
    if np.any(label[np.where(u0[:, :, 0] == 0)]):
      print("Neg clicks on the object detected", label[np.where(u0[:, :, 0] == 0)], file=log.v1)
    if not np.all(label[np.where(u1[:, :, 0] == 0)]):
      print("Pos clicks on background detected", file=log.v1)

    num_clicks = len(pos_clicks + neg_clicks)

    return u0, u1, np.array(neg_clicks).astype(np.int64), np.array(pos_clicks).astype(np.int64), num_clicks

  def use_segmentation_mask(self, res):
    super().use_segmentation_mask(res)
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
      fn = main_folder + filename.split(".")[0].split("/")[-1] + "_" + str(filename.split(":")[-1]) + \
           str(num_clicks) + ".png"
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

  def is_ignore(self, filename):
    return False
