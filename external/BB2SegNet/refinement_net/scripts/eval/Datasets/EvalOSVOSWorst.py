import glob
import os
import random

import numpy as np
from scipy.misc import imread

from refinement_net.core.Measures import compute_measures_for_binary_segmentation_single_image, IOU
from refinement_net.datasets import DataKeys
from refinement_net.datasets.DAVIS import DAVIS
from refinement_net.datasets.Dataset import FileListDataset
from refinement_net.scripts.eval.Datasets.EvalPascalMasked import EvalPascalMaskedDataset

NAME = "OSVOSworst"
DAVIS_PATH = DAVIS.DAVIS_DEFAULT_PATH


def get_fn_with_worst_iou(seq):
  result_fn = None
  result_gt = None
  result_measure = None
  files = glob.glob(seq + "/*.png")
  seq_name = seq.split("/")[-1]
  for file in files:
    fname = file.split("/")[-1]
    img = imread(file)
    img = img / 255

    gt_file = DAVIS_PATH + "/Annotations/480p/" + seq_name + "/" + fname
    gt = imread(gt_file)
    gt = gt / 255
    measure = compute_measures_for_binary_segmentation_single_image(img, gt)
    if measure is None:
      print(fn_file, gt_file, measure)
    if result_measure is None or measure[IOU] < result_measure[IOU]:
      result_measure = measure
      result_fn = DAVIS_PATH + "/JPEGImages/480p/" + seq_name + "/" + fname.replace(".png", ".jpg")
      result_gt = gt_file

  return result_fn, result_gt, result_measure


class OSVOSWorst(FileListDataset):
  def __init__(self, config, subset, name=NAME):
    super(OSVOSWorst, self).__init__(config, name, subset, num_classes=2, default_path=DAVIS_PATH)
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
    mask = mask / 255
    return {DataKeys.SEGMENTATION_LABELS: mask, DataKeys.RAW_SEGMENTATION_LABELS: mask,
            DataKeys.IMAGE_FILENAMES: ann_filename}

  def use_segmentation_mask(self, res):
    self.eval_pascal_dataset.use_segmentation_mask(res)

  def read_inputfile_lists(self):
    pre_computed = DAVIS_PATH + "/pre_computed/"
    imgs = []
    gts = []
    measures = []

    # get all video sequences
    seqs = [os.path.join(pre_computed, f) for f in os.listdir(pre_computed) if os.path.isdir(os.path.join(pre_computed, f))]
    for seq in seqs:
      fn, gt, measure = get_fn_with_worst_iou(seq)
      measures += [measure]
      imgs += [fn]
      gts += [gt]
    
    print(measures)
    ious = [m[IOU] for m in measures]
    print("Average IOU Initial: ", np.average(ious))
    return imgs, gts

