import glob

import tensorflow as tf

from external.BB2SegNet.refinement_net.datasets.Loader import register_dataset
from external.BB2SegNet.refinement_net.datasets.Dataset import FileListDataset
from external.BB2SegNet.refinement_net.datasets.util.Util import username

NUM_CLASSES = 2
VOID_LABEL = 255  # for translation augmentation
DAVIS_DEFAULT_IMAGES_PATH = "/work2/" + username() + "/data/DAVIS/train-val/val/"
DAVIS_DEFAULT_GT_PATH = "/work2/" + username() + "/data/DAVIS/train-val/val-gt/"
DAVIS_DEFAULT_PROPOSALS_PATH = "/home/" + username() + "/vision/maskrcnn_tensorpack/train_log/thesis1/notrain-val/"
DAVIS_IMAGE_SIZE = (480, 854)

@register_dataset("davisjono")
class DAVISjonoDataset(FileListDataset):
  def __init__(self, config, subset, name="davisjono"):
    self.image_dir = config.string("DAVIS_images_dir", DAVIS_DEFAULT_IMAGES_PATH)
    self.gt_dir = config.string("DAVIS_gt_dir", DAVIS_DEFAULT_GT_PATH)
    self.proposals_dir = config.string("DAVIS_proposals_dir", DAVIS_DEFAULT_PROPOSALS_PATH)

    super().__init__(config, name, subset, DAVIS_DEFAULT_IMAGES_PATH, 2)

  def read_inputfile_lists(self):
    img_filenames = glob.glob(self.image_dir + "*/*.jpg")
    seq_tags = [f.split("/")[-2] for f in img_filenames]
    framenum = [f.split('/')[-1].split('.jpg')[0] for f in img_filenames]
    label_filenames = [self.gt_dir + s + '/' + f + '.png' for s,f in zip(seq_tags,framenum)]
    return img_filenames, label_filenames

  def load_annotation(self, img, img_filename, annotation_filename):
    seq = img_filename.split("/")[-2]
    framenum = img_filename.split('/')[-1].split('.jpg')[0]
    proposal_filename = self.proposals_dir + seq + '/' + framenum + '.json'
    def _load_ann(img_, ann_filename):
      json = ann_filename.decode("utf-8")
      seq = json.split("/")[-3]
      t = int(json.split("/")[-1].replace(".json", ""))
      id_ = int(json.split("/")[-2])
      label_ = load_mask_from_json(img_, json)
      bbox_ = self._tracking_gt[seq][id_][t].bbox_x0y0x1y1.astype("float32")[[1, 0, 3, 2]]
      return label_, bbox_
    label, bbox = tf.py_func(_load_ann, [img, annotation_filename], [tf.uint8, tf.float32])
    label.set_shape((None, None, 1))
    bbox.set_shape((4,))
    return {DataKeys.SEGMENTATION_LABELS: label, DataKeys.BBOXES_y0x0y1x1: bbox}