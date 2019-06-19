import numpy as np
import tensorflow as tf

from external.BB2SegNet.refinement_net.core.Log import log
from external.BB2SegNet.refinement_net.datasets.Loader import register_dataset
from external.BB2SegNet.refinement_net.datasets import DataKeys
from external.BB2SegNet.refinement_net.datasets.COCO.COCO import COCODataset

NAME = "COCO_instance"


@register_dataset(NAME)
class COCOInstanceDataset(COCODataset):
  def __init__(self, config, subset, name=NAME):
    super(COCOInstanceDataset, self).__init__(config, subset, name=name, num_classes=2)

  def build_filename_to_anns_dict(self):
    for ann in self.anns:
      ann_id = ann['id']
      img_id = ann['image_id']
      img = self.coco.loadImgs(img_id)
      file_name = img[0]['file_name']

      file_name = file_name + ":" + repr(img_id) + ":" + repr(ann_id)
      if file_name in self.filename_to_anns:
        print("Ignoring instance as an instance with the same id exists in filename_to_anns.", log.v1)
      else:
        self.filename_to_anns[file_name] = [ann]

    self.filter_anns()

  def filter_anns(self):
    self.filter_crowd_images = True
    super().filter_anns()

  def load_image(self, img_filename):
    path = tf.string_split([img_filename], ':').values[0]
    path = tf.string_split([path], '/').values[-1]
    img_dir = tf.cond(tf.equal(tf.string_split([path], '_').values[1], tf.constant("train2014")),
                      lambda: '%s/%s/' % (self.data_dir, "train2014"),
                      lambda: '%s/%s/' % (self.data_dir, "val2014"))
    path = img_dir + path
    return super().load_image(path)

  def load_annotation(self, img, img_filename, annotation_filename):
    label, raw_label = tf.py_func(self._get_mask, [img_filename], [tf.uint8, tf.uint8], name="load_mask")
    label.set_shape((None, None, 1))
    raw_label.set_shape((None, None, 1))
    return {DataKeys.SEGMENTATION_LABELS: label, DataKeys.RAW_SEGMENTATION_LABELS: raw_label,
            DataKeys.IMAGE_FILENAMES: img_filename}

  def _get_mask(self, img_filename):
    img_filename = img_filename.decode("UTF-8")
    label = super()._get_mask(img_filename)
    img_id = img_filename.split(":")[1]
    ann_ids = self.coco.getAnnIds([int(img_id)])
    anns = self.coco.loadAnns(ann_ids)
    raw_label = self.coco.annToMask(anns[0])

    for ann in anns[1:]:
      if not ann['iscrowd']:
        raw_label = np.logical_or(raw_label, self.coco.annToMask(ann))

    raw_label = np.expand_dims(raw_label, axis=2)
    return label.astype(np.uint8), raw_label.astype(np.uint8)
