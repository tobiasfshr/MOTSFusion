import tensorflow as tf
import numpy as np
from external.BB2SegNet.refinement_net.core.Log import log
from external.BB2SegNet.refinement_net.datasets.Dataset import FileListDataset
from external.BB2SegNet.refinement_net.datasets.util.Util import username
from external.BB2SegNet.refinement_net.datasets.Loader import register_dataset

COCO_DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/coco/"
NAME = "COCO"


@register_dataset(NAME)
class COCODataset(FileListDataset):
  def __init__(self, config, subset, num_classes, name=NAME):
    super().__init__(config, name, subset, COCO_DEFAULT_PATH, num_classes)

    if subset == "train":
      self.data_type = "train2014"
      self.filter_crowd_images = config.bool("filter_crowd_images", False)
      self.min_box_size = config.float("min_box_size", -1.0)
    else:
      self.data_type = "val2014"
      self.filter_crowd_images = False
      self.min_box_size = config.float("min_box_size_val", -1.0)
    # Use the minival split as done in https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md
    self.annotation_file = '%s/annotations/instances_%s.json' % (self.data_dir, subset)
    self.restricted_image_category_list = config.string_list("restricted_image_category_list", [])
    if len(self.restricted_image_category_list) == 0:
      self.restricted_image_category_list = None
    self.restricted_annotations_category_list = config.string_list("restricted_annotations_category_list", [])
    if len(self.restricted_annotations_category_list) == 0:
      self.restricted_annotations_category_list = None

    self.exclude_image_category_list = config.string_list("exclude_image_category_list", [])
    if len(self.exclude_image_category_list) == 0:
      self.exclude_image_category_list = None
    self.exclude_annotations_category_list = config.string_list("exclude_annotations_category_list", [])
    if len(self.exclude_annotations_category_list) == 0:
      self.exclude_annotations_category_list = None

    #either both of them or none should be specified for now to avoid unintuitive behaviour
    assert (self.restricted_image_category_list is None and self.restricted_annotations_category_list is None) or \
           (self.restricted_image_category_list is not None and self.restricted_annotations_category_list is not None),\
           (self.restricted_image_category_list, self.restricted_annotations_category_list)

    self.init_coco()

  def init_coco(self):
    # only import this dependency on demand
    import pycocotools.coco as coco
    self.coco = coco.COCO(self.annotation_file)
    ann_ids = self.coco.getAnnIds([])
    self.anns = self.coco.loadAnns(ann_ids)
    self.label_map = {k - 1: v for k, v in self.coco.cats.items()}
    self.filename_to_anns = dict()
    self.build_filename_to_anns_dict()

  def build_filename_to_anns_dict(self):
    for ann in self.anns:
      img_id = ann['image_id']
      img = self.coco.loadImgs(img_id)
      file_name = img[0]['file_name']
      if file_name in self.filename_to_anns:
        self.filename_to_anns[file_name].append(ann)
      else:
        self.filename_to_anns[file_name] = [ann]
        # self.filename_to_anns[file_name] = ann
    self.filter_anns()

  def filter_anns(self):
    # exclude all images which contain a crowd
    if self.filter_crowd_images:
      self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items()
                               if not any([an["iscrowd"] for an in anns])}
    # filter annotations with too small boxes
    if self.min_box_size != -1.0:
      self.filename_to_anns = {f: [ann for ann in anns if ann["bbox"][2] >= self.min_box_size and ann["bbox"][3]
                                   >= self.min_box_size] for f, anns in self.filename_to_anns.items()}

    # remove annotations with crowd regions
    self.filename_to_anns = {f: [ann for ann in anns if not ann["iscrowd"]]
                             for f, anns in self.filename_to_anns.items()}
    # restrict images to contain considered categories
    if self.restricted_image_category_list is not None:
      print("filtering images to contain categories", self.restricted_image_category_list, file=log.v1)
      self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items()
                               if any([self.label_map[ann["category_id"] - 1]["name"]
                                       in self.restricted_image_category_list for ann in anns])}
      for cat in self.restricted_image_category_list:
        n_imgs_for_cat = sum([1 for anns in self.filename_to_anns.values() if
                              any([self.label_map[ann["category_id"] - 1]["name"] == cat for ann in anns])])
        print("number of images containing", cat, ":", n_imgs_for_cat, file=log.v5)
    # exclude images that only contain objects in the given list
    elif self.exclude_image_category_list is not None:
      print("Excluding images categories", self.exclude_image_category_list, file=log.v1)
      self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items()
                               if any([self.label_map[ann["category_id"] - 1]["name"]
                                       not in self.exclude_image_category_list for ann in anns])}

    # restrict annotations to considered categories
    if self.restricted_annotations_category_list is not None:
      print("filtering annotations to categories", self.restricted_annotations_category_list, file=log.v1)
      self.filename_to_anns = {f: [ann for ann in anns if self.label_map[ann["category_id"] - 1]["name"]
                                   in self.restricted_annotations_category_list]
                               for f, anns in self.filename_to_anns.items()}
    elif self.exclude_annotations_category_list is not None:
      print("Excluding annotations for object categories", self.exclude_annotations_category_list, file=log.v1)
      self.filename_to_anns = {f: [ann for ann in anns if self.label_map[ann["category_id"] - 1]["name"]
                                   not in self.exclude_annotations_category_list]
                               for f, anns in self.filename_to_anns.items()}

    # filter out images without annotations
    self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items() if len(anns) > 0}
    n_before = len(self.anns)
    self.anns = []
    for anns in self.filename_to_anns.values():
      self.anns += anns
    n_after = len(self.anns)
    print("filtered annotations:", n_before, "->", n_after, file=log.v1)

  def load_image(self, img_filename):
    path = tf.string_split([img_filename], '/').values[-1]
    # path = tf.Print(path, [path])
    img_dir = tf.cond(tf.equal(tf.string_split([path], '_').values[1], tf.constant("train2014")),
                      lambda: '%s/%s/' % (self.data_dir, "train2014"),
                      lambda: '%s/%s/' % (self.data_dir, "val2014"))
    path = img_dir + path
    return super().load_image(path)

  def load_annotation(self, img, img_filename, annotation_filename):
    label = tf.py_func(self._get_mask, [img_filename], tf.uint8)
    label.set_shape((None, None, 1))
    return label

  def _get_mask(self, img_filename):
    ann = self.filename_to_anns[img_filename.split("/")[-1]]
    img = self.coco.loadImgs(ann[0]['image_id'])[0]

    height = img['height']
    width = img['width']

    label = np.zeros((height, width, 1))
    label[:, :, 0] = self.coco.annToMask(ann[0])[:, :]
    if len(np.unique(label)) == 1:
      print("GT contains only background.", file=log.v1)

    return label.astype(np.uint8)

  def read_inputfile_lists(self):
    img_dir = '%s/%s/' % (self.data_dir, self.data_type)
    # Filtering the image file names since some of them do not have annotations.
    imgs = [img_dir + fn for fn in self.filename_to_anns.keys()]
    img_ids = [anns[0]["image_id"] for fn, anns in self.filename_to_anns.items()]
    return imgs, imgs
