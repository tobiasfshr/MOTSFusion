import random
import numpy as np
from refinement_net.core.Log import log
from refinement_net.datasets.COCO.COCO_masked_dios import COCOMaskedDiosDataset
from refinement_net.scripts.eval.Datasets.EvalPascalMasked import EvalPascalMaskedDataset


class EvalCOCODataset(COCOMaskedDiosDataset):
  def __init__(self, config, subset):
    self.cat_to_filenames = {}
    self.randomly_sample_images = config.bool("randomly_sample_images", False)
    self.save_images = config.bool("save_images", False)
    self.img_dir = config.string("img_dir", str(random.randrange(1, 10000)))
    super(EvalCOCODataset, self).__init__(config, subset)
    self.eval_pascal_dataset = EvalPascalMaskedDataset(config, subset)
    self.previous_epoch_data = self.eval_pascal_dataset.previous_epoch_data

  # def init_coco(self):
  #   # self.data_type = "val2017" if self.subset == "valid" else "train2017"
  #   self.annotation_file = '%s/annotations/instances_%s.json' % (self.data_dir, self.data_type)
  #   # only import this dependency on demand
  #   import pycocotools.coco as coco
  #   self.coco = coco.COCO(self.annotation_file)
  #
  #   ann_ids = self.coco.getAnnIds([])
  #   self.anns = self.coco.loadAnns(ann_ids)
  #   self.label_map = {k - 1: v for k, v in self.coco.cats.items()}
  #
  #   self.filename_to_anns = dict()
  #   self.build_filename_to_anns_dict()

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
    # Filter iscrowd
    self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items()
                             if not any([an["iscrowd"] for an in anns])}
    # restrict images to contain considered categories
    if self.restricted_image_category_list is not None:
      print("filtering images to contain categories", self.restricted_image_category_list, file=log.v1)
      self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items()
                               if any([self.label_map[ann["category_id"] - 1]["name"]
                                       in self.restricted_image_category_list for ann in anns])}

      print("filtering annotations to categories", self.restricted_annotations_category_list, file=log.v1)
      self.filename_to_anns = {f: [ann for ann in anns if self.label_map[ann["category_id"] - 1]["name"]
                                   in self.restricted_annotations_category_list]
                               for f, anns in self.filename_to_anns.items()}

      for cat in self.restricted_image_category_list:
        self.cat_to_filenames[cat] = [fn for fn, anns in self.filename_to_anns.items() if
                                      any([self.label_map[ann["category_id"] - 1]["name"] == cat for ann in anns])]
        print("number of images containing", cat, ":", len(self.cat_to_filenames[cat]), file=log.v5)

    filter_area = self.config.int("filter_area", 0)

    # filter segmentations by area
    self.filename_to_anns = {f: [ann for ann in anns if ann["area"] >= filter_area * filter_area]
                             for f, anns in self.filename_to_anns.items()}

    # filter out images without annotations
    self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items() if len(anns) > 0}
    self.cat_to_filenames = {cat: [fn for fn in fns
                                   if fn in self.filename_to_anns]
                             for cat, fns in self.cat_to_filenames.items()}

    n_before = len(self.anns)
    self.anns = []
    for anns in self.filename_to_anns.values():
      self.anns += anns
    n_after = len(self.anns)
    print("filtered annotations:", n_before, "->", n_after, file=log.v1)

  def _get_mask(self, img_filename):
    img_filename = img_filename.decode("UTF-8")
    img_id = img_filename.split(":")[1]
    img = self.coco.loadImgs([int(img_id)])[0]

    ann_ids = [int(img_filename.split(":")[2])]
    # ann_ids = self.coco.getAnnIds([int(img_id)])
    anns = self.coco.loadAnns(ann_ids)

    height = img['height']
    width = img['width']

    label = np.zeros((height, width, 1))
    label[:, :, 0] = self.coco.annToMask(anns[0])[:, :]
    if len(np.unique(label)) == 1:
      print("GT contains only background.", file=log.v1)

    return label.astype(np.uint8), label.astype(np.uint8)

  def get_extraction_keys(self):
    return self.eval_pascal_dataset.get_extraction_keys()

  def use_segmentation_mask(self, res):
    self.eval_pascal_dataset.use_segmentation_mask(res)

  def postproc_example_before_assembly(self, tensors):
    return self.eval_pascal_dataset.postproc_example_before_assembly(tensors)

  def create_summaries(self, data):
    super().create_summaries(data)

  def read_inputfile_lists(self):
    imgs = []
    fns = []
    img_dir = '%s/%s/' % (self.data_dir, self.data_type)
    if self.randomly_sample_images and self.subset == "valid":
      for cat in self.cat_to_filenames.keys():
        fns = random.sample(self.cat_to_filenames[cat],
                            10)
        for fn in fns:
          anns = [img_dir + fn + ":" + repr(img_id) + ":" + repr(ann_id) + ":" + cat
                  for img_id, ann_id in [(ann['image_id'], ann['id']) for ann in self.filename_to_anns[fn]
                                         if self.label_map[ann["category_id"] - 1]["name"] == cat]]
          if len(anns) > 0:
            # imgs += random.sample(anns, 1)
            imgs += anns

      return imgs, imgs
    else:
      return super().read_inputfile_lists()