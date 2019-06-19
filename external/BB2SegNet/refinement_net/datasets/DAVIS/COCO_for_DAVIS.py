import tensorflow as tf
from PIL import Image
import numpy as np
from random import shuffle

from external.BB2SegNet.refinement_net.datasets.Loader import register_dataset
from external.BB2SegNet.refinement_net.datasets.Mapillary.MapillaryLike_instance import MapillaryLikeInstanceDataset
from external.BB2SegNet.refinement_net.datasets.util.Util import username

DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/coco/train2014/"
LIST_PATH_ROOT = "/home/luiten/vision/youtubevos/refinement_net/"
NAME = "COCO_for_davis"

@register_dataset(NAME)
class DAVISLucidDataset(MapillaryLikeInstanceDataset):
  def __init__(self, config, subset):
    davis_sequence = config.string("model", '')
    # data_list_path = LIST_PATH_ROOT + davis_sequence + '/'
    self.data_dir = config.string('data_dir',DEFAULT_PATH)
    annotation_file = "/fastwork/" + username() + "/mywork/data/coco/annotations/instances_train2014.json"
    self.build_filename_to_coco_anns_dict(annotation_file)
    super().__init__(config, subset, NAME, self.data_dir, "", 100, cat_ids_to_use=None)

# Two things:
  # # 1.) Load a dict to be used later
  # # 2.) Load in the annotations live as needed

  ###################################################################################################
  # # 1.)
  ###################################################################################################

  def read_inputfile_lists(self):
    imgs_ans = []
    for f,anns in self.filename_to_coco_anns.items():
      for id,ann in enumerate(anns):
        id_ = str(id)
        if ann['area'] < self.min_size:
          continue
        imgs_ans.append((f, id_))
    shuffle(imgs_ans)
    imgs = [x[0] for x in imgs_ans]
    ans = [x[1] for x in imgs_ans]
    return imgs, ans

  def build_filename_to_coco_anns_dict(self, annotation_file):
    import pycocotools.coco as coco
    self.coco = coco.COCO(annotation_file)
    ann_ids = self.coco.getAnnIds([])
    all_anns = self.coco.loadAnns(ann_ids)

    imgs = self.coco.loadImgs(self.coco.getImgIds())
    self.filename_to_coco_anns = {self.data_dir + img['file_name']: [] for img in imgs}
    self.filename_to_img_ids = {self.data_dir + img['file_name']: img['id'] for img in imgs}

    # load all annotations for images
    for ann in all_anns:
      img_id = ann['image_id']
      img = self.coco.loadImgs(img_id)
      file_name = self.data_dir + img[0]['file_name']
      self.filename_to_coco_anns[file_name].append(ann)

    # Remove crowd anns
    self.filename_to_coco_anns = {f: [ann for ann in anns if not ann["iscrowd"]]
                                  for f, anns in self.filename_to_coco_anns.items()}

    # filter out images without annotations
    self.filename_to_coco_anns = {f: anns for f, anns in self.filename_to_coco_anns.items() if len(anns) > 0}

  ###################################################################################################
  # # 2.)
  ###################################################################################################

  def segmentation_to_mask(self,polys, height, width):
    import pycocotools.mask as cocomask
    polys = [p.flatten().tolist() for p in polys]
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)

  def load_ann(img,img_filename,annotation_filename):
    img_filename = img_filename.decode('utf-8')
    anns_for_img = self.filename_to_coco_anns[img_filename.split("/")[-1]]
    ann_id = int(annotation_filename.decode('utf-8'))
    ann = anns_for_img[ann_id]
    img_h, img_w = img.shape[:-1]

    if ann['area'] > 1 and isinstance(ann['segmentation'], list):
      segs = ann['segmentation']
      valid_segs = [np.asarray(p).reshape(-1, 2) for p in segs if len(p) >= 6]
      if len(valid_segs) < len(segs):
        print("Image {} has invalid polygons!".format(img_filename))
      output_ann = np.asarray(self.segmentation_to_mask(valid_segs, img_h, img_w), dtype='uint8')[
        ..., np.newaxis]  # Should be 1s and 0s
    else:
      output_ann = np.zeros((img_h, img_w, 1), dtype="uint8")

    return output_ann

  def load_annotation(self, img, img_filename, annotation_filename):

    def load_ann(img, img_filename, annotation_filename):
      img_filename = img_filename.decode('utf-8')
      anns_for_img = self.filename_to_coco_anns[img_filename]
      ann_id = int(annotation_filename.decode('utf-8'))
      ann = anns_for_img[ann_id]
      img_h = img.shape[0]
      img_w = img.shape[1]

      if ann['area'] > 1 and isinstance(ann['segmentation'], list):
        segs = ann['segmentation']
        valid_segs = [np.asarray(p).reshape(-1, 2) for p in segs if len(p) >= 6]
        if len(valid_segs) < len(segs):
          print("Image {} has invalid polygons!".format(img_filename))
        output_ann = np.asarray(self.segmentation_to_mask(valid_segs, img_h, img_w), dtype='uint8')[..., np.newaxis]  # Should be 1s and 0s
      else:
        output_ann = np.zeros((img_h, img_w, 1), dtype="uint8")

      return output_ann

    ann, = tf.py_func(load_ann, [img,img_filename,annotation_filename], [tf.uint8])
    # print(ann)
    # ann = tf.Print(ann,[ann,])
    # ann = ann[0]
    ann.set_shape(img.get_shape().as_list()[:-1] + [1])
    ann = tf.cast(ann, tf.uint8)
    return ann