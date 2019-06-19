import numpy

from external.BB2SegNet.refinement_net.datasets.Loader import register_dataset
from external.BB2SegNet.refinement_net.datasets import DataKeys
from external.BB2SegNet.refinement_net.datasets.Dataset import FileListDataset
from external.BB2SegNet.refinement_net.datasets.util.Util import username
import external.BB2SegNet.refinement_net.core.Measures as Measures
import tensorflow as tf
import os
import json


# From Tensorpack FasterRCNN example, common.py
def segmentation_to_mask(polys, height, width):
  """
  Convert polygons to binary masks.

  Args:
      polys: a list of nx2 float array

  Returns:
      a binary matrix of (height, width)
  """
  import pycocotools.mask as cocomask
  polys = [p.flatten().tolist() for p in polys]
  rles = cocomask.frPyObjects(polys, height, width)
  rle = cocomask.merge(rles)
  return cocomask.decode(rle)


DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/coco/"
# To clarify, COCO uses category IDs from 1 to 90 which we take verbatim (no +1/-1!!!)
# The classes_to_cat array contains the mapping
NUM_CLASSES = 81
N_MAX_DETECTIONS = 100
NAME = "COCO_detection"


@register_dataset(NAME)
class CocoDetectionDataset(FileListDataset):
  def __init__(self, config, subset):
    super().__init__(config, NAME, subset, DEFAULT_PATH, NUM_CLASSES)

    self.add_masks = config.bool("add_masks", True)
    self.exclude_crowd_images = config.bool("exclude_crowd_images", False)
    self.exclude_crowd_annotations = config.bool("exclude_crowd_annotations", True)
    self.min_box_size = config.float("min_box_size", -1.0)

    self.coco = None
    self.filename_to_coco_anns = None
    self.filename_to_img_ids = None
    annotation_file = '%s/annotations/instances_%s2014.json' % (self.data_dir, subset)
    self.build_filename_to_coco_anns_dict(annotation_file)
    self.classes_to_cat = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27,
                           28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
                           54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                           81, 82, 84, 85, 86, 87, 88, 89, 90]
    self.cat_to_class = {v: i for i, v in enumerate(self.classes_to_cat)}

  def build_filename_to_coco_anns_dict(self, annotation_file):
    import pycocotools.coco as coco
    self.coco = coco.COCO(annotation_file)
    ann_ids = self.coco.getAnnIds([])
    all_anns = self.coco.loadAnns(ann_ids)

    imgs = self.coco.loadImgs(self.coco.getImgIds())
    self.filename_to_coco_anns = {img['file_name']: [] for img in imgs}
    self.filename_to_img_ids = {img['file_name']: img['id'] for img in imgs}

    # load all annotations for images
    for ann in all_anns:
      img_id = ann['image_id']
      img = self.coco.loadImgs(img_id)
      file_name = img[0]['file_name']
      self.filename_to_coco_anns[file_name].append(ann)

    if self.subset is not 'val':
      # remove annotations that are crowds
      if self.exclude_crowd_annotations:
        self.filename_to_coco_anns = {f: [ann for ann in anns if not ann["iscrowd"]]
                                      for f, anns in self.filename_to_coco_anns.items()}

      # filter out images without annotations
      self.filename_to_coco_anns = {f: anns for f, anns in self.filename_to_coco_anns.items() if len(anns) > 0}

  def get_data_arrays_for_file(self, img_filename, img_h, img_w):
    img_filename = img_filename.decode('utf-8')
    anns_for_img = self.filename_to_coco_anns[img_filename.split("/")[-1]]
    assert (len(anns_for_img) <= N_MAX_DETECTIONS)

    # they need to be padded to N_MAX_DETECTIONS
    bboxes = numpy.zeros((N_MAX_DETECTIONS, 4), dtype="float32")
    ids = numpy.zeros(N_MAX_DETECTIONS, dtype="int32")
    classes = numpy.zeros(N_MAX_DETECTIONS, dtype="int32")
    is_crowd = numpy.zeros(N_MAX_DETECTIONS, dtype="int32")
    # TODO this will only work for batch size 1 since we arent filling up to N_MAX_DETECTIONS, but less memory needed
    masks_list = []

    for idx, ann in enumerate(anns_for_img):
      x1 = ann["bbox"][0]
      y1 = ann["bbox"][1]
      box_width = ann["bbox"][2]
      box_height = ann["bbox"][3]
      x2 = x1 + box_width
      y2 = y1 + box_height
      bboxes[idx] = [y1, x1, y2, x2]
      ids[idx] = idx + 1
      classes[idx] = self.cat_to_class[ann["category_id"]]  # TODO double check correctness of these
      is_crowd[idx] = ann["iscrowd"]

      # TODO this omits segmentations for is_crowd==True annotations
      if self.add_masks:
        if ann['area'] > 1 and isinstance(ann['segmentation'], list):
          segs = ann['segmentation']
          valid_segs = [numpy.asarray(p).reshape(-1, 2) for p in segs if len(p) >= 6]
          if len(valid_segs) < len(segs):
            print("Image {} has invalid polygons!".format(img_filename))
          masks_list.append(numpy.asarray(segmentation_to_mask(valid_segs, img_h, img_w), dtype='uint8'))  # Should be 1s and 0s
        else:
          masks_list.append(numpy.zeros((img_h, img_w), dtype="uint8"))

    if self.add_masks and len(masks_list) > 0:
      masks = numpy.stack(masks_list, axis=2)  # format HWC, where C the different annotations, for resizing
    else:
      masks = numpy.zeros((img_h, img_w, 1), dtype="uint8")
    return bboxes, ids, classes, is_crowd, masks

  def read_inputfile_lists(self):
    imgs = []
    for img_filename in self.filename_to_coco_anns.keys():
      im = self.data_dir + self.subset + '/' + img_filename
      imgs.append(im)
    return (imgs, )

  def load_annotation(self, img, img_filename, annotation_filename):
    img_shape = tf.shape(img)
    bboxes, ids, classes, is_crowd, mask = tf.py_func(self.get_data_arrays_for_file, [img_filename, img_shape[0], img_shape[1]],
                                                      [tf.float32, tf.int32, tf.int32, tf.int32, tf.uint8],
                                                      name="get_data_arrays_for_file")
    bboxes.set_shape((N_MAX_DETECTIONS, 4))
    ids.set_shape((N_MAX_DETECTIONS,))
    classes.set_shape((N_MAX_DETECTIONS,))
    is_crowd.set_shape((N_MAX_DETECTIONS,))
    if self.add_masks:
      # TODO shape is actually: img_shape[0], img_shape[1], N_MAX_DETECTIONS but would have to use max/min img size - problems with flip AND store dummy seg masks
      mask.set_shape((None, None, None))
    else:
      mask.set_shape(1)

    return_dict = {}
    return_dict[DataKeys.BBOXES_y0x0y1x1] = bboxes
    return_dict[DataKeys.CLASSES] = classes
    return_dict[DataKeys.IDS] = ids
    return_dict[DataKeys.IS_CROWD] = is_crowd
    if self.add_masks:
      return_dict[DataKeys.SEGMENTATION_MASK] = mask
    return return_dict

  def prepare_saving_epoch_measures(self, epoch):
    model_name = self.config.string("model")
    model_folder = "forwarded/" + model_name + "/"
    if not os.path.exists(model_folder):
      os.mkdir(model_folder)
    self.det_file_path = model_folder + model_name + "-" + str(epoch) + "-detections.json"
    self.detections_file = open(self.det_file_path, "w")
    self.detections_file.write("[")
    self.first = True

  def save_epoch_measures(self, measures):
    assert Measures.DET_BOXES in measures
    det_boxes = measures[Measures.DET_BOXES]
    det_scores = measures[Measures.DET_PROBS]
    det_classes = measures[Measures.DET_LABELS]
    img_id = measures[Measures.IMAGE_ID]

    if Measures.DET_MASKS in measures:
      assert self.add_masks
      det_masks = measures[Measures.DET_MASKS]
      zipper = zip(det_boxes, det_scores, det_classes, det_masks)
    else:
      assert not self.add_masks
      zipper = zip(det_boxes, det_scores, det_classes, det_classes)

    for bbox, score, class_, mask in zipper:
      if self.first:
        self.first = False
      else:
        self.detections_file.write(",")
      bbox = list(map(lambda x: float(round(x, 1)), bbox))
      output_string = """{{"image_id": {}, "category_id": {}, "bbox": [{}, {}, {}, {}], "score": {}""".format(
          int(img_id[0]), self.classes_to_cat[class_], bbox[0], bbox[1], bbox[2] - bbox[0],
          bbox[3] - bbox[1], float(round(score, 2)))
      if self.add_masks:
        import pycocotools.mask as cocomask
        rle = cocomask.encode(
          numpy.array(mask[:, :, None], order='F'))[0]
        rle['counts'] = rle['counts'].decode('ascii')
        output_string += """, "segmentation": {}""".format(
          json.dumps(rle))
      output_string += "}"
      self.detections_file.write(output_string)

  def finalize_saving_epoch_measures(self):
    self.detections_file.write("]")
    self.detections_file.close()
    self.detections_file = None

    cocoDt = self.coco.loadRes(self.det_file_path)
    from pycocotools.cocoeval import COCOeval
    cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    new_measures = {Measures.MAP_BBOX: cocoEval.stats[0]}

    if self.add_masks:
      cocoEval = COCOeval(self.coco, cocoDt, 'segm')
      cocoEval.evaluate()
      cocoEval.accumulate()
      cocoEval.summarize()
      new_measures[Measures.MAP_SEGM] = cocoEval.stats[0]

    return new_measures
