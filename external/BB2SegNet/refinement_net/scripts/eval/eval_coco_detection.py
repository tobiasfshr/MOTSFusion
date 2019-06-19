import json
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from refinement_net.datasets.util.Util import username

coco_data_folder = "/fastwork/" + username() + "/mywork/data/coco/"
minival_gt_file = coco_data_folder + "annotations/instances_val2014.json"
minival_det_file = "/home/krause/vision/savitar2/forwarded/frcnn_test/frcnn_test-1-detections.json"


def evaluate_coco():
  c = coco.COCO(minival_gt_file)
  cocoDt = c.loadRes(minival_det_file)
  cocoEval = COCOeval(c, cocoDt, 'bbox')
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()
  print(cocoEval.stats[0])


def adjust_detections():
  c = coco.COCO(minival_gt_file)
  keys = list(c.cats.keys())
  detections_list = json.load(open(minival_det_file))
  for det in detections_list:
    det['category_id'] = keys[det['category_id']]
  json.dump(detections_list, open("/home/krause/vision/savitar2/forwarded/temp_edited.json", 'w'))


if __name__ == '__main__':
  evaluate_coco()
