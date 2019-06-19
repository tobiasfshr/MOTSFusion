import json
import pycocotools.coco as coco
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from refinement_net.datasets.util.Util import username

coco_data_folder = "/fastwork/" + username() + "/mywork/data/coco/"
minival_gt_file = coco_data_folder + "annotations/instances_val2014.json"
minival_det_file = "/home/krause/vision/tensorpack/examples/FasterRCNN/TensorpackModelzooResnet50MinivalDetections.json"

coco = coco.COCO(minival_gt_file)
detections_list = json.load(open(minival_det_file))
detections_by_imgid = {}
for det in detections_list:
  img_id = det['image_id']
  if img_id in detections_by_imgid:
    detections_by_imgid[img_id].append(det)
  else:
    detections_by_imgid[img_id] = [det]


def visualize(img_id):
  img_descriptor = coco.loadImgs(img_id)
  file_name = coco_data_folder + "val/" + img_descriptor[0]['file_name']

  fig, ax = plt.subplots(1)
  img = mpimg.imread(file_name)
  ax.imshow(img)

  gt_ann_ids = coco.getAnnIds(imgIds=[img_id])
  gt_anns = coco.loadAnns(gt_ann_ids)
  dets = detections_by_imgid[img_id]
  print("Image", img_id, "Dets", len(dets), "GT", len(gt_anns))

  for gt in gt_anns:
    draw_box(ax, gt['bbox'], 'r', gt['category_id'], 1.0)
  for det in dets:
    draw_box(ax, det['bbox'], 'b', det['category_id'], det['score'])

  plt.show()


def draw_box(ax, bbox, color, cat_id, alpha):
  rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,
                           edgecolor=color, facecolor='none', alpha=alpha)
  ax.add_patch(rect)
  ax.annotate(str(cat_id), (bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]), color=color, weight='bold',
              fontsize=10, ha='center', va='center', alpha=1.0)


if __name__ == '__main__':
  img_ids = coco.getImgIds()
  for img_id in img_ids:
    visualize(img_id)
