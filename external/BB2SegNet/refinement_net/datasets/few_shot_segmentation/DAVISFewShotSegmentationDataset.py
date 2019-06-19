import numpy as np
import os
import json
import glob
from PIL import Image
from collections import OrderedDict
from multiprocessing import Pool
from functools import partial

from external.BB2SegNet.refinement_net.datasets import DataKeys
from external.BB2SegNet.refinement_net.datasets.util.Util import username
from external.BB2SegNet.refinement_net.datasets.Loader import register_dataset
from external.BB2SegNet.refinement_net.datasets.few_shot_segmentation.FewShotSegmentation import FewShotSegmentationDataset

# DAVIS_DEFAULT_IMAGES_PATH = "/work2/" + username() + "/data/DAVIS/train-val/val17/"
# # DAVIS_DEFAULT_PROPOSALS_PATH = "/home/" + username() + "/vision/maskrcnn_tensorpack/train_log/thesis1/notrain-val/"
# DAVIS_DEFAULT_PROPOSALS_PATH = "/home/" + username() + "/vision/maskrcnn_tensorpack/train_log/thesis1/trained-val-restricted/"

# DAVIS_DEFAULT_IMAGES_PATH = "/home/luiten/vision/PReMVOS/home_data/%s/images/"
# DAVIS_DEFAULT_PROPOSALS_PATH = "/home/luiten/vision/PReMVOS/mask_restricted/%s/images/"
# DAVIS_DEFAULT_PROPOSALS_PATH = "/home/luiten/vision/PReMVOS/post_merge_props/%s/"
# DAVIS_DEFAULT_PROPOSALS_PATH = "/home/luiten/vision/PReMVOS/proposal_expansion/%s/"

DAVIS_DEFAULT_IMAGES_PATH = ""
DAVIS_DEFAULT_PROPOSALS_PATH = "/"

DATA_KEYS_TO_USE = (DataKeys.IMAGES, DataKeys.SEGMENTATION_LABELS, DataKeys.BBOXES_y0x0y1x1, DataKeys.IMAGE_FILENAMES, DataKeys.OBJ_TAGS)


def load_proposal_for_tag(tag, proposals_dir):
  # files = glob.glob(proposals_dir + tag + "/*.json")
  files = sorted(glob.glob(proposals_dir + tag + "*.json"))
  framenum = [f.split('/')[-1].split('.json')[0] for f in files]
  vid_props = OrderedDict()
  for frame in framenum:
    t = int(frame)
    # filename = proposals_dir + "/" + tag + "/" + frame + ".json"
    filename = proposals_dir + tag + frame + ".json"
    with open(filename, "r") as f:
      curr_props = json.load(f)
    # sort by score
    # curr_props.sort(key=lambda x: x["score"], reverse=True)
    vid_props[t] = curr_props
  # proposals[tag] = vid_props
  return vid_props

def load_obj_data(t, data_for_t, img_dir):
  obj_data = {}
  img_filename = img_dir + "%05d" % t + ".jpg"
  img_cache_seq = {}
  img = np.array(Image.open(img_filename)).astype("float32") / 255
  img_cache_seq[img_filename] = img
  label = np.zeros(img.shape[:2] + (1,), dtype=np.uint8)
  for prop_id, data_for_prop in enumerate(data_for_t):

    x0, y0, x1, y1 = data_for_prop['bbox']
    x1 = x1+x0
    y1 = y1+y0
    # print('bbox', x1-x0, y1-y0)
    bbox = [y0, x0, y1, x1]
    obj_data[prop_id] = {DataKeys.IMAGES: img,
                         DataKeys.SEGMENTATION_LABELS: label,
                         DataKeys.IMAGE_FILENAMES: img_filename,
                         DataKeys.BBOXES_y0x0y1x1: bbox,
                         DataKeys.OBJ_TAGS: str(prop_id)}
  if len(obj_data) > 0:
    # data_per_object.append(obj_data)
    return obj_data
  else:
    return None

def load_obj_data_TRAIN(t, data_for_t, img_dir):
  obj_data = {}
  img_filename = img_dir + "%05d" % t + ".jpg"
  img_cache_seq = {}
  img = np.array(Image.open(img_filename)).astype("float32") / 255
  img_cache_seq[img_filename] = img
  label = np.zeros(img.shape[:2] + (1,), dtype=np.uint8)
  for prop_id, data_for_prop in enumerate(data_for_t):

    x0, y0, x1, y1 = data_for_prop['bbox']
    x1 = x1+x0
    y1 = y1+y0
    # print('bbox', x1-x0, y1-y0)
    bbox = [y0, x0, y1, x1]
    obj_data[prop_id] = {DataKeys.IMAGES: img,
                         DataKeys.SEGMENTATION_LABELS: label,
                         DataKeys.IMAGE_FILENAMES: img_filename,
                         DataKeys.BBOXES_y0x0y1x1: bbox,
                         DataKeys.OBJ_TAGS: str(prop_id)}
  if len(obj_data) > 0:
    # data_per_object.append(obj_data)
    return obj_data
  else:
    return None


@register_dataset("jono_davis_fewshot")
class DAVISFewShotSegmentationDataset(FewShotSegmentationDataset):
  def __init__(self, config, subset):

    # model_name = config.string("model","test-challenge/bike-trial")
    # self.image_dir = config.string("DAVIS_images_dir", DAVIS_DEFAULT_IMAGES_PATH)
    # self.image_dir = config.string("DAVIS_images_dir", DAVIS_DEFAULT_IMAGES_PATH)%model_name
    # super().__init__(config, subset, DAVIS_DEFAULT_IMAGES_PATH, data_keys_to_use=DATA_KEYS_TO_USE)
    # proposals_dir = config.string("DAVIS_proposals_dir", DAVIS_DEFAULT_PROPOSALS_PATH)
    # proposals_dir = config.string("DAVIS_proposals_dir", DAVIS_DEFAULT_PROPOSALS_PATH) % model_name


    self.image_dir = config.string("image_input_dir", None)
    super().__init__(config, subset, "", data_keys_to_use=DATA_KEYS_TO_USE)
    proposals_dir = config.string("bb_input_dir", None)
    self.proposals = self.load_proposals(proposals_dir)

    self._video_tags = self._read_video_tags()

  def load_proposals(self,proposals_dir):
    proposals = OrderedDict()
    print("pre-prop-load")
    _load_proposals = partial(load_proposal_for_tag, proposals_dir=proposals_dir)
    with Pool(12) as pool:
      proposals_list = pool.map(_load_proposals, self._video_tags)
    print("post-prop-load")
    for tag, prop in zip(self._video_tags, proposals_list):
      proposals[tag] = prop
    return proposals

  def _read_video_tags(self):

    #####################################################################################
    # Edit for multi-gpu
    curr_run_num = 0
    total_to_run = 1
    # curr_run_num = 7
    # total_to_run = 12
    # curr_run_num = 3
    # total_to_run = 6
    video_dirnames = sorted(glob.glob(self.image_dir + "*/"))

    # done_already = sorted(glob.glob("/home/luiten/vision/youtubevos/ytvos_data/test_all_frames/refined_props-final/*/"))
    # replaced = [d.replace("refined_props-final","JPEGImages") for d in done_already]
    # video_dirnames = [v for v in video_dirnames if v not in replaced]
    # print("JJOOOOOOOOOOONNNNNNNNNNNOOOOOO",len(video_dirnames))

    num_per = int(np.ceil(len(video_dirnames) / total_to_run))
    print(curr_run_num, num_per, curr_run_num * num_per, (curr_run_num + 1) * num_per)
    sub_video_dirnames = video_dirnames[curr_run_num * num_per: (curr_run_num + 1) * num_per]
    tags = [f.split("/")[-2] + "/" for f in sub_video_dirnames]
    print(curr_run_num, num_per, curr_run_num * num_per, (curr_run_num + 1) * num_per,len(video_dirnames),len(sub_video_dirnames))
    #####################################################################################

    # video_dirnames = sorted(glob.glob(self.image_dir + "*/"))
    # tags = [f.split("/")[-2] + "/" for f in video_dirnames]

    # video_dirnames = [self.image_dir,]
    # tags = ["" for f in video_dirnames]
    return tags

  def _load_video_data(self, idx):
    tag = self.get_video_tag()
    vid_proposals = self.proposals[tag]
    img_dir = self.image_dir + tag + "/"
    img_cache_seq = {}
    dummy_label = None
    data_per_object = []
    #for t, data_for_t in vid_proposals.items():
    _load_obj_data = partial(load_obj_data, img_dir=img_dir)
    with Pool(8) as pool:
      data_per_object = [x for x in pool.starmap(_load_obj_data, vid_proposals.items()) if x is not None]
    return data_per_object
