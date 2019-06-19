#!/usr/bin/env python
import glob
import tensorflow as tf
from PIL import Image
import numpy as np
import colorsys
from multiprocessing import Pool
from functools import partial


# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def generate_colors():
  """
  Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """
  N = 30
  brightness = 0.7
  hsv = [(i / N, 1, brightness) for i in range(N)]
  colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
  perm = [15, 13, 25, 12, 19, 8, 22, 24, 29, 17, 28, 20, 2, 27, 11, 26, 21, 4, 3, 18, 9, 5, 14, 1, 16, 0, 23, 7, 6, 10]
  colors = [colors[idx] for idx in perm]
  return colors


# from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def apply_mask(image, mask, color, alpha=0.5):
  """Apply the given mask to the image.
  """
  for c in range(3):
    image[:, :, c] = np.where(mask == 1,
                              image[:, :, c] *
                              (1 - alpha) + alpha * color[c] * 255,
                              image[:, :, c])
  return image


MASK_DIR = "/home/voigtlaender/vision/savitar2/forwarded/mapillary_quarter_jitter005_deeplab_KITTI_tracking_all_fwd_KITTI_2/mask/step_0/"
OUT_DIR = "/home/voigtlaender/vision/savitar2/forwarded/mapillary_quarter_jitter005_deeplab_KITTI_tracking_all_fwd_KITTI_2/merged/"
IMG_DIR = "/work/voigtlaender/data/kitti_training_minimum/training/image_02/"

tf.gfile.MakeDirs(OUT_DIR)


def do_img(img, mask_dir_vid, out_dir_vid):
  colors = generate_colors()
  img_ending = img.split("/")[-1]
  im = np.array(Image.open(img))
  mask_files = sorted(glob.glob(mask_dir_vid + "/*/" + img_ending))
  for mask_file, color in zip(mask_files, colors):
    mask = np.array(Image.open(mask_file)).astype("bool")
    id_ = int(mask_file.split("/")[-2])
    # im[mask, 0] = 1
    apply_mask(im, mask, colors[id_ % len(colors)])
  print(out_dir_vid + "/" + img_ending)
  Image.fromarray(im).save(out_dir_vid + "/" + img_ending)


def do_vid(vid_id):
  vid_str = "%04d" % vid_id
  out_dir_vid = OUT_DIR + "/" + vid_str
  tf.gfile.MakeDirs(out_dir_vid)
  mask_dir_vid = MASK_DIR + "/" + vid_str
  imgs = sorted(glob.glob(IMG_DIR + vid_str + "/*.png"))
  do_img_ = partial(do_img, mask_dir_vid=mask_dir_vid, out_dir_vid=out_dir_vid)
  Pool(processes=16).map(do_img_, imgs)
  #for img in imgs:
  #  do_img(img)


for id_ in range(21):
  do_vid(id_)
