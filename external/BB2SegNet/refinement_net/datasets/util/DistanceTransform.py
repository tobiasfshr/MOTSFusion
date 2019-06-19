import pdb

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import norm


def get_distance_transform(pts, label):
  dt = np.ones_like(label)
  if len(pts) > 0:
    for y, x in pts:
      dt[y, x] = 0
    dt = distance_transform_edt(dt)
    return dt
  else:
    # This is important since we divide it by 255 while normalizing the inputs.
    return dt * 255


def encode_extreme_points(mask, use_gaussian):
  nz = mask.nonzero()
  assert nz[0].size > 0, "mask cannot be empty"
  ymin_idx = nz[0].argmin()
  ymax_idx = nz[0].argmax()
  xmin_idx = nz[1].argmin()
  xmax_idx = nz[1].argmax()
  ymin = (nz[0][ymin_idx], nz[1][ymin_idx])
  ymax = (nz[0][ymax_idx], nz[1][ymax_idx])
  xmin = (nz[0][xmin_idx], nz[1][xmin_idx])
  xmax = (nz[0][xmax_idx], nz[1][xmax_idx])
  pts = (ymin, ymax, xmin, xmax)
  distance_transform_extreme_pts = get_distance_transform(pts, mask)
  distance_transform_extreme_pts[distance_transform_extreme_pts > 20] = 20
  if use_gaussian:
    distance_transform_extreme_pts = norm.pdf(distance_transform_extreme_pts, loc=0, scale=10) * 25
  else:
    distance_transform_extreme_pts /= 20.0
  return distance_transform_extreme_pts.astype(np.float32)


def signed_distance_transform(mask):
  sdt = tf.py_func(signed_distance_transform_np, [mask], tf.float32)
  sdt.set_shape(mask.get_shape())
  return sdt


def signed_distance_transform_np(mask):
  # Handle empty mask
  if not np.any(mask):
    dt_pos = np.ones_like(mask) * 128
  else:
    dt_pos = distance_transform_edt(np.logical_not(mask))
  dt_neg = distance_transform_edt(mask)
  sdt = dt_pos - dt_neg
  sdt = np.clip(sdt, -128, 128)
  sdt /= 128
  return sdt.astype("float32")


def unsigned_distance_transform(mask):
  sdt = signed_distance_transform(mask)
  udt = tf.maximum(sdt, 0)
  return udt


def test_signed_distance_transform_np():
  import matplotlib.pyplot as plt
  # mask_filename = "/work/voigtlaender/data/LASER/KITTI/KITTI_laser_v1/000000.png"
  # mask = np.array(Image.open(mask_filename)) > 0
  mask = np.zeros([250,250])
  mask[20:50, 20:50] = 1
  sdt = signed_distance_transform_np(mask)
  plt.imshow(sdt)
  plt.show()


if __name__ == "__main__":
  test_signed_distance_transform_np()
