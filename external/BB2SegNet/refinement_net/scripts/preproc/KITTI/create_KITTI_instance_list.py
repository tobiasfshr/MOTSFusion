#!/usr/bin/env python3
import glob
from PIL import Image
import numpy as np
import tqdm

from refinement_net.datasets.util.Util import username
import datasets.KITTI

PATH = "/home/" + username() + "/data/KITTI_instance/"
KITTI_CODE_PATH = datasets.KITTI.__file__.replace("__init__.py", "")

with open(KITTI_CODE_PATH + "training.txt", "w") as f:
  instances = glob.glob(PATH + "instance/*.png")
  for inst in tqdm.tqdm(instances):
    x = np.array(Image.open(inst))
    ids = np.unique(x)
    im = inst.replace("/instance/", "/image_2/")
    print(im.replace(PATH, ""), inst.replace(PATH, ""), file=f, sep=" ", end="")
    for id_ in ids:
      n = (x == id_).sum()
      print(" ", id_, ":", n, file=f, sep="", end="")
    print(file=f)
