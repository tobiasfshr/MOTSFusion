#!/usr/bin/env python3
import glob
from PIL import Image
import numpy as np
import tqdm

from refinement_net.datasets.util.Util import username
import datasets.DAVIS

PATH = "/home/luiten/vision/PReMVOS/data/first/bike-trial/"
DAVIS_CODE_PATH = datasets.DAVIS.__file__.replace("__init__.py", "")

for subset, outfile in [("train", "training.txt"), ("val", "validation.txt")]:
  with open(DAVIS_CODE_PATH + outfile, "w") as f:
    instances = glob.glob(PATH + "lucid_data_dreaming/*.png")
    for inst in tqdm.tqdm(instances):
      x = np.array(Image.open(inst))
      ids = np.unique(x)
      im = inst.replace(".png", ".jpg")
      print(im.replace(PATH, ""), inst.replace(PATH, ""), file=f, sep=" ", end="")
      for id_ in ids[1:]:
        n = (x == id_).sum()
        print(" ", id_, ":", n, file=f, sep="", end="")
      print(file=f)
