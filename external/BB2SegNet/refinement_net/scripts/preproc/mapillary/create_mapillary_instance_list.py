#!/usr/bin/env python3
import glob
from PIL import Image
import numpy as np

from refinement_net.datasets.util.Util import username
import datasets.Mapillary

PATH = "/fastwork/" + username() + "/mywork/data/mapillary_quarter/"
MAPILLARY_CODE_PATH = datasets.Mapillary.__file__.replace("__init__.py", "")

for subset in ["training", "validation"]:
  with open(MAPILLARY_CODE_PATH + subset, "w") as f:
    instances = glob.glob(PATH + subset + "/instances/*.png")
    for inst in instances:
      x = np.array(Image.open(inst))
      ids = np.unique(x)
      im = inst.replace("/instances/", "/images/").replace(".png", ".jpg")
      print(im.replace(PATH, ""), inst.replace(PATH, ""), file=f, sep=" ", end="")
      for id_ in ids:
        n = (x == id_).sum()
        print(" ", id_, ":", n, file=f, sep="", end="")
      print(file=f)
