#!/usr/bin/env python3
import glob
from PIL import Image
import numpy as np
import tqdm

from refinement_net.datasets.util.Util import username
import datasets.cityscapes

PATH = "/fastwork/" + username() + "/mywork/data/cityscapes/"
CITYSCAPES_CODE_PATH = datasets.cityscapes.__file__.replace("__init__.py", "")

for subset, outfile in [("train", "training.txt"), ("val", "validation.txt")]:
  with open(CITYSCAPES_CODE_PATH + outfile, "w") as f:
    instances = glob.glob(PATH + "gtFine/" + subset + "/*/*_instanceIds.png")
    for inst in tqdm.tqdm(instances):
      x = np.array(Image.open(inst))
      ids = np.unique(x)
      im = inst.replace("/gtFine/", "/leftImg8bit/").replace("_gtFine_instanceIds", "_leftImg8bit")
      print(im.replace(PATH, ""), inst.replace(PATH, ""), file=f, sep=" ", end="")
      for id_ in ids:
        n = (x == id_).sum()
        print(" ", id_, ":", n, file=f, sep="", end="")
      print(file=f)
