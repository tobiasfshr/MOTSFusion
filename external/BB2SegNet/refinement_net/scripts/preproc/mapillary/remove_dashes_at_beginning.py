#!/usr/bin/env python3

import glob
import os

from refinement_net.datasets.util.Util import username

PATH = "/fastwork/" + username() + "/mywork/data/mapillary/"
files = glob.glob(PATH + "training/*/-*.png") + glob.glob(PATH + "training/*/-*.jpg") + \
        glob.glob(PATH + "validation/*/-*.png") + glob.glob(PATH + "validation/*/-*.jpg")
with open(PATH + "renamed.txt", "w") as f_out:
  for f in files:
    idx_begin = f.rfind("/-")
    idx_end = idx_begin + 1
    while f[idx_end] == "-":
      idx_end += 1
    f_new = f[:idx_begin+1] + f[idx_end:]
    print(f, "->", f_new)
    os.rename(f, f_new)
    print(f, f_new, file=f_out)
