#!/usr/bin/env python
import matplotlib.pyplot as plt
import sys


def doit(fn, tag):
  train = []
  val = []
  with open(fn) as f:
    for l in f:
      if "finished" in l and "epoch" in l:
        sp = [x.replace("{", "").replace("}", "").replace(",", "") for x in l.split()]
        indices = [i + 1 for i, x in enumerate(sp) if x == tag + ":"]
        assert len(indices) == 2, sp
        tr = float(sp[indices[0]])
        va = float(sp[indices[1]])
        train.append(tr)
        val.append(va)
  print(train, val)
  x_axis = range(1, len(train) + 1)
  plt.plot(x_axis, train, label="train")
  plt.plot(x_axis, val, label="val")
  plt.legend()
  plt.title(fn + " " + tag)

assert len(sys.argv) == 2
doit(sys.argv[1], "loss")
plt.figure()
doit(sys.argv[1], "IoU")
plt.show()
