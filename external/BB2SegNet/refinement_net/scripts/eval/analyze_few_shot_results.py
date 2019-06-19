#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

assert len(sys.argv) == 2
COMBINE_INSTANCE_RESULTS_OVER_TIME = False
PRINT_LARGEST_DECREASE = False
RECALL_PERCENTAGES = [50, 70, 80, 90]
VISUALIZE_DIFFERENCES = True

log = sys.argv[1]
step_ious = {}
lines = []
with open(log) as f:
  for l in f:
    if COMBINE_INSTANCE_RESULTS_OVER_TIME and l.startswith("results of"):
      step = int(l.split("after step ")[1].split(" {")[0].strip())
    elif not COMBINE_INSTANCE_RESULTS_OVER_TIME and l.startswith("forwarded/"):
      step = int(l.split("step_")[1].split("/")[0])
    else:
      continue
    #iou = float(l.split("'IoU': '")[1].split("'")[0].strip())
    iou = float(l.split("IoU: ")[1].split(",")[0].strip())
    if step not in step_ious:
      step_ious[step] = []
    step_ious[step].append(iou)
    if step == 0:
      lines.append(l)
    print(step, iou)

print("----")
for step in sorted(step_ious.keys()):
  s = str(step) + " " + str(len(step_ious[step])) + " " + str(np.mean(step_ious[step]))
  for pct in RECALL_PERCENTAGES:
    s += " recall@{}%: {}".format(pct, np.mean(np.array(step_ious[step]) > pct / 100))
  print(s)

ious = [np.array(step_ious[k]) for k in sorted(step_ious.keys())]
minlen = min([x.size for x in ious])
shortened = [x[:minlen] for x in ious]
maxed = np.maximum.reduce(shortened)
s = "oracle " + str(maxed.size) + " " + str(maxed.mean())
for pct in RECALL_PERCENTAGES:
  s += " recall@{}%: {}".format(pct, np.mean(np.array(maxed) > pct / 100))
print(s)

if PRINT_LARGEST_DECREASE:
  print("----")
  print("largest decrease in performance by fine-tuning")

  # find largest decrease in performance
  x = np.stack(ious)
  diff = x[1] - x[0]
  indices = np.argsort(diff)
  for idx in indices[:30]:
    #print(diff[idx], lines[idx], end="")
    fn = lines[idx].split(" ")[0]
    print(fn.split("/")[-3] + "/" + fn.split("/")[-1], diff[idx])

if VISUALIZE_DIFFERENCES:
  minlen = min([len(i) for i in ious])
  ious = [i[:minlen] for i in ious]
  x = np.array(ious)
  print(x.shape)
  diff = x[-1] - x[0]
  plt.hist(diff, bins=30)
  plt.show()
