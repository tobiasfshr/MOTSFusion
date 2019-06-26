## MOTSFusion: Multi-Object Tracking, Segmentation and Dynamic Reconstruction in 3D

### Introduction

This repository contains the corresponding source code for the paper "MOTSFusion: Multi-Object Tracking, Segmentation and Dynamic Reconstruction in 3D" [arXiv PrePrint coming soon].

### Requirements
The code was tested on:
- CUDA 9, cuDNN 7
- Tensorflow 1.13
- Python 3.6

*Note: For some external code that is required to run in the precompute script, you need different requirements (see references). Please refer to the corresponding repositories to obtain the requirements for these.*

### Instructions

At first, download the datasets in section References (Stereo image pairs) as well as the detections and adapt the config files in ./configs according to your desired setup. The file "config_default" will run 2D as well as 3D tracking on the KITTI MOTS validation set. Next, download our [pretrained segmentation network](https://drive.google.com/open?id=1Jj3VpAo7WJ-8Tvr7M3XLTA2WrUivvvNA) and extract it into './external/BB2SegNet'. Before running the main script, run:
```
python precompute.py -config ./configs/config_default
```

After this, all necessary information such as segmentations, optical flow, disparity and the corresponding pointcloud should be computed. Now you can run the tracker using:

```
python main.py -config ./configs/config_default
```
After the tracker has completed all sequences, results will be evaluated automatically.
### References
- RRC Detections: https://github.com/JunaidCS032/MOTBeyondPixels - can be downloaded along with other features in section 'Running the demo script'
- Optical flow/disparity: https://github.com/lmb-freiburg/netdef_models
- Camera pose: https://github.com/raulmur/ORB_SLAM2
- MOTS dataset and TrackRCNN detections & segmentations: https://www.vision.rwth-aachen.de/page/mots
- MOT dataset: http://www.cvlibs.net/datasets/kitti/eval_tracking.php

### Citation
Coming soon.

### License
MIT License
