## MOTSFusion: Track to Reconstruct and Reconstruct to Track
![Method Overview](https://github.com/tobiasfshr/MOTSFusion/blob/master/imgs/overview.png)

### Introduction

This repository contains the corresponding source code for the paper "Track to Reconstruct and Reconstruct to Track" [arXiv PrePrint](http://arxiv.org/abs/1910.00130).

### News
This paper has been accepted as both a conference paper at ICRA 2020, as well as being accepted as a journal paper in Robotics and Automation Letters (RA-L)!!!

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

### Results
We release our results on the MOTS test set [cars](https://drive.google.com/open?id=1v6AIJ2qRkHKLTnR7Sma3QA3Be6VcQJ2U)
[pedestrians](https://drive.google.com/file/d/1yqkjqt-0t8zVU7jxwbon2TWyhXXmbMAt/view?usp=sharing)

As well as the detections  and segmentations ([test-RRC](https://drive.google.com/open?id=1QmArTCHaxS2a9jciGBqA6LAQQ4bcPeKE), [test-TRCNN](https://drive.google.com/open?id=14YLMwTDi2gpUVOSgDiNxMLrTgvX_Nb6Q), [train/val-RRC](https://drive.google.com/open?id=194Yj_L9_cc5Yio-Khk6DGFOUQ6RS7PvV), [train/val-TRCNN](https://drive.google.com/open?id=1Rb63G4j6lap2Zk4zKOlYX_YJozAdduPD)).

### References
- RRC Detections: https://github.com/JunaidCS032/MOTBeyondPixels - can be downloaded along with other features in section 'Running the demo script'
- Optical flow/disparity: https://github.com/lmb-freiburg/netdef_models
- Camera pose: https://github.com/raulmur/ORB_SLAM2
- MOTS dataset and TrackRCNN detections & segmentations: https://www.vision.rwth-aachen.de/page/mots
- MOT dataset: http://www.cvlibs.net/datasets/kitti/eval_tracking.php

### Citation
```
@article{luiten2019track,
  title={Track to Reconstruct and Reconstruct to Track},
  author={Luiten, Jonathon and Fischer, Tobias and Leibe, Bastian},
  journal={arXiv:1910.00130},
  year={2019}
}
```
### License
MIT License
