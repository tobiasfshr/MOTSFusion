# netdef_models
Repository for different network models related to flow/disparity from the following papers: 

**NOTE: We only provide deployment code for these networks. We do not publish any training code and also do not offer support about questions for training networks.**

* **Occlusions, Motion and Depth Boundaries with a Generic Network for Disparity, Optical Flow or Scene Flow**  
(E. Ilg and T. Saikia and M. Keuper and T. Brox published at ECCV 2018)  [[paper]](http://lmb.informatik.uni-freiburg.de/Publications/2018/ISKB18) [[video]](https://www.youtube.com/watch?v=SwOdSaBRysI)

* **Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow**  
(E. Ilg and Ö. Cicek and S. Galesso and A. Klein and O. Makansi and F. Hutter and T. Brox published at ECCV 2018)  [[paper]](https://lmb.informatik.uni-freiburg.de/Publications/2018/ICKMB18/) [[video]](https://www.youtube.com/watch?v=HvyovWSo8uE)


## Setup
* Install [tensorflow (1.4)](https://www.tensorflow.org/install/) (pip3 install tensorflow-gpu==1.4)
* Compile and install [lmbspecialops](https://github.com/lmb-freiburg/lmbspecialops/tree/eccv18). Please use the branch `eccv18` instead of `master`
* Install [netdef_slim](https://github.com/lmb-freiburg/netdef_slim)
* Clone this repository

## Running networks

* Change your directory to the network directory (Eg: FlowNet3)
* Run download_snapshots.sh. This takes a while to download all snapshots
* Now you should be ready to run the networks. Change your directory to a network type (Eg: css).
  Use the following command to test the network on an image pair:
  `python3 controller.py eval image0_path image1_path out_dir`

## Output formats

The networks are executed using the controller.py scripts in the respective folders. Just running this controller will produce several output files in a folder (note that you can also obtain this output just as numpy arrays and write it to some custom files; see next section). 

For optical flow we use the standard `.flo` format. 
The other modalities use a custom binary format called `.float3`. To read `.float3` files to numpy arrays, please use the
netdef_slim.utils.io module.

Example usage:
```
from netdef_slim.utils.io import read 
occ_file = 'occ.float3'
occ_data = read(occ_file) # returns a numpy array

# to visualize
import matplotlib.pyplot as plt
plt.imshow(occ_data[:,:,0])

```
## Controller eval
The eval method of the controller writes to the disk by default.
To avoid writing to disk, create a Controller object and use the `eval` method available in the `net_actions` member variable.
This can be useful if you want to process the output of our networks in memory and not incur additional disk I/O.

Example usage:
```
import netdef_slim as nd
nd.load_module('FlowNet3/css/controller.py')
c = Controller() 
out = c.net_actions.eval(img0, img1)
# out is an OrderedDict with the following structure
#OrderedDict(['flow[0].fwd',     np.array[...],
              'occ[0].fwd',      np.array[...],
              'occ_soft[0].fwd', np.array[...],
              'mb[0].fwd',       np.array[...],
              'mb_soft[0].fwd',  np.array[...],
              ])       

```
## License

netdef_models is under the [GNU General Public License v3.0](LICENSE.txt)
