import glob
# IMAGE_DIR = "/work2/luiten/data/DAVIS/train-val/val17/"
# FLOW_DIR = "/home/luiten/vision/flownet2/"
# FLOW_OUT_DIR = "/work2/luiten/data/DAVIS/train-val/flow/"

# IMAGE_DIR = "/work2/luiten/data/DAVIS/test-challenge/JPEGImages/480p/"
# FLOW_DIR = "/home/luiten/vision/flownet2/"
# FLOW_OUT_DIR = "/work2/luiten/data/DAVIS/test-challenge/flow/"

# IMAGE_DIR = "/work2/luiten/data/DAVIS/test-dev/JPEGImages/480p/"
# FLOW_DIR = "/home/luiten/vision/flownet2/"
# FLOW_OUT_DIR = "/work2/luiten/data/DAVIS/test-dev/flow/"

IMAGE_DIR = "/work2/luiten/data/DAVIS/2016/DAVIS-data/DAVIS/train/"
FLOW_DIR = "/home/luiten/vision/flownet2/"
FLOW_OUT_DIR = "/work2/luiten/data/DAVIS/train/flow/"

new_file= open(FLOW_DIR + "filelist.txt","w+")
folders = glob.glob(IMAGE_DIR+"*/")
for folder in folders:
  folder_solo = folder.split('/')[-2]
  files = glob.glob(folder+"/*.jpg")
  nums = [int(f.split('/')[-1].split('.jpg')[0]) for f in files]
  nums.sort()
  firsts = nums[:-1]
  lasts = nums[1:]
  first_fs = [IMAGE_DIR+folder_solo+'/%05d.jpg'%f for f in firsts]
  last_fs = [IMAGE_DIR+folder_solo+'/%05d.jpg'%l for l in lasts]
  to_saves_1 = [FLOW_OUT_DIR + folder_solo+'-%05d-%05d.flo'%(f,l) for f,l in zip(firsts,lasts)]
  to_saves_2 = [FLOW_OUT_DIR + folder_solo + '-%05d-%05d.flo' % (l, f) for f, l in zip(firsts, lasts)]
  texts_1 = [f + ' ' + ' ' + l + ' ' + t+'\n' for f, l, t in zip(first_fs, last_fs, to_saves_1)]
  texts_2 = [l + ' ' + ' ' + f + ' ' + t+'\n' for f, l, t in zip(first_fs, last_fs, to_saves_2)]
  for t in texts_1:
    new_file.write(t)
  for t in texts_2:
    new_file.write(t)
