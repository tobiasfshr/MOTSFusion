from abc import abstractmethod
from random import shuffle

from external.BB2SegNet.refinement_net.datasets.Dataset import DataKeys
from external.BB2SegNet.refinement_net.datasets.FeedDataset import FeedDataset
from external.BB2SegNet.refinement_net.datasets.util.Util import username
from external.BB2SegNet.refinement_net.core.Log import log

NUM_CLASSES = 2
DEFAULT_DATA_KEYS = (DataKeys.IMAGES, DataKeys.SEGMENTATION_LABELS)


class FewShotSegmentationDataset(FeedDataset):
  def __init__(self, config, subset, default_data_dir_suffix, data_keys_to_use, num_classes=2):
    super().__init__(config, subset, data_keys_to_use, num_classes)
    self._data_dir = config.dir("dataset_path", "/fastwork/" + username() + "/mywork/data/" + default_data_dir_suffix)
    self._video_idx = None
    self._object_idx = None
    self._curr_video_data = None
    self._keys_to_sample = []
    # TODO: maybe later share the cache between the subsets
    self._video_data_cache = {}
    self._video_tags = self._read_video_tags()

  # reasonable default implementation which might be overridden by subclasses
  def _read_video_tags(self):
    image_set = self.config.string("image_set", "all.txt")
    image_set_filename = self._data_dir + "/ImageSets/" + image_set
    with open(image_set_filename) as f:
      video_tags = [l.strip() for l in f if len(l.strip()) > 0]
    return video_tags

  def get_feed_dict_for_next_step(self):
    assert self._video_idx is not None
    assert self._object_idx is not None
    feed_dict = {}
    for idx in range(self._batch_size):
      example = self._get_next_single_example()
      for data_key in self._data_keys_to_use:
        assert data_key in example, data_key
        feed_dict[self._placeholders[data_key][idx]] = example[data_key]
    return feed_dict

  def get_placeholders(self, key):
    return self._placeholders[key]

  # reasonable default implementation which might be overridden by subclasses
  def _get_next_single_example(self):
    obj_data = self._curr_video_data[self._object_idx]
    assert obj_data is not None

    # if all keys are used up, reset the keys to cover the full dataset
    if len(self._keys_to_sample) == 0:
      if hasattr(obj_data, "keys"):
        self._keys_to_sample = list(obj_data.keys())
      else:
        assert isinstance(obj_data, list)
        self._keys_to_sample = list(range(len(obj_data)))
      assert len(self._keys_to_sample) > 0
      # shuffle the keys if we are training
      if self.subset == "train":
        shuffle(self._keys_to_sample)
      else:
        self._keys_to_sample.sort(reverse=True)

    frame_idx = self._keys_to_sample.pop()
    data_dict = obj_data[frame_idx]
    return data_dict

  @abstractmethod
  def _load_video_data(self, idx):
    pass

  def n_examples_per_epoch(self):
    assert self._curr_video_data is not None
    assert self._object_idx is not None
    return len(self._curr_video_data[self._object_idx])

  def n_videos(self):
    return len(self._video_tags)

  def set_video_idx(self, idx):
    assert idx < self.n_videos()
    self._keys_to_sample = []
    if idx != self._video_idx:
      self._video_idx = idx
      if idx in self._video_data_cache:
        self._curr_video_data = self._video_data_cache[idx]
      else:
        print("loading data for video", idx, file=log.v5)
        # it seems that keeping everything in cache takes way too much memory and is also useless, so let's clear
        # the cache when changing the sequence for now
        self._video_data_cache = {}
        self._curr_video_data = self._load_video_data(idx)
        self._video_data_cache = {idx: self._curr_video_data}

  def get_video_tag(self):
    assert self._video_idx is not None
    return self._video_tags[self._video_idx]

  def set_object_idx_in_video(self, idx):
    assert idx < self.n_objects_in_video()
    self._keys_to_sample = []
    if idx != self._object_idx:
      self._object_idx = idx

  def get_object_idx_in_video(self):
    assert self._object_idx is not None
    return self._object_idx

  def n_objects_in_video(self):
    assert self._curr_video_data is not None
    return len(self._curr_video_data)
