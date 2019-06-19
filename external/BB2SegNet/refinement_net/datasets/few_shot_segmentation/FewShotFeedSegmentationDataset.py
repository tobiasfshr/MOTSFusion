from abc import abstractmethod
from random import shuffle
import numpy as np

from external.BB2SegNet.refinement_net.datasets.Dataset import DataKeys
from external.BB2SegNet.refinement_net.datasets.FeedDataset import FeedDataset
from external.BB2SegNet.refinement_net.datasets.util.Util import username
from external.BB2SegNet.refinement_net.core.Log import log
from external.BB2SegNet.refinement_net.datasets.Loader import register_dataset

NUM_CLASSES = 2
DEFAULT_DATA_KEYS = (DataKeys.IMAGES, DataKeys.SEGMENTATION_LABELS)
DATA_KEYS_TO_USE = (DataKeys.IMAGES, DataKeys.SEGMENTATION_LABELS, DataKeys.BBOXES_y0x0y1x1, DataKeys.IMAGE_FILENAMES, DataKeys.OBJ_TAGS)


@register_dataset("jono_vos_feed")
class FewShotFeedSegmentationDataset(FeedDataset):
  def __init__(self, config, subset):
    data_keys_to_use = DATA_KEYS_TO_USE
    num_classes = 2
    super().__init__(config, subset, data_keys_to_use, num_classes)

  def get_feed_dict_for_next_step(self,image_data,bbox_idx):
    feed_dict = {}
    for idx in range(self._batch_size):
      example = image_data[bbox_idx]
      for data_key in self._data_keys_to_use:
        assert data_key in example, data_key
        feed_dict[self._placeholders[data_key][idx]] = example[data_key]
    return feed_dict

  def get_placeholders(self, key):
    return self._placeholders[key]

  def set_up_data_for_image(self,image,boxes):
    obj_data = {}
    image = image/255
    label = np.zeros(image.shape[:2] + (1,), dtype=np.uint8)
    for box_id,box in enumerate(boxes):
      x0, y0, x1, y1 = box
      x1 = x1 + x0
      y1 = y1 + y0
      bbox = [y0, x0, y1, x1]
      obj_data[box_id] = {DataKeys.IMAGES: image,
                           DataKeys.SEGMENTATION_LABELS: label,
                           DataKeys.IMAGE_FILENAMES: "",
                           DataKeys.BBOXES_y0x0y1x1: bbox,
                           DataKeys.OBJ_TAGS: str(box_id)}
    if len(obj_data) > 0:
      return obj_data
    else:
      return None