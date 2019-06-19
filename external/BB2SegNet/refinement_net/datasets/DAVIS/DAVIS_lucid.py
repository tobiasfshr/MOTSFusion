import tensorflow as tf
from PIL import Image
import numpy as np

from external.BB2SegNet.refinement_net.datasets.Loader import register_dataset
from external.BB2SegNet.refinement_net.datasets.Mapillary.MapillaryLike_instance import MapillaryLikeInstanceDataset
from external.BB2SegNet.refinement_net.datasets.util.Util import username

# DEFAULT_PATH = "/home/luiten/vision/PReMVOS/data/"
# LIST_PATH_ROOT = "/home/luiten/vision/PReMVOS/refinement_finetuning/"
# DEFAULT_PATH = "/home/luiten/vision/youtubevos/ytvos_data/paris/ldd/"
# LIST_PATH_ROOT = "/home/luiten/vision/youtubevos/ytvos_data/paris/ref_weights/"
# DEFAULT_PATH = "/home/luiten/vision/PReMVOS/home_data/"
# LIST_PATH_ROOT = "/home/luiten/vision/youtubevos/DAVIS/"
# DEFAULT_PATH = "/home/luiten/vision/youtubevos/ytvos_data/train/"
# LIST_PATH_ROOT = "/home/luiten/vision/youtubevos/ytvos_data/train/"

# DEFAULT_PATH = "/home/luiten/vision/youtubevos/ytvos_data/together/generated/"
# LIST_PATH_ROOT = "/home/luiten/vision/youtubevos/ytvos_data/together/generated/"

DEFAULT_PATH = "/home/luiten/vision/youtubevos/DAVIS/davis_together/"
LIST_PATH_ROOT = "/home/luiten/vision/youtubevos/DAVIS/davis_together/"

NAME = "davis_lucid"

@register_dataset(NAME)
class DAVISLucidDataset(MapillaryLikeInstanceDataset):
  def __init__(self, config, subset):
    davis_sequence = config.string("model", '')
    data_list_path = LIST_PATH_ROOT + davis_sequence + '/'
    super().__init__(config, subset, NAME, DEFAULT_PATH, data_list_path, 100, cat_ids_to_use=None)

  def load_annotation(self, img, img_filename, annotation_filename):
    annotation_filename_without_id = tf.string_split([annotation_filename], ':').values[0]
    #ann_data = tf.read_file(annotation_filename_without_id)
    #ann = tf.image.decode_png(ann_data, dtype=tf.uint16, channels=1)
    ann, = tf.py_func(load_ann_with_colormap, [annotation_filename_without_id], [tf.uint8])
    ann.set_shape(img.get_shape().as_list()[:-1] + [1])
    ann = self.postproc_annotation(annotation_filename, ann)
    return ann


def load_ann_with_colormap(filename):
  filename = filename.decode("utf-8")
  return np.array(Image.open(filename))[..., np.newaxis]
