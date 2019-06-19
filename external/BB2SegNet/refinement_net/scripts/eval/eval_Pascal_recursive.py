import os
import sys

import numpy as np
import tensorflow as tf

from refinement_net.core import Extractions, Measures
from refinement_net.core.Config import Config
from refinement_net.core.Engine import Engine
from refinement_net.core.Log import log
from refinement_net.core.Measures import measures_string_to_print, accumulate_measures, compute_measures_average
from refinement_net.core.Saver import Saver
from refinement_net.core.Timer import Timer
from refinement_net.core.Trainer import Trainer
from refinement_net.datasets import DataKeys
from refinement_net.datasets.PascalVOC.PascalVOC_context_clicks import PascalVOCContextClicksDataset
from refinement_net.datasets.PascalVOC.PascalVOC_masked_dios import PREV_MASK
from refinement_net.datasets.util.DistanceTransform import get_distance_transform
from refinement_net.datasets.util.Util import decodeMask
from refinement_net.network.Network import Network
from refinement_net.scripts.eval.Datasets.EvalCOCO import EvalCOCODataset
from refinement_net.scripts.eval.Datasets.EvalDAVIS import EvalDAVISDataset, EvalDAVISContextClicks
from refinement_net.scripts.eval.Datasets.EvalGrabcut import EvalGrabcutDataset
from refinement_net.scripts.eval.Datasets.EvalOSVOSWorst import OSVOSWorst
from refinement_net.scripts.eval.Datasets.EvalPascalMasked import EvalPascalMaskedDataset, CURRENT_CLICK

NAME = "pascal_recursive_eval"


def load_eval_dataset(config, subset):
  name = config.string("dataset").lower()
  if name == "pascalvoc_masked_dios":
    return EvalPascalMaskedDataset(config, subset)
  elif name == "pascalvoc_context_clicks":
    dataset = PascalVOCContextClicksDataset(config, subset)
    dataset.initialise_with_single_click = True
    return dataset
  elif name == "davis":
    return EvalDAVISDataset(config, subset)
  elif name == "davis_context_clicks":
    return EvalDAVISContextClicks(config, subset)
  elif name == "coco_masked":
    return EvalCOCODataset(config, subset)
  elif name == "grabcut":
    return EvalGrabcutDataset(config, subset)
  elif name == "osvos_worst":
    return OSVOSWorst(config, subset)
  else:
    assert False, ("unknown dataset", name)


def extract(extractions, key):
  if key not in extractions:
    return None
  val = extractions[key]

  # for now assume we only use 1 gpu for forwarding
  assert len(val) == 1, len(val)
  val = val[0]

  # for now assume, we use a batch size of 1 for forwarding
  assert val.shape[0] == 1, val.shape[0]
  val = val[0]

  return val


class EvalPascalRecursive(Engine):
  def __init__(self, config, session=None):
    self.config = config
    self.save = config.bool("save", True)
    self.task = config.string("task", "train")
    self.dataset = config.string("dataset").lower()
    self.num_epochs = config.int("num_epochs", 1000)
    self.session = self._create_session(session)
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.iou_threshold = config.float("iou_threshold", 0.85)
    self.avg_clicks = {}

    # TODO
    need_train = True
    if need_train:
      self.train_data = load_eval_dataset(config, "train")
      freeze_batchnorm = config.bool("freeze_batchnorm", False)
      print("creating trainnet...", file=log.v1)
      self.train_network = Network(self.config, self.train_data, is_trainnet=True, freeze_batchnorm=freeze_batchnorm,
                                   name="trainnet")
    else:
      self.train_data = None
      self.train_network = None

    need_val = self.task != "train_no_val"
    if need_val:
      self.valid_data = load_eval_dataset(config, "valid")
      print("creating testnet...", file=log.v1)
      self.test_network = Network(config, self.valid_data, is_trainnet=False, freeze_batchnorm=True, name="testnet")
    else:
      self.valid_data = None
      self.test_network = None
    self.trainer = Trainer(config, self.train_network, self.test_network, self.global_step, self.session)
    self.saver = Saver(config, self.session)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    self.start_epoch = self.saver.try_load_weights()
    self.session.graph.finalize()
    self.recursive_rounds = config.int("recursive_rounds", 20)
    self.avg_clicks={}
    self.locality = config.bool("locality", False)
    self.locality_threshold = config.int("locality_threshold", 100)
    self.monotonicity = config.bool("monotonicity", False)

    if self.locality:
      print("Using locality postprocessing...", log.v1)
    if self.monotonicity:
      print("Using monotonicity postprocessing...", log.v1)

  def eval(self):
    for round in range(self.recursive_rounds):
      timer_round = Timer()
      n_examples_processed = 0
      n_examples_per_epoch = self.valid_data.n_examples_per_epoch()
      measures_accumulated = {}

      while n_examples_processed < n_examples_per_epoch:
        timer = Timer()
        extraction_keys = [Extractions.SEGMENTATION_MASK_INPUT_SIZE, DataKeys.IMAGE_FILENAMES, DataKeys.RAW_IMAGES,
                           DataKeys.INPUTS, DataKeys.SEGMENTATION_LABELS, Extractions.SEGMENTATION_POSTERIORS]
        res = self.trainer.validation_step(round, extraction_keys=extraction_keys)
        res = self.postprocess(res)
        measures = res[Measures.MEASURES]
        n_examples_processed += measures[Measures.N_EXAMPLES]
        measures_str = measures_string_to_print(compute_measures_average(measures, for_final_result=False))
        # add tag to the measures so that it is printed (postprocessing for davis)
        measures_str = self.add_tag(measures_str, round, res[Extractions.EXTRACTIONS])
        accumulate_measures(measures_accumulated, measures)
        elapsed = timer.elapsed()
        self.valid_data.use_segmentation_mask(res)
        self.update_iou(measures, res[Extractions.EXTRACTIONS], round + 1)
        print("{:>5}".format(n_examples_processed), '/', n_examples_per_epoch, measures_str, "elapsed", elapsed,
              file=log.v5)
      measures_averaged = compute_measures_average(measures_accumulated, for_final_result=True)
      print("Click ", round, " eval finished. elapsed:", timer_round.elapsed(), measures_averaged, file=log.v1)

    print("Samples: " + str(len(self.avg_clicks.keys())), file=log.v1)
    print("Avg clicks for", self.iou_threshold * 100, " % IOU is ", np.average(list(self.avg_clicks.values())), file=log.v1)

  def add_tag(self, measures, round, extractions):
    img_filename = extract(extractions, DataKeys.IMAGE_FILENAMES)
    basename = img_filename.decode("utf-8").split("/")[-2:]
    img_filename = basename[0] + basename[1]
    measures += "  {" + img_filename + "}  " + "{clicks: " + str(round) + "}"
    return measures

  def update_iou(self, measures, extractions, click):
    img_filename = extract(extractions, DataKeys.IMAGE_FILENAMES)
    measures_avg =compute_measures_average(measures, for_final_result=False)
    iou = float(measures_avg[Measures.IOU])

    if (iou >= self.iou_threshold or click == self.recursive_rounds) and img_filename not in self.avg_clicks:
      self.avg_clicks[img_filename] = click

  def postprocess(self, res):
    result_measures = res[Measures.MEASURES]
    img_filename = extract(res[Extractions.EXTRACTIONS], DataKeys.IMAGE_FILENAMES)
    prediction = extract(res[Extractions.EXTRACTIONS], Extractions.SEGMENTATION_MASK_INPUT_SIZE)
    prev_mask =  decodeMask(self.valid_data.previous_epoch_data[img_filename][PREV_MASK]) \
      if img_filename in self.valid_data.previous_epoch_data and \
         PREV_MASK in self.valid_data.previous_epoch_data[img_filename] else None
    label = extract(res[Extractions.EXTRACTIONS], DataKeys.SEGMENTATION_LABELS)

    if prev_mask is not None:
      measures = result_measures.copy()
      current_click = self.valid_data.previous_epoch_data[img_filename][CURRENT_CLICK]
      dt = get_distance_transform(current_click, label)[:, :, 0]
      if self.locality:
        prediction = np.where(dt > self.locality_threshold, prev_mask, prediction)
        measures = Measures.compute_measures_for_binary_segmentation_single_image(prediction, label[:, :, 0])
      if self.monotonicity:
        if len(current_click) > 0:
          if label[current_click[0][0], current_click[0][1]] == 1:
            # positive click
            mask = prev_mask - prediction
          else:
            mask = prediction - prev_mask
          prediction = np.where(mask == 1, prev_mask, prediction)
          measures = Measures.compute_measures_for_binary_segmentation_single_image(prediction, label[:, :, 0])

      for k, v in measures.items():
        result_measures[k] = v

    res[Measures.MEASURES] = result_measures
    res[Extractions.EXTRACTIONS][Extractions.SEGMENTATION_MASK_INPUT_SIZE][0] = [prediction]

    return res

def init_log(config):
  log_dir = config.dir("log_dir", "logs")
  model = config.string("model")
  filename = log_dir + model + ".log"
  verbosity = config.int("log_verbosity", 3)
  log.initialize([filename], [verbosity], [])


def main(_):
  assert len(sys.argv) == 2, "usage: main.py <config>"
  config_path = sys.argv[1]
  assert os.path.exists(config_path), config_path
  try:
    config = Config(config_path)
  except ValueError as e:
    print("Malformed config file:", e)
    return -1
  init_log(config)
  # dump the config into the log
  print(open(config_path).read(), file=log.v4)
  engine = EvalPascalRecursive(config)
  engine.run()


if __name__ == '__main__':
  os.chdir("/home/mahadevan/vision/savitar2/")
  tf.app.run(main)
