import tensorflow as tf

from external.BB2SegNet.refinement_net.core import Measures
from external.BB2SegNet.refinement_net.core.Log import log
from external.BB2SegNet.refinement_net.core.Measures import measures_string_to_print, accumulate_measures, compute_measures_average
from external.BB2SegNet.refinement_net.core.Saver import Saver
from external.BB2SegNet.refinement_net.core.Timer import Timer
from external.BB2SegNet.refinement_net.core.Trainer import Trainer
from external.BB2SegNet.refinement_net.datasets.Loader import load_dataset
from external.BB2SegNet.refinement_net.forwarding.FewShotSegmentationForwarder import FewShotSegmentationForwarder
from external.BB2SegNet.refinement_net.network.Network import Network


class Engine:
  def __init__(self, config, session=None):
    self.config = config
    self.save = config.bool("save", True)
    self.task = config.string("task", "train")
    self.dataset = config.string("dataset").lower()
    self.num_epochs = config.int("num_epochs", 1000)
    self.session = self._create_session(session)
    self.global_step = tf.Variable(0, name='global_step', trainable=False)


    # need_train = True  # TODO should be self.task != "eval", but right now testnet needs to reuse variables from train
    need_train = config.bool("need_train",True)
    if need_train:
      self.train_data = load_dataset(config, "train", self.session, self.dataset)
      freeze_batchnorm = config.bool("freeze_batchnorm", False)
      print("creating trainnet...")
      self.train_network = Network(self.config, self.train_data, is_trainnet=True, freeze_batchnorm=freeze_batchnorm,
                                   name="trainnet")
    else:
      self.train_data = None
      self.train_network = None

    need_val = self.task != "train_no_val"
    if need_val:
      self.valid_data = load_dataset(config, "val", self.session, self.dataset)
      print("creating testnet...")
      reuse_variables = None if need_train else False
      # with tf.variable_scope('refinement_net'):
      self.test_network = Network(config, self.valid_data, is_trainnet=False, freeze_batchnorm=True, name="testnet",
                              reuse_variables=reuse_variables)
    else:
      self.valid_data = None
      self.test_network = None
    self.trainer = Trainer(config, self.train_network, self.test_network, self.global_step, self.session)
    self.saver = Saver(config, self.session)

    tf.global_variables_initializer().run(session=self.session)
    tf.local_variables_initializer().run(session=self.session)

    self.start_epoch = self.saver.try_load_weights()
    # self.session.graph.finalize()

  @staticmethod
  def _create_session(sess):
    if sess is None:
      sess_config = tf.ConfigProto(allow_soft_placement=True)
      sess_config.gpu_options.allow_growth = True
      # sess = tf.Session(graph=tf.Graph(),config=sess_config)
      sess = tf.Session(config=sess_config)
      # sess = tf.InteractiveSession(config=sess_config)
    return sess

  def run(self):
    if self.task in ("train", "train_no_val"):
      self.train()
    elif self.task == "eval":
      self.eval()
    elif self.task == "test_dataset_speed":
      self.test_dataset_speed()
    elif self.task == "few_shot_segmentation":
      FewShotSegmentationForwarder(self).forward()
    # elif self.task == "davis_iterative_few_shot_segmentation":
    #   DAVISIterativeFewShotSegmentationForwarder(self).forward()
    else:
      assert False, ("unknown task", self.task)

  def test_dataset_speed(self):
    n_total = self.train_data.n_examples_per_epoch()
    batch_size = self.config.int("batch_size")
    input_tensors_dict = self.train_network.input_tensors_dict
    n_curr = 0
    with Timer(message="elapsed"):
      while n_curr < n_total:
        self.session.run(input_tensors_dict)
        n_curr += batch_size
        print("{:>5}".format(n_curr), "/", n_total)

  def train(self):
    print("starting training", file=log.v1)
    for epoch in range(self.start_epoch, self.num_epochs):
      timer = Timer()
      train_measures = self.run_epoch(self.trainer.train_step, self.train_data, epoch, is_train_run=True)
      if self.valid_data is not None:
        valid_measures = self.run_epoch(self.trainer.validation_step, self.valid_data, epoch, is_train_run=False)
      else:
        valid_measures = {}
      if self.save:
        self.saver.save_model(epoch + 1)
        if hasattr(self.train_data, "save_masks"):
          self.train_data.save_masks(epoch + 1)
      elapsed = timer.elapsed()
      train_measures_str = measures_string_to_print(train_measures)
      val_measures_str = measures_string_to_print(valid_measures)
      print("epoch", epoch + 1, "finished. elapsed:", "%.5f" % elapsed, "train:", train_measures_str,
            "valid:", val_measures_str, file=log.v1)

  def eval(self):
    timer = Timer()
    measures = self.run_epoch(self.trainer.validation_step, self.valid_data, epoch=0, is_train_run=False)
    elapsed = timer.elapsed()
    print("eval finished. elapsed:", elapsed, measures, file=log.v1)

  @staticmethod
  def run_epoch(step_fn, data, epoch, is_train_run):
    n_examples_processed = 0
    n_examples_per_epoch = data.n_examples_per_epoch()
    extraction_keys = data.get_extraction_keys()
    measures_accumulated = {}
    if not is_train_run and hasattr(data, "prepare_saving_epoch_measures"):
      data.prepare_saving_epoch_measures(epoch + 1)
    while n_examples_processed < n_examples_per_epoch:
      timer = Timer()
      res = step_fn(epoch, extraction_keys=extraction_keys)
      measures = res[Measures.MEASURES]
      n_examples_processed += measures[Measures.N_EXAMPLES]
      measures_str = measures_string_to_print(compute_measures_average(measures, for_final_result=False))
      accumulate_measures(measures_accumulated, measures)
      if not is_train_run and hasattr(data, "save_epoch_measures"):
        data.save_epoch_measures(measures)
      if hasattr(data, "use_segmentation_mask"):
        data.use_segmentation_mask(res)
      elapsed = timer.elapsed()
      print("{:>5}".format(n_examples_processed), '/', n_examples_per_epoch, measures_str, "elapsed", elapsed, file=log.v5)
    measures_averaged = compute_measures_average(measures_accumulated, for_final_result=True)
    if not is_train_run and hasattr(data, "finalize_saving_epoch_measures"):
      new_measures = data.finalize_saving_epoch_measures()
      measures_averaged.update(new_measures)
    return measures_averaged
