from math import ceil


from external.BB2SegNet.refinement_net.forwarding.Forwarder import Forwarder
from external.BB2SegNet.refinement_net.core import Measures, Extractions
from external.BB2SegNet.refinement_net.core.Timer import Timer
from external.BB2SegNet.refinement_net.datasets import DataKeys
# import DataKeys
from external.BB2SegNet.refinement_net.core.Measures import accumulate_measures, measures_string_to_print, compute_measures_average
# from refinement_net.datasets.util.Util import username
import numpy as np
import json
import os
from pycocotools.mask import encode

class FewShotSegmentationForwarder(Forwarder):
  def __init__(self, engine):
    super().__init__(engine)
    self.forward_interval = self.config.int("forward_interval", 99999999)
    self.forward_initial = self.config.bool("forward_initial", False)
    self.n_finetune_steps = self.config.int("n_finetune_steps", 0)
    self.model_name = self.config.string("model")
    self.video_ids = self.config.int_list("video_ids", [])
    self.object_ids = self.config.int_key_dict("object_ids", {})
    self.print_per_object_stats = self.config.bool("print_per_object_stats", True)

  def forward(self):
    video_ids = range(self.val_data.n_videos()) if len(self.video_ids) == 0 else self.video_ids
    for video_idx in video_ids:
      self.val_data.set_video_idx(video_idx)
      if self.train_data is not None:
        self.train_data.set_video_idx(video_idx)
      tag = self.val_data.get_video_tag()
      n_objects = self.val_data.n_objects_in_video()
      obj_ids = range(n_objects) if video_idx not in self.object_ids else self.object_ids[video_idx]
      for obj_idx in obj_ids:
        with Timer("object " + str(obj_idx) if self.print_per_object_stats else None):
          self.val_data.set_object_idx_in_video(obj_idx)
          if self.train_data is not None:
            self.train_data.set_object_idx_in_video(obj_idx)
          # reset weights and optimizer for next video/object (only if we actually do fine-tuning)
          if self.n_finetune_steps > 0:
            self.saver.try_load_weights()
            self.trainer.reset_optimizer()
          # print("finetuning on", tag + ":" + str(obj_idx), file=log.v4)
          self._forward_object(video_idx, obj_idx, tag)

  def _forward_object(self, video_idx, obj_idx, tag):
    if self.forward_initial or self.n_finetune_steps == 0:
      self.val_data.set_video_idx(video_idx)
      self.val_data.set_object_idx_in_video(obj_idx)
      self._forward(0)

    if self.n_finetune_steps == 0:
      return
    forward_interval = self.forward_interval
    if forward_interval > self.n_finetune_steps:
      forward_interval = self.n_finetune_steps
      n_partitions = 1
    else:
      n_partitions = int(ceil(self.n_finetune_steps / forward_interval))
    for i in range(n_partitions):
      start_step = forward_interval * i
      end_step = min(start_step + forward_interval, self.n_finetune_steps)
      self._finetune(tag + ":" + str(obj_idx), start_step, end_step)
      self.val_data.set_video_idx(video_idx)
      self.val_data.set_object_idx_in_video(obj_idx)
      self._forward(end_step)

  def _finetune(self, tag, start_step, end_step):
    for step in range(start_step, end_step):
      feed_dict = self.train_data.get_feed_dict_for_next_step()
      step_res = self.trainer.train_step(epoch=step, feed_dict=feed_dict)
      step_measures = step_res[Measures.MEASURES]
      loss = step_measures[Measures.LOSS]
      iou = step_measures[Measures.IOU]
      try:
        placeholders = self.train_data.get_placeholders(DataKeys.IMAGE_FILENAMES)
        filenames = [feed_dict[ph] for ph in placeholders]
        extended_tag = tag + " (" + ",".join(filenames) + ")"
      except KeyError:
        extended_tag = tag
      print("finetuning on", extended_tag, step, "/", end_step, "loss:", loss, " iou:", iou)

  def _forward(self, step_num):
    video_tag = self.val_data.get_video_tag()
    time_step_id = self.val_data.get_object_idx_in_video()
    img_filename = self.val_data._curr_video_data[time_step_id][0][DataKeys.IMAGE_FILENAMES]
    timestep_name = img_filename.split('/')[-1].replace('.jpg','')
    if self.print_per_object_stats:
      print("forwarding on", video_tag + ":" + str(time_step_id),"after step", step_num, "proposals:",len(self.val_data._curr_video_data[time_step_id]))
    measures = {}

    # Get proposals:
    proposals_dir = self.config.string("bb_input_dir", None)
    output_dir = self.config.string("output_dir", None)

    curr = video_tag + timestep_name.zfill(5) + ".json"
    in_dir = proposals_dir + curr
    out_dir = output_dir + curr
    with open(in_dir, "r") as f:
      proposals = json.load(f)

    for idx in range(self.val_data.n_examples_per_epoch()):
      feed_dict = self.val_data.get_feed_dict_for_next_step()
      # step_res = self.trainer.validation_step(feed_dict=feed_dict, extraction_keys=[
      #   Extractions.SEGMENTATION_POSTERIORS_ORIGINAL_SIZE, Extractions.SEGMENTATION_MASK_ORIGINAL_SIZE,
      #   DataKeys.IMAGE_FILENAMES, DataKeys.RAW_IMAGES, DataKeys.OBJ_TAGS])
      step_res = self.trainer.validation_step(feed_dict=feed_dict, extraction_keys=[
        Extractions.SEGMENTATION_POSTERIORS_ORIGINAL_SIZE, Extractions.SEGMENTATION_MASK_ORIGINAL_SIZE, DataKeys.OBJ_TAGS])

      extractions = step_res[Extractions.EXTRACTIONS]
      step_measures = step_res[Measures.MEASURES]
      accumulate_measures(measures, step_measures)

      def extract(key):
        if key not in extractions:
          return None
        val = extractions[key]

        # for now assume we only use 1 gpu for forwarding
        assert len(val) == 1, len(val)
        val = val[0]

        # # for now assume, we use a batch size of 1 for forwarding
        assert val.shape[0] == 1, val.shape[0]
        val = val[0]

        return val

      predicted_segmentation = extract(Extractions.SEGMENTATION_MASK_ORIGINAL_SIZE)
      obj_tag = extract(DataKeys.OBJ_TAGS)
      posteriors = extract(Extractions.SEGMENTATION_POSTERIORS_ORIGINAL_SIZE)
      # img_filename = extract(DataKeys.IMAGE_FILENAMES)
      # img = extract(DataKeys.RAW_IMAGES)

      ########### New code for saving json directly
      # Insert mask into proposals
      obj_tag = int(obj_tag.decode('utf-8'))
      mask = predicted_segmentation.astype("uint8") * 255
      encoded_mask = encode(np.asfortranarray(mask))
      encoded_mask['counts'] = encoded_mask['counts'].decode("utf-8")
      proposals[obj_tag]["segmentation"] = encoded_mask

      conf_scores = posteriors.copy()
      conf_scores[predicted_segmentation==0] = 1-posteriors[predicted_segmentation==0]
      conf_scores = 2*conf_scores - 1
      conf_score = conf_scores[:].mean()
      proposals[obj_tag]["conf_score"] = str(conf_score)

    create_out_dir = '/'.join(out_dir.split('/')[:-1])
    if not os.path.exists(create_out_dir):
      os.makedirs(create_out_dir)
    with open(out_dir, 'w') as f:
      json.dump(proposals, f)

      ########### Old code for saving image out

      # # convert from bytes to python3 string
      # img_filename = img_filename.decode('utf-8')
      # obj_tag = obj_tag.decode('utf-8')
      # tag = self.val_data.get_video_tag()
      # sp = img_filename.split("/")
      # assert tag == sp[-2]
      # filename = sp[-1].replace(".jpg", ".png")
      # filename2 = filename.replace('.png','.posterior.png')
      #
      # DAVIS_forwarding = True
      # if DAVIS_forwarding:
      #   timestep = filename.split('.png')[0]
      #   filename = obj_tag + ".png"
      #   filename2 = filename.replace('.png', '.posterior.png')
      #   # out_folder_mask = "forwarded/" + self.model_name + "/restricted/" + tag + "/" + timestep + "/"
      #   model_name = self.config.string("model", "test-challenge/bike-trial")
      #   out_folder_mask = "/home/luiten/vision/PReMVOS/post_proposal_expansion/" + model_name + "/" + timestep + "/"
      # else:
      #   out_folder_mask = "forwarded/" + self.model_name + "/mask/step_" + str(step_num) + "/" + tag + "/" + obj_tag + "/"
      #
      # tf.gfile.MakeDirs(out_folder_mask)
      # out_filename_mask = out_folder_mask + filename
      # Image.fromarray(predicted_segmentation.astype("uint8") * 255).save(out_filename_mask)
      # Image.fromarray((posteriors * 255).astype("uint8")).save(out_folder_mask + filename2)
      # # np.save(out_folder_mask + filename2,posteriors)
      #
      # do_viz = False
      # if do_viz:
      #   out_folder_viz = "forwarded/" + self.model_name + "/viz/step_" + str(
      #     step_num) + "/" + tag + "/" + obj_tag + "/"
      #   tf.gfile.MakeDirs(out_folder_viz)
      #   out_filename_viz = out_folder_viz + filename
      #   print(out_filename_viz, measures_string_to_print(compute_measures_average(step_measures, for_final_result=False)),
      #         file=log.v5)
      #   # darken a bit so that we can better see the mask
      #   img *= 0.8
      #   # apply mask
      #   img[predicted_segmentation == 1, 0] = 1.0
      #   im = Image.fromarray((img * 255).astype("uint8"))
      #   im.save(out_filename_viz)


    # if self.print_per_object_stats:
    #   print("results of", self.val_data.get_video_tag() + ":" + str(self.val_data.get_object_idx_in_video()),
    #         "after step", "{:05d}".format(step_num),
    #         measures_string_to_print(compute_measures_average(measures, for_final_result=True)), file=log.v4)
