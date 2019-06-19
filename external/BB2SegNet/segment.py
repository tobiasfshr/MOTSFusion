from scipy.misc import imread
from pycocotools.mask import encode
import numpy as np
import json
import os

from external.BB2SegNet.refinement_net.core.Engine import Engine
from external.BB2SegNet.refinement_net.core.Config import Config
from external.BB2SegNet.refinement_net.core import Extractions
import external.BB2SegNet.refinement_net.datasets.DataKeys as DataKeys


def refinement_net_init(config_path):
    config = Config(config_path)
    engine = Engine(config)
    return engine


def extract(key, extractions):
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


def do_refinement(detections, image, refinement_net):
    data = refinement_net.valid_data
    boxes = [[det['bbox'][0], det['bbox'][1], det['bbox'][2] - det['bbox'][0], det['bbox'][3] - det['bbox'][1]] for det in detections]
    image_data = data.set_up_data_for_image(image, boxes)

    segmentations = []
    for idx in range(len(boxes)):
        segmentation = {}
        feed_dict = data.get_feed_dict_for_next_step(image_data, idx)
        step_res = refinement_net.trainer.validation_step(feed_dict=feed_dict, extraction_keys=[
            Extractions.SEGMENTATION_POSTERIORS_ORIGINAL_SIZE,
            Extractions.SEGMENTATION_MASK_ORIGINAL_SIZE, DataKeys.OBJ_TAGS])
        extractions = step_res[Extractions.EXTRACTIONS]
        predicted_segmentation = extract(Extractions.SEGMENTATION_MASK_ORIGINAL_SIZE, extractions)
        mask = predicted_segmentation.astype("uint8") * 255
        encoded_mask = encode(np.asfortranarray(mask))
        segmentation['counts'] = encoded_mask['counts'].decode("UTF-8")

        posteriors = extract(Extractions.SEGMENTATION_POSTERIORS_ORIGINAL_SIZE, extractions)
        conf_scores = posteriors.copy()
        conf_scores[predicted_segmentation == 0] = 1 - posteriors[predicted_segmentation == 0]
        conf_scores = 2 * conf_scores - 1
        conf_score = conf_scores[:].mean()
        segmentation['score'] = str(conf_score)
        segmentation['size'] = [image.shape[0], image.shape[1]]

        segmentations.append(segmentation)

    return segmentations


def compute_segmentations(refinement_net, sequence_dir, save_path, detections):
    sequence_dir = sequence_dir + 'image_2/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_list = sorted(os.listdir(sequence_dir))
    segmentations = []
    for i in range(len(file_list)):
        curr_segmentations = do_refinement(detections[i], imread(sequence_dir + file_list[i]), refinement_net)
        segmentations.append(curr_segmentations)

    json.dump(segmentations, open(save_path + 'segmentations.json', 'w'))
    del refinement_net


def segment(detections, image, refinement_net, save_path=None):
    segmentations = do_refinement(detections, image, refinement_net)
    if save_path is not None:
        json.dump(segmentations, open(save_path + 'segmentations.json', 'w'))
    return segmentations
