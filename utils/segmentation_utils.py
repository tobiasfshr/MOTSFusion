import pycocotools.mask as cocomask
import numpy as np
from scipy.spatial.distance import cdist
from utils.geometry_utils import bbox_iou
import cv2


def calculate_association_similarities(detections_t, last_tracks, flow_tm1_t, tracker_options):
    association_similarities = np.zeros((len(detections_t), len(last_tracks)))
    if tracker_options["reid_weight"] != 0:
        curr_reids = np.array([x[1] for x in detections_t], dtype="float64")
        last_reids = np.array([x.reid for x in last_tracks], dtype="float64")
        reid_dists = cdist(curr_reids, last_reids, "euclidean")
        reid_similarities = tracker_options["reid_euclidean_scale"] * \
                            (tracker_options["reid_euclidean_offset"] - reid_dists)
        association_similarities += tracker_options["reid_weight"] * reid_similarities
    if tracker_options["mask_iou_weight"] != 0:
        masks_t = [v[2] for v in detections_t]
        masks_tm1 = [v.mask for v in last_tracks]
        masks_tm1_warped = [warp_flow(mask, flow_tm1_t) for mask in masks_tm1]
        mask_ious = cocomask.iou(masks_t, masks_tm1_warped, [False] * len(masks_tm1_warped))
        association_similarities += tracker_options["mask_iou_weight"] * mask_ious
    if tracker_options["bbox_center_weight"] != 0:
        centers_t = [v[0][0:2] + (v[0][2:4] - v[0][0:2]) / 2 for v in detections_t]
        centers_tm1 = [v.box[0:2] + (v.box[2:4] - v.box[0:2]) / 2 for v in last_tracks]
        box_dists = cdist(np.array(centers_t), np.array(centers_tm1), "euclidean")
        box_similarities = tracker_options["box_scale"] * \
                           (tracker_options["box_offset"] - box_dists)
        association_similarities += tracker_options["bbox_center_weight"] * box_similarities
    if tracker_options["bbox_iou_weight"] != 0:
        bboxes_t = [v[0] for v in detections_t]
        bboxes_tm1 = [v.box for v in last_tracks]
        bboxes_tm1_warped = [warp_box(box, flow_tm1_t) for box in bboxes_tm1]
        bbox_ious = np.array([[bbox_iou(box1, box2) for box1 in bboxes_tm1_warped] for box2 in bboxes_t])
        assert (0 <= bbox_ious).all() and (bbox_ious <= 1).all()
        association_similarities += tracker_options["bbox_iou_weight"] * bbox_ious
    return association_similarities


def warp_flow(mask_as_rle, flow):
    # unpack
    mask = cocomask.decode([mask_as_rle])
    # warp
    warped = _warp(mask, flow)
    # pack
    packed = cocomask.encode(np.asfortranarray(warped))
    return packed


def _warp(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    # for some reason the result is all zeros with INTER_LINEAR...
    # res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    res = cv2.remap(img, flow, None, cv2.INTER_NEAREST)
    res = np.equal(res, 1).astype(np.uint8)
    return res


def warp_box(box, flow):
    box_rounded = np.maximum(box.round().astype("int32"), 0)
    x0, y0, x1, y1 = box_rounded
    flows = flow[y0:y1, x0:x1]
    flows_x = flows[:, :, 0]
    flows_y = flows[:, :, 1]
    flow_x = np.median(flows_x)
    flow_y = np.median(flows_y)
    box_warped = box + [flow_x, flow_y, flow_x, flow_y]
    return box_warped


def masktoBbox(mask):
    [x,y,w,h] = cocomask.toBbox(mask)
    return np.asarray([x,y, x+w, y+h])
