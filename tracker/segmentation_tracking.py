from utils.segmentation_utils import warp_box, warp_flow
import numpy as np
from scipy.special import expit as sigmoid
from collections import namedtuple
import munkres
from scipy.spatial.distance import cdist
from utils.geometry_utils import bbox_iou
from tracker.tracked_sequence import TrackedSequence
import matplotlib.pyplot as plt
import pycocotools.mask as cocomask

munkres_obj = munkres.Munkres()

TrackElement = namedtuple("TrackElement", ["track_id", "step"])


def create_tracklets(config, detections_raw, masks_raw, flow):
    # perform tracking per class and in the end combine the results
    classes = config.int_list('classes_to_track')
    tracker_options_class = {"tracker": config.str("tracker")}

    if flow is None:
        assert False, "optical flow did not load correctly"

    while len(flow) > len(detections_raw):
        detections_raw.append([])

    while len(flow) > len(masks_raw):
        masks_raw.append([])

    tracks = TrackedSequence(len(flow))

    for class_ in classes:
        if class_ == 1:
            tracker_options_class["detection_confidence_threshold"] = config.float(
                "detection_confidence_threshold_car")
            tracker_options_class["association_threshold"] = config.float("association_threshold_car")
        elif class_ == 2:
            tracker_options_class["detection_confidence_threshold"] = config.float(
                "detection_confidence_threshold_pedestrian")
            tracker_options_class["association_threshold"] = config.float("association_threshold_pedestrian")
        else:
            assert False, "unknown class"

        tracks = tracker_per_class(tracks, tracker_options_class, detections_raw, masks_raw, class_, optical_flow=flow)

    return tracks


def tracker_per_class(tracks, tracker_options, detections_raw, masks_raw, class_to_track, optical_flow=None):
    active_tracks = []

    for t, (detections_raw_t, masks_t, flow_tm1_t) in enumerate(zip(detections_raw, masks_raw, optical_flow)):

        # if flow_tm1_t is not None:
        #     print(t)
        #     show_flow(flow_tm1_t)
        detections_t = []
        for detection, mask in zip(detections_raw_t, masks_t):
            if detection['class'] != class_to_track:
                continue
            if mask is None:
                continue
            if cocomask.area(mask) <= 10:
                continue
            if detection['score'] >= tracker_options["detection_confidence_threshold"]:
                detections_t.append((detection, mask))

        if len(active_tracks) == 0:
            for det in detections_t:
                active_tracks.append(TrackElement(track_id=tracks.start_new_track(t, det[0], det[1]), step=t))
        elif len(detections_t) != 0:

            association_similarities = np.zeros((len(detections_t), len(active_tracks)))
            masks_t = [v[1] for v in detections_t]
            masks_tm1 = [tracks.get_mask(t-1, v.track_id, decode=False) for v in active_tracks]
            masks_tm1_warped = [warp_flow(mask, flow_tm1_t) for mask in masks_tm1]
            mask_ious = cocomask.iou(masks_t, masks_tm1_warped, [False] * len(masks_tm1_warped))
            association_similarities += mask_ious

            # for mask_new, mask_warped in zip(masks_t, masks_tm1_warped):
            #     print('warped mask')
            #     plt.imshow(cocomask.decode(mask_warped))
            #     plt.show()
            #     print('new mask')
            #     plt.imshow(cocomask.decode(mask_new))
            #     plt.show()
            #
            # print('current time', t)
            # print('len detections_t', len(detections_t))
            # print('active_tracks', active_tracks)
            # print('overall tracks', tracks.get_num_ids())
            # print('association_similarities', association_similarities)
            # print('')

            detections_assigned = [False for _ in detections_t]
            if tracker_options["tracker"] == "greedy":
                while True:
                    idx = association_similarities.argmax()
                    idx = np.unravel_index(idx, association_similarities.shape)
                    val = association_similarities[idx]
                    if val < tracker_options["association_threshold"]:
                        break
                    det = detections_t[idx[0]]
                    tracks.add_to_track(t, active_tracks[idx[1]].track_id, det[0], det[1])
                    active_tracks[idx[1]] = active_tracks[idx[1]]._replace(step=t)

                    detections_assigned[idx[0]] = True
                    association_similarities[idx[0], :] = -1e10
                    association_similarities[:, idx[1]] = -1e10

            elif tracker_options["tracker"] == "hungarian":
                cost_matrix = munkres.make_cost_matrix(association_similarities)
                disallow_indices = np.argwhere(association_similarities <= tracker_options["association_threshold"])
                for ind in disallow_indices:
                    cost_matrix[ind[0]][ind[1]] = 1e9
                indexes = munkres_obj.compute(cost_matrix)
                for row, column in indexes:
                    value = cost_matrix[row][column]
                    if value == 1e9:
                        continue

                    det = detections_t[row]
                    tracks.add_to_track(t, active_tracks[column].track_id, det[0], det[1])
                    active_tracks[column] = active_tracks[column]._replace(step=t)
                    detections_assigned[row] = True
                    # print('detection ' + str(row) + ' assigned to track', active_tracks[column].track_id)
            else:
                assert False, "no appropriate tracking algorithm selected"

            for det, assigned in zip(detections_t, detections_assigned):
                if not assigned:
                    active_tracks.append(TrackElement(track_id=tracks.start_new_track(t, det[0], det[1]), step=t))
        else:
            active_tracks = []

        # print('written to memory', tracks.get_active_tracks(t))
        # print('--------')

        active_tracks = [track for track in active_tracks if track.step >= t]

    return tracks

