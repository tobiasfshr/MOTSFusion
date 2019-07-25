import numpy as np
import os
import matplotlib.pyplot as plt
from utils.reconstruction_utils import transform, get_depth_covariance, get_center_point, shift_transform, \
    inv_shift_transform, flowwarp_box, get_point_box_distance, compute_final_bbox, get_bbox_points, compute_bboxes, compute_final_bbox_flow
from external.BB2SegNet.segment import segment
from visualization.visualize import show_mask, show_detection
from utils.segmentation_utils import masktoBbox


def compute_cov_norm(covariance):
    if covariance is not None:
        return np.linalg.norm(covariance.diagonal(), 2)
    else:
        return np.inf


def get_trust_regions(config, tracks, track_id, start_step=None, stop_step=None):
    track_covariances = tracks.get_track_attribute(track_id, 'transform_covariances')
    residual_thresh = config.float('residual_thresh')

    if start_step is None:
        start_step = 0

    if stop_step is None:
        stop_step = tracks.timesteps-1

    trust_regions = []
    transform_per_region = []
    step = start_step

    while step < stop_step:
        if compute_cov_norm(track_covariances[step]) < residual_thresh and step < stop_step-1 and tracks.get_attribute(step - 1, track_id, 'global_positions') is not None:
            trust_regions.append([step])
            center_point = get_center_point(config, tracks.get_attribute(step-1, track_id, 'global_points'), tracks.get_detection(step - 1, track_id)['class'])
            transform_per_region.append([shift_transform(center_point, tracks.get_attribute(step, track_id, 'dynamic_transforms'))])
            center_point = transform(tracks.get_attribute(step - 1, track_id, 'global_positions'),
                                     tracks.get_attribute(step, track_id, 'dynamic_transforms'))
            #min_r = compute_cov_norm(track_covariances[step])

            count = 1
            r = compute_cov_norm(track_covariances[step + count])
            while r < residual_thresh:
                trust_regions[-1].append(step + count)
                transform_per_region[-1].append(shift_transform(center_point, tracks.get_attribute(step + count, track_id, 'dynamic_transforms')))
                center_point = transform(center_point, tracks.get_attribute(step, track_id, 'dynamic_transforms'))
                #if r < min_r:
                    #transform_per_region[-1] = relative_poses[step + count][track_id]
                    #min_r = r
                count += 1
                if step + count < len(track_covariances):
                    r = compute_cov_norm(track_covariances[step + count])
                else:
                    break

            step += count
        else:
            step += 1

    return [region for region in trust_regions if len(region) > 1], [transform for transform in transform_per_region if len(transform) > 1]


def get_params(config, tracks, point_img_list, reference_id, step, steps_to_extrapolate, forward=True):
    if forward:
        trust_regions, transform_per_region = get_trust_regions(config, tracks, reference_id, stop_step=step)
    else:
        trust_regions, transform_per_region = get_trust_regions(config, tracks, reference_id, start_step=step)
    # if forward:
    #     print('trust_regions_ref', trust_regions)
    #     for transforms in transform_per_region:
    #         print('transform_per_region', transforms)

    if len(trust_regions):
        if forward:
            idx = -1
        else:
            idx = 0

        avg_transform = np.median(transform_per_region[idx], axis=0)

        if forward:
            steps_to_interpolate = step - trust_regions[idx][idx]
        else:
            steps_to_interpolate = trust_regions[idx][idx] - step

        start_point = tracks.get_attribute(trust_regions[idx][idx], reference_id, 'global_positions')
        for _ in range(steps_to_interpolate):
            shifted_avg_transform = inv_shift_transform(start_point, avg_transform)
            if forward:
                start_point = transform(start_point, shifted_avg_transform)
            else:
                start_point = transform(start_point, shifted_avg_transform, inverse=True)

        predict_points = [start_point]
        for _ in range(steps_to_extrapolate):
            shifted_avg_transform = inv_shift_transform(predict_points[-1], avg_transform)
            if forward:
                predict_points.append(transform(predict_points[-1], shifted_avg_transform))
            else:
                predict_points.append(transform(predict_points[-1], shifted_avg_transform, inverse=True))

    else:
        if forward:
            predict_points = [tracks.get_attribute(step, reference_id, 'global_positions'), tracks.get_attribute(step - 1, reference_id, 'global_positions')]
        else:
            predict_points = [tracks.get_attribute(step, reference_id, 'global_positions'), tracks.get_attribute(step + 1, reference_id, 'global_positions')]

        if predict_points[0] is not None and predict_points[1] is not None:
            if forward:
                predict_points.reverse()
                avg_transform = tracks.get_attribute(step, reference_id, 'dynamic_transforms')
            else:
                avg_transform = tracks.get_attribute(step + 1, reference_id, 'dynamic_transforms')

            if avg_transform is None:
                avg_transform = np.asarray([0, 0, 0])
        else:
            predict_points = [predict_points[0]]
            avg_transform = np.asarray([0, 0, 0])

    return predict_points, avg_transform, trust_regions


def track_consistency(config, tracks, reference_track, track_proposals, start_step, current_step, point_img_list, img_list, pose_list, calibration_params):
    steps_to_extrapolate = current_step - start_step
    if config.bool('debug'):
        print('.......................................................................')
        print('reference_track', reference_track)
        print('start_step', start_step)
        print('current_step',current_step)

    predict_points_ref, avg_transform_ref, trust_regions_ref = get_params(config, tracks, point_img_list, reference_track, start_step, steps_to_extrapolate)

    #print('predict_points_ref', predict_points_ref)


    result_proposal_id = -1
    result_proposal_trust_region_len = 0
    fill_bbox_params = None
    thresh = config.float('merge_treshold')

    for proposal in track_proposals:
        #print('proposal', proposal)
        predict_points_proposal, avg_transform_proposal, trust_regions_proposal = get_params(config, tracks, point_img_list, proposal, current_step, steps_to_extrapolate, forward=False)

        #print('predict_points_proposal', predict_points_proposal)
        consistency_score = 0

        num_timesteps = 0

        if len(predict_points_ref) > len(predict_points_proposal):
            predict_points_ref.reverse()

        if not reference_track == proposal:
            for point_ref, point_proposal in zip(predict_points_ref, predict_points_proposal):
                diff = np.asarray(point_ref) - np.asarray(point_proposal)
                diff = [diff[0], diff[2]]

                depth_covariance_ref = get_depth_covariance(point_ref, pose_list[start_step + num_timesteps], calibration_params)
                depth_covariance_proposal = get_depth_covariance(point_proposal, pose_list[current_step - num_timesteps], calibration_params)

                if config.bool('debug'):
                    print('depth_covariance_ref', depth_covariance_ref)
                    print("depth_covariance_proposal", depth_covariance_proposal)

                    print('avg_transform_ref', avg_transform_ref)
                    print('avg_transform_proposal', avg_transform_proposal)
                    avg_transform = ((avg_transform_ref + avg_transform_proposal) / 2)
                    print('avg_transform', avg_transform)

                diff[0] = diff[0] / np.log(depth_covariance_ref[0][0] + depth_covariance_proposal[0][0] + np.e) #/ np.log(np.abs(avg_transform[0]) + 2)
                diff[1] = diff[1] / np.log(depth_covariance_ref[2][2] + depth_covariance_proposal[2][2] + np.e) #/ np.log(np.abs(avg_transform[1]) + 2)
                #
                # print('diff[0]', diff[0])
                # print('diff[1]', diff[1])

                consistency_score += np.linalg.norm(diff, 2)
                num_timesteps += 1

            consistency_score = consistency_score / num_timesteps
        else:
            consistency_score = 0

        if config.bool('debug'):
            print('consistency score for ' + str(reference_track) + ' and ' + str(proposal) + ': ' + str(consistency_score))
            print('..............')

        if consistency_score < thresh:
            result_proposal_id = proposal
            result_proposal_trust_region_len = len(trust_regions_proposal)
            warping_start = avg_transform_ref
            warping_current = avg_transform_proposal
            overlapping_points = zip(predict_points_ref, predict_points_proposal)
            thresh = consistency_score

    if len(trust_regions_ref) and result_proposal_trust_region_len and steps_to_extrapolate > 1:
        prop_points = tracks.get_attribute(current_step, result_proposal_id, 'global_points')
        ref_points = tracks.get_attribute(start_step, reference_track, 'global_points')
        if prop_points is not None and ref_points is not None:
            evaluated_class = tracks.get_detection(start_step, reference_track)['class']
            bbox_start = get_center_point(config, ref_points, evaluated_class)
            bbox_current = get_center_point(config, prop_points, evaluated_class)
            fill_bbox_params = [bbox_start, bbox_current, warping_start, warping_current, overlapping_points]

    return result_proposal_id, fill_bbox_params


def fill_merged_track(config, tracks, fill_bbox_params, start_step, current_step, reference_id, flow, point_img_list, pose_list, img_list, refinement_net, calibration_params):
    evaluated_class = tracks.get_detection(start_step, reference_id)['class']
    bbox_start = masktoBbox(tracks.get_mask(start_step, reference_id, decode=False))  # here we want modal bboxes not amodal, hence take mask
    bbox_current = masktoBbox(tracks.get_mask(current_step, reference_id, decode=False))

    num_steps = current_step - (start_step + 1)
    bbox_delta = (bbox_current - bbox_start) / num_steps

    bboxes_all = [(bbox_start, [0,0,0])]
    for timestep in range(num_steps):
        bboxes_all.append((bboxes_all[-1][0] + bbox_delta, flowwarp_box(bboxes_all[-1][0], flow[start_step + timestep])))

    bboxes_all.pop(0)

    bboxes_ref, bboxes_proposal, bbox_start2, bbox_current2 = compute_bboxes(fill_bbox_params, start_step, current_step)

    start_point_box_distance = get_point_box_distance(
        compute_final_bbox(bbox_start, get_bbox_points(config, bbox_start2), point_img_list[start_step],
                           pose_list[start_step], calibration_params),
        point_img_list[start_step], [bbox_start2[0], bbox_start2[2]])

    current_point_box_distance = get_point_box_distance(
        compute_final_bbox(bbox_current, get_bbox_points(config, bbox_current2), point_img_list[current_step],
                           pose_list[current_step], calibration_params),
        point_img_list[current_step], [bbox_current2[0], bbox_current2[2]])

    avg_point_box_distance = (start_point_box_distance + current_point_box_distance) / 2

    if config.bool('debug'):
        print('................................................................')
        print('reference_id', reference_id)
        print('start_step', start_step)
        print('avg_point_box_distance', avg_point_box_distance)
        print('bbox_delta', bbox_delta)

    for index, bboxs in enumerate(bboxes_all):
        bbox, bbox_flow = bboxs

        points = get_bbox_points(config, (np.asarray(bboxes_ref[index]) + np.asarray(bboxes_proposal[index])) / 2)

        point_img = point_img_list[start_step + 1 + index]

        [x1, y1, x2, y2] = compute_final_bbox_flow(bbox, bbox_flow, points, point_img, pose_list[start_step + index], calibration_params)

        avg_predicted_position = (np.asarray(bboxes_ref[index]) + np.asarray(bboxes_proposal[index])) / 2
        avg_predicted_position = np.asarray([avg_predicted_position[0], avg_predicted_position[2]])

        point_box_distance = get_point_box_distance([x1, y1, x2, y2], point_img, avg_predicted_position)

        if config.bool('debug'):
            print('point_box_distance', point_box_distance)

        if point_box_distance < np.minimum(4, 1.8 * avg_point_box_distance):
            detection = {}
            detection['bbox'] = [x1, y1, x2, y2]
            detection['score'] = 1.0
            detection['class'] = evaluated_class
            #show_detection(img_list[current_step], detection)
            mask = segment([detection], img_list[start_step + index + 1], refinement_net)[0]
            tracks.add_to_track(start_step + index + 1, reference_id, detection, mask)
            if config.bool('debug'):
                avg_box = (np.asarray(bboxes_ref[index]) + np.asarray(bboxes_proposal[index]))
                avg_box[3] = np.arctan((np.sin(bboxes_ref[index][3]) + np.sin(bboxes_proposal[index][3])) / (np.cos(bboxes_ref[index][3]) + np.cos(bboxes_proposal[index][3])))
                tracks.set_attribute(start_step + index + 1, reference_id, 'global_3D_bbox', avg_box)


def extrapolate_terminated_track(config, tracks, flow_list, pose_list, point_img_list, calibration_params, img_list, refinement_net, start_step, reference_id, inverse=False):
    evaluated_class = tracks.get_detection(start_step, reference_id)['class']
    bbox_t1 = masktoBbox(tracks.get_mask(start_step, reference_id, decode=False))  # here we want modal bboxes not amodal, hence take mask

    if inverse:
        if tracks.get_detection(start_step+1, reference_id) is None:
            return

        bbox_t0 = masktoBbox(tracks.get_mask(start_step+1, reference_id, decode=False))
        flow = flow_list[start_step+1]
        bbox_flow = flowwarp_box(bbox_t0, flow, inverse=True)
        bbox_delta = (bbox_t0 - bbox_t1)
    else:
        if tracks.get_detection(start_step-1, reference_id) is None:
            return

        bbox_t0 = masktoBbox(tracks.get_mask(start_step-1, reference_id, decode=False))
        flow = flow_list[start_step]
        bbox_flow = flowwarp_box(bbox_t0, flow)
        bbox_delta = (bbox_t1 - bbox_t0)

    bbox = bbox_t1

    bbox_ref = get_center_point(config, tracks.get_attribute(start_step, reference_id, 'global_points'), evaluated_class)

    if inverse:
        _, avg_transform_ref, trust_regions_ref = get_params(config, tracks, point_img_list, reference_id, start_step, 0, forward=False)
    else:
        _, avg_transform_ref, trust_regions_ref = get_params(config, tracks, point_img_list, reference_id, start_step, 0)

    avg_point_box_distance = get_point_box_distance(
                                compute_final_bbox_flow(bbox_t1, bbox_flow, get_bbox_points(config, bbox_ref, evaluated_class),
                                                        point_img_list[start_step], pose_list[start_step], calibration_params),
                                                            point_img_list[start_step], [bbox_ref[0], bbox_ref[2]])

    if config.bool('debug'):
        print('................................................................')
        print('reference_id', reference_id)
        print('start_step', start_step)
        print('avg_point_box_distance',avg_point_box_distance)
        print('bbox_delta', bbox_delta)
        print('bbox_start', bbox)

    count = 1
    in_frame = True
    point_box_distance = 0
    if inverse:
        reached_eos = start_step - count <= 0
    else:
        reached_eos = start_step + count >= len(point_img_list)

    while in_frame and point_box_distance < np.minimum(4, 1.8 * avg_point_box_distance) and not reached_eos:
        # update
        shifted_avg_transform = inv_shift_transform(bbox_ref[0:3], avg_transform_ref)
        if inverse:
            current_step = start_step - count
            flow = flow_list[current_step + 1]

            bbox_flow = flowwarp_box(bbox, flow, inverse=True)
            bbox = bbox - bbox_delta
            bbox_ref[0:3] = transform(bbox_ref[0:3], shifted_avg_transform, inverse=True)
            bbox_ref[3] = bbox_ref[3] - shifted_avg_transform[2]

        else:
            current_step = start_step + count
            flow = flow_list[current_step]

            bbox_flow = flowwarp_box(bbox, flow)
            bbox = bbox + bbox_delta
            bbox_ref[0:3] = transform(bbox_ref[0:3], shifted_avg_transform)
            bbox_ref[3] = bbox_ref[3] + shifted_avg_transform[2]

        point_img = point_img_list[current_step]

        # get bbox
        points = get_bbox_points(config, bbox_ref, evaluated_class)
        [x1, y1, x2, y2] = compute_final_bbox_flow(bbox, bbox_flow, points, point_img, pose_list[current_step], calibration_params)

        # check bbox
        predicted_position = np.asarray([bbox_ref[0], bbox_ref[2]])
        point_box_distance = get_point_box_distance([x1, y1, x2, y2], point_img, predicted_position)

        aspect_ratio = (x2 - x1) / (y2 - y1)

        in_frame = x1 < point_img.shape[1] and y1 < point_img.shape[0] and x2 > 0 and y2 > 0 and 0.5 < aspect_ratio < 2.5

        if config.bool('debug'):
            print('....................')
            print('current_step', current_step)
            print('bbox', [x1, y1, x2, y2])
            print('point_box_distance', point_box_distance)
            print('aspect_ratio', aspect_ratio)
            print('in_frame', in_frame and point_box_distance < np.minimum(4, 1.8 * avg_point_box_distance))

        if in_frame and point_box_distance < np.minimum(3, 1.8 * avg_point_box_distance):
            detection = {}
            detection['bbox'] = [x1, y1, x2, y2]
            detection['score'] = 1.0
            detection['class'] = evaluated_class
            #show_detection(img_list[current_step], detection)
            mask = segment([detection], img_list[current_step], refinement_net)[0]

            tracks.add_to_track(current_step, reference_id, detection, mask)

            if config.bool('debug'):
                tracks.set_attribute(current_step, reference_id, 'global_3D_bbox', bbox_ref)

            count += 1
            if not inverse:
                in_frame = x1 > 20 and y1 > 10 and x2 < point_img.shape[1] - 20 and y2 < point_img.shape[0] - 10

            if inverse:
                reached_eos = start_step - count <= 0
            else:
                reached_eos = start_step + count >= len(point_img_list)


def merge_tracks(config, tracks, point_img_list, img_list, pose_list, flow, calibration_params, refinement_net):
    merged_ids = {}

    for step in range(tracks.timesteps-1):

        for ref_id in tracks.get_active_tracks(step):

            if not tracks.is_active(step + 1, ref_id) and tracks.get_attribute(step, ref_id, 'global_positions') is not None:  #if is active and is not an already filled track (=no additional information to mask and detection)
                timesteps = range(step + 1, np.minimum(step + config.int('merging_timesteps'), tracks.timesteps - 1))
                for t in timesteps:
                    proposals = []
                    for candidate_id in tracks.get_active_tracks(t):
                        if not tracks.is_active(t - 1, candidate_id):
                            proposals.append(candidate_id)

                    if len(proposals):
                        merging_candidate, fill_bbox_params = track_consistency(config, tracks, ref_id, proposals, step,
                                                                                t, point_img_list, img_list, pose_list,
                                                                                calibration_params)
                    else:
                        merging_candidate = -1
                        fill_bbox_params = None

                    if merging_candidate != -1:

                        # plausibility check
                        plausible = True
                        for track in tracks.track_ids:
                            if track[merging_candidate] and track[ref_id]:
                                plausible = False

                        if plausible:
                            if merging_candidate < ref_id:
                                temp = merging_candidate
                                merging_candidate = ref_id
                                ref_id = temp

                            tracks.merge_tracks(ref_id, merging_candidate)

                            if config.bool('debug'):
                                print('merged following tracks:')
                                print(ref_id)
                                print(merging_candidate)
                                print('At following timesteps:')
                                print(step)
                                print(t)
                                print('')

                            if not ref_id in merged_ids.keys():
                                merged_ids[ref_id] = [merging_candidate]
                            else:
                                merged_ids[ref_id].append(merging_candidate)

                            if config.str('mode') == 'MOTS' and t - step > 1 and fill_bbox_params is not None:
                                fill_merged_track(config, tracks, fill_bbox_params, step, t, ref_id, flow, point_img_list,
                                                  pose_list, img_list, refinement_net, calibration_params)
                            break

    if config.bool('debug'):
        print("merged ids", merged_ids)
    return tracks


def extrapolate_final_tracks(config, tracks, flow_list, pose_list, point_img_list, calibration_params, img_list, refinement_net):
    if config.str('mode') == 'MOTS':
        for ref_id in range(tracks.get_num_ids()):
            track = tracks.get_track(ref_id)
            if any(track):
                locations = [loc for loc, val in enumerate(track) if val is True]
                end_step = max(locations)
                start_step = min(locations)

                if 0 < end_step < tracks.timesteps-1:
                        extrapolate_terminated_track(config, tracks, flow_list, pose_list, point_img_list, calibration_params, img_list, refinement_net, end_step, ref_id)

                if 0 < start_step < tracks.timesteps-1:
                        extrapolate_terminated_track(config, tracks, flow_list, pose_list, point_img_list, calibration_params, img_list, refinement_net, start_step, ref_id, inverse=True)

    return tracks
