import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from visualization.vtkVisualization import *
from utils.geometry_utils import flow_warp_idx
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(suppress=True)
import pickle
from matplotlib import cm
from utils.geometry_utils import bbox_iou, pose_vec2mat
import cv2


def get_points_from_masks(mask_t0, mask_t1, point_img_t0, point_img_t1, flow_t1_t0, img_t0, img_t1, calibration_params):
    #point_img_t0[np.logical_not(mask_t0)] = [0, 0, 0]

    h, w = flow_t1_t0.shape[:2]
    flow = -flow_t1_t0
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]

    point_img_t0 = cv2.remap(point_img_t0, flow, None, cv2.INTER_NEAREST)

    mask_t0_warped = cv2.remap(mask_t0, flow, None, cv2.INTER_NEAREST)
    mask_t0_warped = np.equal(mask_t0_warped, 1).astype(np.uint8)

    mask_overlap = np.logical_and(mask_t0_warped.astype(np.bool), mask_t1.astype(np.bool))

    object_points = np.concatenate((np.expand_dims(point_img_t0[mask_overlap], axis=1), np.expand_dims(point_img_t1[mask_overlap], axis=1)), axis=1)
    colors = np.concatenate((np.expand_dims(img_t0[mask_overlap], axis=1), np.expand_dims(img_t1[mask_overlap], axis=1)), axis=1)

    return object_points, colors


def get_overlapping_box(boxes, ref_box, iou_thresh=0.5):
    ious = []
    for box in boxes:
        ious.append(bbox_iou(box, ref_box))

    for idx, iou in enumerate(ious):
        if iou > iou_thresh:
            return boxes[idx]

    return None


def compute_jacobian(proj_mat, point):
    w = proj_mat[2, 0] * point[0] + proj_mat[2, 1] * point[1] + proj_mat[2, 2] * point[2] + proj_mat[2, 3]
    w_2 = np.square(w)

    F = np.zeros((2, 3))
    for i in range(F.shape[0]):
        sum_row_by_point = proj_mat[i, 0] * point[0] + proj_mat[i, 1] * point[1] + proj_mat[i, 2] * point[2] + proj_mat[
            i, 3]
        for j in range(F.shape[1]):
            F[i, j] = proj_mat[i, j] / w - sum_row_by_point * proj_mat[2, j] / w_2

    return F


def compute_uncertainty(point, calibration_params, output_covar=False):
    FL = compute_jacobian(calibration_params.cam_projection_matrix_l, point)
    FR = compute_jacobian(calibration_params.cam_projection_matrix_r, point)

    c2d_mat = np.linalg.inv(np.identity(2) * 0.5)
    FLTFL = np.matmul(np.matmul(FL.transpose(), c2d_mat), FL)
    FRTFR = np.matmul(np.matmul(FR.transpose(), c2d_mat), FR)

    cov_uncertainty = np.linalg.inv(np.add(FLTFL, FRTFR) +0.00001*np.random.rand(3, 3))  # add a bit noise to avoid singular matrix error
    uncertainty = 1 / np.linalg.norm(cov_uncertainty, 2)
    if not output_covar:
        return uncertainty
    else:
        return cov_uncertainty


def get_dynamic_transform(points):
    filler = np.ones(len(points))
    A = np.asarray(np.column_stack((points[:, 1, 0], points[:, 1, 2])))
    B = np.asarray(np.column_stack((points[:, 0, 0], points[:, 0, 2], filler)))
    # from A and B get delta x,z,theta (relative objection motion in birds eye view)
    init = [0, 0, 0]

    def func(pose):
        trans_mat = pose_vec2mat(pose)
        r = np.asarray(B @ trans_mat.T)[:, 0:2] - A
        return np.linalg.norm(r, 2)

    result = scipy.optimize.minimize(func, np.asarray(init))
    trans_mat = pose_vec2mat(result.x)
    residual = np.asarray(B @ trans_mat.T)[:, 0:2] - A
    init = result.x
    covariance = np.diag(np.mean(residual, axis=0))
    return init, covariance


def get_position_covariance(points, calibration_params):
    position_covariance = np.var(points)
    depth_covariance = compute_uncertainty(np.mean(points, axis=0), calibration_params, output_covar=True)
    return position_covariance + depth_covariance


def get_depth_covariance(point, transformation_matrix, calibration_params):
    point = np.column_stack(([point], [[1]]))
    point = point @ np.linalg.inv(transformation_matrix.T)
    point = point[:, 0:3]
    covariance = compute_uncertainty(point[0], calibration_params, output_covar=True)
    return covariance


def warp_points(relative_poses, id, points):
    global_pose = pose_vec2mat([0, 0, 0])
    for pose in relative_poses:
        if pose is not None:
            global_pose = np.matmul(global_pose, np.linalg.inv(pose_vec2mat(pose)))

    filler = np.ones(len(points))
    points_2d = np.column_stack((points[:, 0], points[:, 2], filler))
    warped_points = points_2d @ global_pose.T
    warped_points = np.column_stack((warped_points[:, 0], points[:, 1], warped_points[:, 1]))
    return warped_points


def transform(pos, trans_vec, inverse=False):
    if len(pos) == 3:
        pos_y = pos[1]
        pos = np.asarray([pos[0], pos[2], 1.])
    else:
        pos_y = None
        pos = np.asarray([pos[0], pos[1], 1.])

    if inverse:
        new_position = (pos @ np.linalg.inv(pose_vec2mat(trans_vec)).T)[0:2]
    else:
        new_position = (pos @ pose_vec2mat(trans_vec).T)[0:2]

    if pos_y is None:
        return np.asarray([new_position[0], new_position[1]])
    else:
        return np.asarray([new_position[0], pos_y, new_position[1]])


def inv_shift_transform(new_center_point, relative_pose):
    old_x = relative_pose[0]
    old_z = relative_pose[1]
    theta = relative_pose[2]

    new_x = -np.cos(theta) * new_center_point[0] + np.sin(theta) * new_center_point[2] + new_center_point[0] + old_x
    new_z = -np.sin(theta) * new_center_point[0] - np.cos(theta) * new_center_point[2] + new_center_point[2] + old_z

    return np.asarray([new_x, new_z, theta])


def shift_transform(center_point, relative_pose):
    old_x = relative_pose[0]
    old_z = relative_pose[1]
    theta = relative_pose[2]

    new_x = np.cos(theta) * center_point[0] - np.sin(theta) * center_point[2] - center_point[0] + old_x
    new_z = np.sin(theta) * center_point[0] + np.cos(theta) * center_point[2] - center_point[2] + old_z

    return np.asarray([new_x, new_z, theta])


def get_bbox_points(config, bbox_params, object_class=1):
    if object_class == 1:
        half_w = config.float('car_avg_w') / 2
        half_h = config.float('car_avg_h') / 2
        half_l = config.float('car_avg_l') / 2
    else:
        half_w = config.float('pedestrian_avg_w') / 2
        half_h = config.float('pedestrian_avg_h') / 2
        half_l = config.float('pedestrian_avg_l') / 2

    bbox_points = np.asarray(
        [[half_w, half_h, half_l], [-half_w, -half_h, half_l], [-half_w, half_h, half_l], [half_w, -half_h, half_l],
         [half_w, half_h, -half_l], [-half_w, -half_h, -half_l], [-half_w, half_h, -half_l],
         [half_w, -half_h, -half_l]])

    bbox_points2 = np.column_stack((bbox_points[:, 0], bbox_points[:, 2]))
    bbox_points2 = bbox_points2 @ np.asarray(
        [[np.cos(bbox_params[3]), -np.sin(bbox_params[3])], [np.sin(bbox_params[3]), np.cos(bbox_params[3])]]).T
    bbox_points = np.column_stack((bbox_points2[:, 0], bbox_points[:, 1], bbox_points2[:, 1]))
    bbox_points = bbox_points + bbox_params[0:3]
    return bbox_points


def get_center_point(config, points, object_class, past_points=None):
    # if object_class == 1:
    #     half_w = config.float('car_avg_w') / 2
    #     half_h = config.float('car_avg_h') / 2
    #     half_l = config.float('car_avg_l') / 2
    # else:
    #     half_w = config.float('pedestrian_avg_w') / 2
    #     half_h = config.float('pedestrian_avg_h') / 2
    #     half_l = config.float('pedestrian_avg_l') / 2

    points = np.asarray(points)
    x_mean, y_mean, z_mean = np.median(points, axis=0)
    theta_best = 0

    if past_points is not None:
        current = np.median(points, axis=0)
        past = np.median(past_points, axis=0)
        if np.linalg.norm(current - past, 2) > 0.7:
            dy = current[2] - past[2]
            dx = current[0] - past[0]
            theta_best = np.arctan(dy / dx)

    if theta_best == 0:
        ps = points
        ps = np.column_stack((ps[:, 0], ps[:, 2]))

        from sklearn.decomposition.pca import PCA

        pca = PCA(n_components=2)
        pca.fit(ps)

        theta_best = np.arctan(pca.components_[0][1] / pca.components_[0][0])

    theta_best = theta_best + np.pi / 2

    return np.array((x_mean,y_mean,z_mean,theta_best))


def compute_bboxes(fill_bbox_params, start_step, current_step):
    [bbox_start, bbox_current, warping_start, warping_current, overlapping_points] = fill_bbox_params
    points_ref, points_proposal = zip(*overlapping_points)

    bboxes_ref = [bbox_start.tolist()]
    for timestep in range(start_step + 1, current_step):
        bboxes_ref.append(np.append(points_ref[timestep - start_step], bboxes_ref[-1][3] + warping_start[2]).tolist())

    bboxes_proposal = [bbox_current.tolist()]
    for timestep in reversed(range(start_step + 1, current_step)):
        bboxes_proposal.append(
            np.append(points_proposal[timestep - start_step], bboxes_proposal[-1][3] - warping_current[2]).tolist())

    bboxes_ref.remove(bbox_start.tolist())
    bboxes_proposal.remove(bbox_current.tolist())

    return bboxes_ref, bboxes_proposal, bbox_start, bbox_current


def get_point_box_distance(bbox, point_img, avg_predicted_position):
    [x1, y1, x2, y2] = bbox
    points_in_bbox = point_img[int(y1):int(y2), int(x1):int(x2)]
    points_in_bbox = points_in_bbox.reshape((points_in_bbox.shape[0] * points_in_bbox.shape[1], points_in_bbox.shape[2]))
    points_in_bbox = np.column_stack((points_in_bbox[:, 0], points_in_bbox[:, 2]))
    return np.linalg.norm(np.median(points_in_bbox, axis=0) - avg_predicted_position, 2)


def compute_final_bbox(bbox, points, point_img, transformation_matrix, calibration_params, w_delta=0.9, w_3D=0.1):
    # transform points according to estimated camera motion (pose)
    points = np.column_stack((points, np.ones(len(points))))
    points = points @ np.linalg.inv(transformation_matrix.T)
    points = points[:, 0:3]

    # transform points into image pixels (camera parameters)
    points = points @ calibration_params.camera_matrix.T

    points = points[:, 0:2] / np.expand_dims(points[:, 2], axis=-1)

    x1 = np.maximum(np.min(points[:, 0]), 0)
    y1 = np.maximum(np.min(points[:, 1]), 0)
    x2 = np.minimum(np.max(points[:, 0]), point_img.shape[1])
    y2 = np.minimum(np.max(points[:, 1]), point_img.shape[0])

    x1 = np.maximum(w_delta * bbox[0] + w_3D * x1, 0)
    y1 = np.maximum(w_delta * bbox[1] + w_3D * y1, 0)
    x2 = np.minimum(w_delta * (bbox[2]) + w_3D * x2, point_img.shape[1])
    y2 = np.minimum(w_delta * (bbox[3]) + w_3D * y2, point_img.shape[0])

    return [x1, y1, x2, y2]


def compute_final_bbox_flow(bbox, bbox_flow, points, point_img, transformation_matrix, calibration_params, w_delta=0.5, w_flow=0.4, w_3D=0.1):
    # transform points according to estimated camera motion (pose)
    points = np.column_stack((points, np.ones(len(points))))
    points = points @ np.linalg.inv(transformation_matrix.T)
    points = points[:, 0:3]

    # transform points into image pixels (camera parameters)
    points = points @ calibration_params.camera_matrix.T

    points = points[:, 0:2] / np.expand_dims(points[:, 2], axis=-1)

    x1 = np.maximum(np.min(points[:, 0]), 0)
    y1 = np.maximum(np.min(points[:, 1]), 0)
    x2 = np.minimum(np.max(points[:, 0]), point_img.shape[1])
    y2 = np.minimum(np.max(points[:, 1]), point_img.shape[0])

    x1 = np.maximum(w_delta * bbox[0] + w_flow * bbox_flow[0] + w_3D * x1, 0)
    y1 = np.maximum(w_delta * bbox[1] + w_flow * bbox_flow[1] + w_3D * y1, 0)
    x2 = np.minimum(w_delta * (bbox[2]) + w_flow * bbox_flow[2] + w_3D * x2, point_img.shape[1])
    y2 = np.minimum(w_delta * (bbox[3]) + w_flow * bbox_flow[3] + w_3D * y2, point_img.shape[0])

    return [x1, y1, x2, y2]


def flowwarp_box(bbox, flow, inverse=False):
    start_x = int(np.clip(int(bbox[0]), 0, flow.shape[1]))
    end_x = int(np.clip(int(bbox[2] + bbox[0]), 0, flow.shape[1]))
    start_y = int(np.clip(bbox[1], 0, flow.shape[1]))
    end_y = int(np.clip(int(bbox[3] + bbox[1]), 0, flow.shape[0]))
    avg_flow = np.median(flow[start_y:end_y, start_x:end_x], axis=(0, 1))
    if any(np.isnan(avg_flow)):
        avg_flow = [0, 0]

    if inverse:
        x1 = bbox[0] - avg_flow[0]
        y1 = bbox[1] - avg_flow[1]
        x2 = (bbox[2]) - avg_flow[0]
        y2 = (bbox[3]) - avg_flow[1]
    else:
        x1 = bbox[0] + avg_flow[0]
        y1 = bbox[1] + avg_flow[1]
        x2 = (bbox[2]) + avg_flow[0]
        y2 = (bbox[3]) + avg_flow[1]

    return [x1, y1, x2, y2]
