import numpy as np
import cv2
import collections
import os
import scipy.misc, scipy.stats
import matplotlib.pyplot as plt
import skimage.transform
from file_io.io_utils import readFloat


def euler2mat(z, y, x):
    """Converts euler angles to rotation matrix
     TODO: remove the dimension for 'N' (deprecated for converting all source
           poses altogether)
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        z: rotation angle along z axis (in radians) -- size = [B, N]
        y: rotation angle along y axis (in radians) -- size = [B, N]
        x: rotation angle along x axis (in radians) -- size = [B, N]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """
    B =np.shape(z)[0]
    N = 1
    z =np.clip(z, -np.pi, np.pi)
    y =np.clip(y, -np.pi, np.pi)
    x =np.clip(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z =np.expand_dims(np.expand_dims(z, -1), -1)
    y =np.expand_dims(np.expand_dims(y, -1), -1)
    x =np.expand_dims(np.expand_dims(x, -1), -1)

    zeros =np.zeros([B, N, 1, 1])
    ones =np.ones([B, N, 1, 1])

    cosz =np.cos(z)
    sinz =np.sin(z)
    rotz_1 =np.concatenate([cosz, -sinz, zeros], axis=3)
    rotz_2 =np.concatenate([sinz, cosz, zeros], axis=3)
    rotz_3 =np.concatenate([zeros, zeros, ones], axis=3)
    zmat =np.concatenate([rotz_1, rotz_2, rotz_3], axis=2)

    cosy =np.cos(y)
    siny =np.sin(y)
    roty_1 =np.concatenate([cosy, zeros, siny], axis=3)
    roty_2 =np.concatenate([zeros, ones, zeros], axis=3)
    roty_3 =np.concatenate([-siny, zeros, cosy], axis=3)
    ymat =np.concatenate([roty_1, roty_2, roty_3], axis=2)

    cosx =np.cos(x)
    sinx =np.sin(x)
    rotx_1 =np.concatenate([ones, zeros, zeros], axis=3)
    rotx_2 =np.concatenate([zeros, cosx, -sinx], axis=3)
    rotx_3 =np.concatenate([zeros, sinx, cosx], axis=3)
    xmat =np.concatenate([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat =np.matmul(np.matmul(zmat, ymat), xmat)
    return rotMat


def pose_vec2mat(vec):
    """Converts parameters to transformation matrix
    Args:
        vec: 6DoF/3DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [1, 6] or theta, tx, ty -- [3]
    Returns:
        A transformation matrix -- [1, 4, 4]
    """
    if len(vec) == 3:
        return np.array([np.cos(vec[2]), -np.sin(vec[2]), vec[0],
                         np.sin(vec[2]), np.cos(vec[2]), vec[1],
                         0., 0., 1.], dtype=np.float).reshape((3, 3))
    else:
        batch_size = np.shape(vec)[0]
        translation =vec[0, 0:3] * -np.pi
        translation =np.expand_dims([translation], -1)
        rx =[[vec[0, 3]* -1]]
        ry =[[vec[0, 4]* -1]]
        rz =[[vec[0, 5]* -1]]
        rot_mat = euler2mat(rz, ry, rx)
        rot_mat =np.squeeze([rot_mat], axis=[1])
        rot_mat = [rot_mat]
        filler =[[[0., 0., 0., 1.]]]
        filler =np.tile(filler, [batch_size, 1, 1])
        transform_mat =np.concatenate([rot_mat, translation], axis=2)
        transform_mat =np.concatenate([transform_mat, filler], axis=1)
        return transform_mat


def computePointImage(depth_img, calibration_params):
    pix_y = np.arange(0, depth_img.shape[0])
    pix_x = np.arange(0, depth_img.shape[1])

    x, y = np.meshgrid(pix_x, pix_y)

    x = x.flatten()
    y = y.flatten()

    coords = np.stack([x, y, np.ones(x.shape)], axis=-1)
    coords = coords @ np.linalg.inv(calibration_params.camera_matrix.T)
    coords = np.expand_dims(depth_img.flatten(), axis=-1) * coords

    coord_img = np.reshape(coords, (depth_img.shape[0], depth_img.shape[1], 3))

    return coord_img


def transformPointImage(point_img, pose):
    coords = point_img.reshape((point_img.shape[0] * point_img.shape[1], point_img.shape[2]))

    # transform points according to estimated camera motion (pose)
    transformation_matrix = pose
    coords = np.column_stack((coords, np.ones(len(coords))))
    coords = coords @ transformation_matrix.T
    coords = coords[:, 0:3]

    coord_img = np.reshape(coords, point_img.shape)
    return coord_img


def computeColorMatrix(source_img, img_shape, coords):
    source_img = cv2.bilateralFilter(source_img, 9, 75, 75)
    source_img = cv2.resize(np.asarray(source_img), img_shape)
    return np.asarray(source_img).flatten().reshape(coords.shape)


def diff(x, y):
    transmat_y = pose_vec2mat(np.expand_dims(y, 0))[0]
    translation = np.dot(x[0:3, 0:3], transmat_y[0:3, 3]) + x[0:3, 3]
    rotation = np.matmul(x[0:3, 0:3], transmat_y[0:3, 0:3])
    filler = np.array([[0.0, 0.0, 0.0, 1.0]])
    transform_mat = np.concatenate([rotation, np.expand_dims(translation, -1)], axis=1)
    transform_mat = np.concatenate([transform_mat, filler], axis=0)
    return transform_mat


def process_poses(poses_raw, type):
    if type == 'geonet':
        f_poses = np.zeros((len(poses_raw)+2, 4, 4))
        init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        # cover corner cases
        f_poses[0] = pose_vec2mat(np.expand_dims(poses_raw[0][0], 0))[0]
        f_poses[1] = pose_vec2mat(np.expand_dims(poses_raw[0][1], 0))[0]
        f_poses[2] = init

        #compute global poses
        for i in range(2, len(poses_raw)):
            f_poses[i + 1] = diff(f_poses[i], poses_raw[i-2][3])

        # cover corner cases
        f_poses[-3] = diff(f_poses[len(poses_raw)-1], poses_raw[len(poses_raw) - 2][3])
        f_poses[-2] = diff(f_poses[len(poses_raw)], poses_raw[len(poses_raw)-1][3])
        f_poses[-1] = diff(f_poses[len(poses_raw)+1], poses_raw[len(poses_raw)-1][4])
    else:
        f_poses = np.zeros((len(poses_raw), 4, 4))
        filler = np.array([[0.0, 0.0, 0.0, 1.0]])
        for i in range(len(f_poses)):
            f_poses[i] = np.concatenate([poses_raw[i].reshape(3, 4), filler], axis=0)

    return f_poses


def flow_warp_idx(x, y, flow, shape):
    flow_point = flow[x, y]
    warped_x = np.maximum(np.minimum(x + int(flow_point[1]), shape[0]-1), 0)
    warped_y = np.maximum(np.minimum(y + int(flow_point[0]), shape[1]-1), 0)
    return warped_x, warped_y


# def get_posewarped_pointcloud(pointcloud_t1, poses_t0_t1):
#     R_0_inv = np.linalg.inv(poses_t0_t1[0][0:3, 0:3])
#     R_1 = poses_t0_t1[1][0:3, 0:3]
#     t_0 = poses_t0_t1[0][0:3, 3]
#     t_1 = poses_t0_t1[1][0:3, 3]
#     filler = np.array([[0.0, 0.0, 0.0, 1.0]])
#     relative_pose_matrix = np.concatenate([np.matmul(R_0_inv, R_1), np.expand_dims(np.dot(R_0_inv, (t_1 - t_0)), -1)],
#                                           axis=1)
#     relative_pose_matrix = np.concatenate([relative_pose_matrix, filler], axis=0)
#
#     pointcloud_t1_in_t0 = np.column_stack((pointcloud_t1, np.ones(len(pointcloud_t1))))
#     pointcloud_t1_in_t0 = pointcloud_t1_in_t0 @ relative_pose_matrix.T
#     pointcloud_t1_in_t0 = pointcloud_t1_in_t0[:, 0:3]
#     return pointcloud_t1_in_t0


def flowwarp_pointimg(point_img_t1, point_img_t0, flow, occ):
    flow_img = np.zeros(point_img_t1.shape)
    for x in range(flow_img.shape[0]):
        for y in range(flow_img.shape[1]):
            if occ[x, y] or any(np.isnan(point_img_t0[x, y])):
                flow_img[x, y] = np.NaN
            else:
                flow_img[x, y] = point_img_t1[flow_warp_idx(x, y, flow, point_img_t1.shape)]

    return flow_img


# def compute_jacobian(proj_mat, point):
#     w = proj_mat[2, 0] * point[0] + proj_mat[2, 1] * point[1] + proj_mat[2, 2] * point[2] + proj_mat[2, 3]
#     w_2 = np.square(w)
#
#     F = np.zeros((2, 3))
#     for i in range(F.shape[0]):
#         sum_row_by_point = proj_mat[i, 0] * point[0] + proj_mat[i, 1] * point[1] + proj_mat[i, 2] * point[2] + proj_mat[i, 3]
#         for j in range(F.shape[1]):
#             F[i, j] = proj_mat[i, j] / w - sum_row_by_point * proj_mat[2, j] / w_2
#
#     return F


# def compute_uncertainty_map(point_img):
#     uncertainty_map = np.zeros((point_img.shape[0], point_img.shape[1], 3, 3))
#     for x in range(point_img.shape[0]):
#         for y in range(point_img.shape[1]):
#             point = point_img[x, y]
#             if any(np.isnan(point)):
#                 uncertainty_map[x, y] = np.NaN
#             else:
#                 FL = compute_jacobian(cam_projection_matrix_l, point)
#                 FR = compute_jacobian(cam_projection_matrix_r, point)
#
#                 c2d_mat = np.linalg.inv(np.identity(2) * 0.5)
#                 FLTFL = np.matmul(np.matmul(FL.transpose(), c2d_mat), FL)
#                 FRTFR = np.matmul(np.matmul(FR.transpose(), c2d_mat), FR)
#
#                 cov_uncertainty = np.linalg.inv(np.add(FLTFL, FRTFR))
#                 uncertainty_map[x, y] = cov_uncertainty
#     return uncertainty_map


def compute_uncertainty_map(point_img):
    uncertainty_map = np.square(point_img[:,:,2])
    uncertainty_map = uncertainty_map / np.max(np.where(np.isnan(uncertainty_map), 0., uncertainty_map))
    uncertainty_map = np.square(uncertainty_map)
    return uncertainty_map


# def get_dynamic_region_mask(point_img, diff_pose_flow, uncertainty_map):
#     dynamic_region_mask = np.zeros((point_img.shape[0], point_img.shape[1]))
#
#     def is_not_static(mu, sigma):
#         #sigma_mu = np.split(sigma_mu, [9])
#         #sigma = np.reshape(sigma_mu[0], (3, 3))
#         #mu = sigma_mu[1]
#         x = np.asarray([0., 0., 0.])
#         m_dist_x = np.dot((x - mu).transpose(), np.linalg.inv(sigma))
#         m_dist_x = np.dot(m_dist_x, (x - mu))
#         return scipy.stats.chi2.cdf(m_dist_x, 3)
#
#     #shape = (uncertainty_map.shape[0], uncertainty_map.shape[1], uncertainty_map.shape[2] * uncertainty_map.shape[3])
#     #merged = np.concatenate((uncertainty_map.reshape(shape), diff_pose_flow), axis=2)
#     #dynamic_region_mask = np.where(np.logical_or(is_not_static(merged), np.isnan(merged)), True, merged)
#     for x in range(diff_pose_flow.shape[0]):
#         for y in range(diff_pose_flow.shape[1]):
#             if not any(np.isnan(diff_pose_flow[x,y])):
#                 dynamic_region_mask[x,y] = is_not_static(diff_pose_flow[x,y], uncertainty_map[x,y])
#
#     return dynamic_region_mask


def get_dynamic_region_mask(sceneflow, uncertainty_map):
    dynamic_region_mask = np.zeros((sceneflow.shape[0], sceneflow.shape[1]), dtype=np.bool)

    diff_pose_flow_t2_t1 = np.linalg.norm(sceneflow, 2, axis=2)
    diff_pose_flow_t2_t1 = np.where(diff_pose_flow_t2_t1 > 5, np.NaN, diff_pose_flow_t2_t1)

    diff_pose_flow_scaled = diff_pose_flow_t2_t1#np.divide(diff_pose_flow_t2_t1, uncertainty_map)
    median = np.median(diff_pose_flow_scaled[~np.isnan(diff_pose_flow_scaled)])
    dynamic_region_mask = np.where(np.logical_or(diff_pose_flow_scaled > median, np.isnan(diff_pose_flow_scaled)),
                                   True, dynamic_region_mask)

    # fig = plt.figure(figsize=(8, 8))
    # print(median)
    # fig.add_subplot(3, 1, 1)
    # plt.imshow(diff_pose_flow_scaled)
    #
    # fig.add_subplot(3, 1, 2)
    # plt.imshow(diff_pose_flow_t2_t1)
    #
    # fig.add_subplot(3, 1, 3)
    # plt.imshow(uncertainty_map)
    # plt.show()

    return dynamic_region_mask


def filter_dynamic_regions(point_imgs, sceneflow, imgs):
    uncertainty_map = compute_uncertainty_map(point_imgs[1])
    point_img = point_imgs[0]
    dynamic_region_mask = get_dynamic_region_mask(sceneflow, uncertainty_map)

    # converged = False
    #
    # while not converged:
    #     dynamic_region_mask = get_dynamic_region_mask(point_img, flow_img, uncertainty_map)
    #
    #     point_img_serialized = point_img
    #     flow_img_serialized = flow_img
    #
    #     point_img_serialized[dynamic_region_mask] = np.NaN
    #     flow_img_serialized[dynamic_region_mask] = np.NaN
    #
    #     plt.imshow(point_img_serialized)
    #     plt.show()
    #
    #     point_img_serialized = np.where(np.isnan(point_img_serialized), np.zeros(point_img_serialized.shape[2]), point_img_serialized)
    #     flow_img_serialized = np.where(np.isnan(flow_img_serialized), np.zeros(flow_img_serialized.shape[2]), flow_img_serialized)
    #
    #     shape_serialized = (flow_img.shape[0] * flow_img.shape[1], flow_img.shape[2])
    #     flow_img_serialized = flow_img_serialized.reshape(shape_serialized)
    #     flow_img_serialized = np.column_stack((flow_img_serialized, np.ones(len(flow_img_serialized))))
    #     point_img_serialized = point_img_serialized.reshape(shape_serialized)
    #     point_img_serialized = np.column_stack((point_img_serialized, np.ones(len(point_img_serialized))))
    #
    #     pose_residual = optimize_pose(flow_img_serialized, point_img_serialized)
    #
    #     #pose_residual = tf.Session().run(tf.linalg.lstsq(point_img_serialized, flow_img_serialized))
    #
    #     #pose_residual = tf.inv(tf.matmul(tf.matmul(tf.transpose(point_img), uncertainty_map), point_img))
    #
    #     point_img = transformPointImage(point_img, pose_residual)

    point_img[dynamic_region_mask] = np.NaN
    # plt.imshow(point_img)
    # plt.show()
    return point_img


def get_filtered_points(sequence, idx, config, points_list, sceneflow_list, img_list):
    imgs = [img_list[idx], img_list[idx + 1]]

    point_img_t0 = np.asarray(readFloat(config.point_imgs_savedir + sequence + '/' + points_list[idx]))
    point_img_t1 = np.asarray(readFloat(config.point_imgs_savedir + sequence + '/' + points_list[idx+1]))
    point_imgs = [point_img_t0, point_img_t1]

    sceneflow = np.asarray(readFloat(config.sceneflow_savedir + sequence + '/' + sceneflow_list[idx]))
    import pickle
    # remove sky with deeplabv3 output
    skymask_list = pickle.load(open(config.skymask_savedir + sequence + '/skymask.pkl', 'rb'))
    semsec_im = skymask_list[idx]
    semsec_mask = np.where(semsec_im != 10, 0, semsec_im).astype('uint16')
    semsec_mask = skimage.transform.resize(semsec_mask, (point_img_t0.shape[0], point_img_t0.shape[1]), order=0)
    semsec_mask = cv2.dilate(semsec_mask, np.ones((11, 11)), iterations=5)
    semsec_mask = np.where(semsec_mask == 10, 1, semsec_mask).astype('bool')
    point_imgs[1][semsec_mask] = np.NaN

    # point_imgs[1] = filter_dynamic_regions(point_imgs, sceneflow, imgs)
    #
    #
    #
    # anno_im = skimage.file_io.imread('./data/KITTI_tracking/train/instances/0002/' + format(idx, "06") + '.png')
    # anno_im = np.uint8(anno_im)
    # anno_im = cv2.dilate(anno_im, np.ones((5, 5)), iterations=1)
    # anno_mask = anno_im.astype('bool')
    #
    # point_imgs[1][anno_mask] = np.NaN

    # only add points not seen before
    occ_fwd_list = sorted(filter(lambda x: 'occ[0].fwd' in x, os.listdir(config.flow_disp_savedir + sequence + '/')))
    occ_t1_t0 = readFloat(config.flow_disp_savedir + sequence + '/' + occ_fwd_list[idx])[:, :, 0].astype(np.bool)
    if idx != 119:
       point_imgs[1][np.logical_not(occ_t1_t0)] = np.NaN

    return point_imgs[1]


def filter_nans(coords, colors):
    merged = np.column_stack(np.asarray((coords, colors)))
    mask = [np.isnan(elem).any() for elem in merged]
    merged = merged[np.logical_not(mask)]
    return merged[:, :3], merged[:, 3:]


def bbox_iou(box1, box2):
    x0_min = min(box1[0], box2[0])
    x0_max = max(box1[0], box2[0])
    y0_min = min(box1[1], box2[1])
    y0_max = max(box1[1], box2[1])
    x1_min = min(box1[2], box2[2])
    x1_max = max(box1[2], box2[2])
    y1_min = min(box1[3], box2[3])
    y1_max = max(box1[3], box2[3])
    I = max(x1_min - x0_max, 0) * max(y1_min - y0_max, 0)
    U = (x1_max - x0_min) * (y1_max - y0_min)
    if U == 0:
        return 0.0
    else:
        return I / U

