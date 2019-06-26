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


def flowwarp_pointimg(point_img_t1, point_img_t0, flow, occ):
    flow_img = np.zeros(point_img_t1.shape)
    for x in range(flow_img.shape[0]):
        for y in range(flow_img.shape[1]):
            if occ[x, y] or any(np.isnan(point_img_t0[x, y])):
                flow_img[x, y] = np.NaN
            else:
                flow_img[x, y] = point_img_t1[flow_warp_idx(x, y, flow, point_img_t1.shape)]

    return flow_img


def compute_uncertainty_map(point_img):
    uncertainty_map = np.square(point_img[:,:,2])
    uncertainty_map = uncertainty_map / np.max(np.where(np.isnan(uncertainty_map), 0., uncertainty_map))
    uncertainty_map = np.square(uncertainty_map)
    return uncertainty_map


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

