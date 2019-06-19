from utils.geometry_utils import pose_vec2mat
import numpy as np
import matplotlib.pyplot as plt
import cv2
from visualization.vtkVisualization import VTKVisualization


def generate_colors():
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    import colorsys
    N = 30
    brightness = 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    perm = [15, 13, 25, 12, 19, 8, 22, 24, 29, 17, 28, 20, 2, 27, 11, 26, 21, 4, 3, 18, 9, 5, 14, 1, 16, 0, 23, 7,
            6, 10]
    colors = [colors[idx] for idx in perm]
    del colors[::2]
    return colors


def initVisualization(point_cloud):
    point_cloud.set_point_size(3)
    visualization = VTKVisualization()
    visualization.add_entity(point_cloud)
    visualization.init()
    return visualization


def show_flow(flow):
    # Use Hue, Saturation, Value colour model
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    plt.imshow(bgr)
    plt.show()


def warp_points(relative_poses, points):
    global_pose = pose_vec2mat([0, 0, 0])
    for pose in relative_poses:
            global_pose = np.matmul(global_pose, np.linalg.inv(pose_vec2mat(pose)))

    filler = np.ones(len(points))
    points_2d = np.column_stack((points[:, 0], points[:, 2], filler))
    warped_points = points_2d @ global_pose.T
    warped_points = np.column_stack((warped_points[:, 0], points[:, 1], warped_points[:, 1]))
    return warped_points


def draw_bbox(bbox_params, avg_h, avg_w, avg_l):
    half_w = avg_w / 2
    half_h = avg_h / 2
    half_l = avg_l / 2
    bbox_points = np.asarray(
        [[half_w, half_h, half_l], [-half_w, -half_h, half_l], [-half_w, half_h, half_l], [half_w, -half_h, half_l],
         [half_w, half_h, -half_l], [-half_w, -half_h, -half_l], [-half_w, half_h, -half_l],
         [half_w, -half_h, -half_l]])

    bbox_points = bbox_points.tolist()
    for i in range(150):
        bbox_points.append([bbox_points[0][0], bbox_points[0][1], bbox_points[0][2] - half_l * 2 / 150 * (i + 1)])
        bbox_points.append([bbox_points[1][0], bbox_points[1][1], bbox_points[1][2] - half_l * 2 /  150 * (i + 1)])
        bbox_points.append([bbox_points[2][0], bbox_points[2][1], bbox_points[2][2] - half_l * 2 /  150 * (i + 1)])
        bbox_points.append([bbox_points[3][0], bbox_points[3][1], bbox_points[3][2] - half_l * 2 /  150 * (i + 1)])

        bbox_points.append([bbox_points[0][0], bbox_points[0][1] - half_h * 2 /  150 * (i + 1), bbox_points[0][2]])
        bbox_points.append([bbox_points[2][0], bbox_points[2][1] - half_h * 2 /  150 * (i + 1), bbox_points[2][2]])
        bbox_points.append([bbox_points[4][0], bbox_points[4][1] - half_h * 2 /  150 * (i + 1), bbox_points[4][2]])
        bbox_points.append([bbox_points[6][0], bbox_points[6][1] - half_h * 2 /  150 * (i + 1), bbox_points[6][2]])

        bbox_points.append([bbox_points[0][0] - half_w * 2 / 150 * (i + 1), bbox_points[0][1], bbox_points[0][2]])
        bbox_points.append([bbox_points[3][0] - half_w * 2 /  150 * (i + 1), bbox_points[3][1], bbox_points[3][2]])
        bbox_points.append([bbox_points[4][0] - half_w * 2 / 150 * (i + 1), bbox_points[4][1], bbox_points[4][2]])
        bbox_points.append([bbox_points[7][0] - half_w * 2 / 150 * (i + 1), bbox_points[7][1], bbox_points[7][2]])

    bbox_points = np.asarray(bbox_points)

    bbox_points2 = np.column_stack((bbox_points[:, 0], bbox_points[:, 2]))
    bbox_points2 = bbox_points2 @ np.asarray(
        [[np.cos(bbox_params[3]), -np.sin(bbox_params[3])], [np.sin(bbox_params[3]), np.cos(bbox_params[3])]]).T
    bbox_points = np.column_stack((bbox_points2[:, 0], bbox_points[:, 1], bbox_points2[:, 1]))
    bbox_points = bbox_points + bbox_params[0:3]

    return bbox_points


def get_bbox_points_dense(config, bbox_params, object_class):
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

    bbox_points = bbox_points.tolist()
    for i in range(100):
        bbox_points.append([bbox_points[0][0], bbox_points[0][1], bbox_points[0][2] - half_l * 2 / 100 * (i + 1)])
        bbox_points.append([bbox_points[1][0], bbox_points[1][1], bbox_points[1][2] - half_l * 2 /  100 * (i + 1)])
        bbox_points.append([bbox_points[2][0], bbox_points[2][1], bbox_points[2][2] - half_l * 2 /  100 * (i + 1)])
        bbox_points.append([bbox_points[3][0], bbox_points[3][1], bbox_points[3][2] - half_l * 2 /  100 * (i + 1)])

        bbox_points.append([bbox_points[0][0], bbox_points[0][1] - half_h * 2 /  100 * (i + 1), bbox_points[0][2]])
        bbox_points.append([bbox_points[2][0], bbox_points[2][1] - half_h * 2 /  100 * (i + 1), bbox_points[2][2]])
        bbox_points.append([bbox_points[4][0], bbox_points[4][1] - half_h * 2 /  100 * (i + 1), bbox_points[4][2]])
        bbox_points.append([bbox_points[6][0], bbox_points[6][1] - half_h * 2 /  100 * (i + 1), bbox_points[6][2]])

        bbox_points.append([bbox_points[0][0] - half_w * 2 / 100 * (i + 1), bbox_points[0][1], bbox_points[0][2]])
        bbox_points.append([bbox_points[3][0] - half_w * 2 /  100 * (i + 1), bbox_points[3][1], bbox_points[3][2]])
        bbox_points.append([bbox_points[4][0] - half_w * 2 / 100 * (i + 1), bbox_points[4][1], bbox_points[4][2]])
        bbox_points.append([bbox_points[7][0] - half_w * 2 / 100 * (i + 1), bbox_points[7][1], bbox_points[7][2]])

    bbox_points = np.asarray(bbox_points)

    bbox_points2 = np.column_stack((bbox_points[:, 0], bbox_points[:, 2]))
    bbox_points2 = bbox_points2 @ np.asarray(
        [[np.cos(bbox_params[3]), -np.sin(bbox_params[3])], [np.sin(bbox_params[3]), np.cos(bbox_params[3])]]).T
    bbox_points = np.column_stack((bbox_points2[:, 0], bbox_points[:, 1], bbox_points2[:, 1]))
    bbox_points = bbox_points + bbox_params[0:3]

    return bbox_points
