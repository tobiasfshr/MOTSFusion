import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(suppress=True)
from matplotlib import cm
from sklearn.neighbors import LocalOutlierFactor
from imblearn.under_sampling import ClusterCentroids
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from utils.reconstruction_utils import warp_points, get_position_covariance, get_points_from_masks, get_dynamic_transform, get_center_point
from visualization.visualize import visualize
np.random.seed(42)


def remove_outliers(object_points):
    if len(object_points) > 100:
        points_t0 = object_points[:, 0]
        points_t1 = object_points[:, 1]
        mask = np.zeros(len(object_points), dtype=np.bool)
        # fit the model for outlier detection (default)
        for points in [points_t0, points_t1]:
            clf = LocalOutlierFactor(n_neighbors=20)
            clf.fit_predict(points)
            X_scores = clf.negative_outlier_factor_
            X_scores = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
            median_score = np.median(X_scores)
            mask = np.logical_or([X_scores[i] > median_score for i in range(len(points))], mask)

            # print(X_scores)
            # print('median_score ', mean_score)
            # plt.title("Local Outlier Factor (LOF)")
            # plt.scatter(points[:, 0], points[:, 2], color='k', s=3., label='Data points')
            # # plot circles with radius proportional to the outlier scores
            # plt.scatter(points[:, 0], points[:, 2], s=1000 * X_scores, edgecolors='r',
            #             facecolors='none', label='Outlier scores')
            # plt.axis('tight')
            # plt.xlim((-5, 5))
            # plt.ylim((-5, 5))
            # legend = plt.legend(loc='upper left')
            # legend.legendHandles[0]._sizes = [10]
            # legend.legendHandles[1]._sizes = [20]

            # points = points[np.logical_not(mask)]
            # X_scores = X_scores[np.logical_not(mask)]
            # plt.title("Local Outlier Factor (LOF)")
            # plt.scatter(points[:, 0], points[:, 2], color='k', s=3., label='Data points')
            # # plot circles with radius proportional to the outlier scores
            # plt.scatter(points[:, 0], points[:, 2], s=1000 * X_scores, edgecolors='r',
            #             facecolors='none', label='Outlier scores')
            # plt.axis('tight')
            # plt.xlim((-5, 5))
            # plt.ylim((-5, 5))
            # legend = plt.legend(loc='upper left')
            # legend.legendHandles[0]._sizes = [10]
            # legend.legendHandles[1]._sizes = [20]
            # plt.show()
        if len(object_points[np.logical_not(mask)]) > 10:
            object_points = object_points[np.logical_not(mask)]

    return object_points


def sparsify(object_points):
    if len(object_points) > 200:
        undersample = np.random.uniform(0, len(object_points), 200).astype(np.int)
        object_points = np.take(object_points, undersample, axis=0)

    return object_points


def create_dynamic_transforms(config, tracks, flow, point_imgs, raw_imgs, calibration_params):
    tracks.add_new_attribute('dynamic_transforms')
    tracks.add_new_attribute('transform_covariances')
    tracks.add_new_attribute('global_positions')
    tracks.add_new_attribute('position_covariances')
    tracks.add_new_attribute('global_points')

    if config.bool('debug'):
        tracks.add_new_attribute('global_pointclouds')
        tracks.add_new_attribute('global_colors')
        tracks.add_new_attribute('global_3D_bbox')
        tracks.add_new_attribute('global_points_unprocessed')

    for step in range(tracks.timesteps-1):
        for id in tracks.get_active_tracks(step):
            if id in tracks.get_active_tracks(step+1):
                mask_t0 = tracks.get_mask(step, id, postprocess=True)

                mask_t1 = tracks.get_mask(step+1, id, postprocess=True)

                points, colors = get_points_from_masks(mask_t0, mask_t1, point_imgs[step], point_imgs[step+1], flow[step+1], raw_imgs[step], raw_imgs[step+1], calibration_params)

                if len(points) > 1:
                    points_processed = remove_outliers(sparsify(points))

                    # points_vis = np.concatenate((points[:, 0], points_processed[:, 0]), axis=0)
                    # colors_vis = np.concatenate((colors[:, 0], np.tile([255, 255, 0], (len(points_processed), 1))), axis=0)
                    #
                    # visualize(points_vis, colors_vis)

                    if not tracks.is_active(step-1, id):
                        tracks.set_attribute(step, id, 'position_covariances', get_position_covariance(points_processed[:, 0], calibration_params))
                        tracks.set_attribute(step, id, 'global_positions', np.mean(points_processed[:, 0], axis=0))
                        tracks.set_attribute(step, id, 'global_points', points_processed[:, 0])

                        if config.bool('debug'):
                            tracks.set_attribute(step, id, 'global_3D_bbox', get_center_point(config, points_processed[:, 0], tracks.get_detection(step, id)['class']))
                            tracks.set_attribute(step, id, 'global_pointclouds', points[:, 0])
                            tracks.set_attribute(step, id, 'global_points_unprocessed', points[:, 0])
                            tracks.set_attribute(step, id, 'global_colors', colors[:, 0])

                    dynamic_transform, transform_covariance = get_dynamic_transform(points_processed)

                    tracks.set_attribute(step + 1, id, 'dynamic_transforms', dynamic_transform)
                    tracks.set_attribute(step + 1, id, 'transform_covariances', transform_covariance)
                    tracks.set_attribute(step + 1, id, 'global_positions', np.mean(points_processed[:, 1], axis=0))
                    tracks.set_attribute(step + 1, id, 'position_covariances', get_position_covariance(points_processed[:, 1], calibration_params))
                    tracks.set_attribute(step + 1, id, 'global_points', points_processed[:, 1])

                    if config.bool('debug'):
                        tracks.set_attribute(step + 1, id, 'global_3D_bbox', get_center_point(config, points_processed[:, 1], tracks.get_detection(step, id)['class'], points_processed[:, 0]))
                        tracks.set_attribute(step + 1, id, 'global_pointclouds', warp_points(tracks.get_track_attribute(id, 'dynamic_transforms'), id, points_processed[:, 1]))
                        tracks.set_attribute(step + 1, id, 'global_points_unprocessed', points[:, 1])
                        tracks.set_attribute(step + 1, id, 'global_colors', colors[:, 1])
                else:
                    points = np.concatenate((np.expand_dims(point_imgs[step+1][mask_t1.astype(np.bool)], axis=1), np.expand_dims(point_imgs[step+1][mask_t1.astype(np.bool)], axis=1)), axis=1)
                    colors = np.concatenate((np.expand_dims(raw_imgs[step+1][mask_t1.astype(np.bool)], axis=1), np.expand_dims(raw_imgs[step+1][mask_t1.astype(np.bool)], axis=1)), axis=1)
                    points_processed = remove_outliers(sparsify(points))

                    if not tracks.is_active(step - 1, id):
                        tracks.set_attribute(step, id, 'position_covariances', get_position_covariance(points_processed[:, 0], calibration_params))
                        tracks.set_attribute(step, id, 'global_positions', np.mean(points_processed[:, 0], axis=0))
                        tracks.set_attribute(step, id, 'global_points', points_processed[:, 0])

                        if config.bool('debug'):
                            tracks.set_attribute(step, id, 'global_3D_bbox', get_center_point(config, points_processed[:, 0], tracks.get_detection(step, id)['class']))
                            tracks.set_attribute(step, id, 'global_pointclouds', points[:, 0])
                            tracks.set_attribute(step, id, 'global_points_unprocessed', points[:, 0])
                            tracks.set_attribute(step, id, 'global_colors', colors[:, 0])

                    dynamic_transform = np.asarray([0, 0, 0])
                    transform_covariance = np.eye(3)
                    tracks.set_attribute(step + 1, id, 'dynamic_transforms', dynamic_transform)
                    tracks.set_attribute(step + 1, id, 'transform_covariances', transform_covariance)

                    tracks.set_attribute(step + 1, id, 'global_positions', np.mean(points_processed[:, 1], axis=0))
                    tracks.set_attribute(step + 1, id, 'position_covariances', get_position_covariance(points_processed[:, 1], calibration_params))
                    tracks.set_attribute(step + 1, id, 'global_points', points_processed[:, 1])

                    if config.bool('debug'):
                        tracks.set_attribute(step + 1, id, 'global_3D_bbox', get_center_point(config, points_processed[:, 1], tracks.get_detection(step, id)['class'], points_processed[:, 0]))
                        tracks.set_attribute(step + 1, id, 'global_pointclouds', points[:, 1])
                        tracks.set_attribute(step + 1, id, 'global_points_unprocessed', points[:, 1])
                        tracks.set_attribute(step + 1, id, 'global_colors', colors[:, 1])
            else:
                if not tracks.is_active(step - 1, id):
                    mask_t0 = tracks.get_mask(step, id, postprocess=True).astype(np.bool)
                    points = point_imgs[step][mask_t0]
                    colors = raw_imgs[step][mask_t0]
                    points_processed = remove_outliers(sparsify(np.concatenate((np.expand_dims(points, axis=1), np.expand_dims(points, axis=1)), axis=1)))
                    dynamic_transform = np.asarray([0, 0, 0])
                    transform_covariance = np.eye(3)
                    tracks.set_attribute(step + 1, id, 'dynamic_transforms', dynamic_transform)
                    tracks.set_attribute(step + 1, id, 'transform_covariances', transform_covariance)
                    tracks.set_attribute(step, id, 'position_covariances', get_position_covariance(points_processed[:, 0], calibration_params))
                    tracks.set_attribute(step, id, 'global_positions', np.mean(points_processed[:, 0], axis=0))
                    tracks.set_attribute(step, id, 'global_points', points_processed[:, 0])

                    if config.bool('debug'):
                        tracks.set_attribute(step, id, 'global_3D_bbox', get_center_point(config, points_processed[:, 0], tracks.get_detection(step, id)['class']))
                        tracks.set_attribute(step, id, 'global_pointclouds', points)
                        tracks.set_attribute(step, id, 'global_points_unprocessed', points)
                        tracks.set_attribute(step, id, 'global_colors', colors)

    return tracks


