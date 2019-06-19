from visualization.vtkVisualization import VTKPointCloud
from utils.visualization_utils import warp_points, generate_colors, draw_bbox, initVisualization, get_bbox_points_dense
from utils.reconstruction_utils import get_center_point
import pandas
import os
import numpy as np
import pycocotools.mask as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from visualization.visualize_mots import apply_mask


def visualize(points, colors=None):
    if colors is None:
        colors = np.tile([255, 255, 0], (len(points), 1))

    point_cloud_all = VTKPointCloud(points=np.asarray(points), colors=np.asarray(colors))
    vis_all = initVisualization(point_cloud_all)
    vis_all.interactor.Render()
    vis_all.interactor.Start()


def visualize_pointcloud(save_path):
    assert os.path.exists(save_path + 'all_coords.csv'), 'please compute pointcloud beforehand'

    all_coords = pandas.read_csv(save_path + 'all_coords.csv').to_records().tolist()
    all_coords = np.asarray(all_coords)[:, 1:]
    all_colors = pandas.read_csv(save_path + 'all_colors.csv').to_records().tolist()
    all_colors = np.asarray(all_colors)[:, 1:]
    point_cloud_all = VTKPointCloud(points=all_coords, colors=all_colors)
    vis_all = initVisualization(point_cloud_all)
    vis_all.interactor.Render()
    vis_all.interactor.Start()


def show_mask(img, mask, bbox=None):
    color = [255, 0, 0]
    binary_mask = cm.decode(mask)
    img = apply_mask(img, binary_mask, color)
    fig = plt.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = fig.subplots()
    ax.set_axis_off()
    ax.imshow(img)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


def show_detection(img, detection):
    fig = plt.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = fig.subplots()
    ax.set_axis_off()
    ax.imshow(img)
    bbox = detection['bbox']
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


def visualize_sequence_3D(config, tracks, sequence, ids=None):
    global_scene_visualization = {'points': [],
                                  'colors': []}
    colors = generate_colors()
    save_path = config.dir('pointcloud_savedir') + sequence + '/'

    for step in range(tracks.timesteps):

        for ref_id in tracks.get_active_tracks(step):

            if ids is not None:
                if not ref_id in ids:
                    continue

            if tracks.get_attribute(step, ref_id, 'global_3D_bbox') is not None:
                global_scene_visualization['points'].extend(
                    get_bbox_points_dense(config,
                                          tracks.get_attribute(step, ref_id, 'global_3D_bbox'),
                                          tracks.get_detection(step, ref_id)['class'])
                )
                color = np.asarray(colors[ref_id % len(colors)]) * 255
                global_scene_visualization['colors'].extend(np.tile(color, (1208, 1)))

            global_scene_visualization['points'].extend(tracks.get_attribute(step, ref_id, 'global_points'))
            global_scene_visualization['colors'].extend(tracks.get_attribute(step, ref_id, 'global_colors'))

    if os.path.exists(save_path + 'all_coords.csv'):
        all_coords = pandas.read_csv(save_path + 'all_coords.csv').to_records().tolist()
        all_coords = np.asarray(all_coords)[:, 1:]
        all_colors = pandas.read_csv(save_path + 'all_colors.csv').to_records().tolist()
        all_colors = np.asarray(all_colors)[:, 1:]

        global_scene_visualization['points'].extend(all_coords)
        global_scene_visualization['colors'].extend(all_colors)

    visualize(global_scene_visualization['points'], global_scene_visualization['colors'])

