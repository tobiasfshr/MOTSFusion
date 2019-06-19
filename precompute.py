from external.ORB_SLAM2.compute_poses import compute_poses_orb
from external.netdef_models.compute_disp_flow import compute_flow_and_disp
#from external.DeepLabv3.compute_sky_mask import compute_sky_mask

#from visualization.visualize import visualize_pointcloud
from utils.calibration_params import CalibrationParameters
from file_io.io_utils import readFloat, readFlow, writeFloat
from utils.geometry_utils import flowwarp_pointimg, computePointImage, transformPointImage, process_poses, get_filtered_points, filter_nans, computeColorMatrix
from config import Config
from file_io.import_utils import import_detections
from external.BB2SegNet.segment import compute_segmentations, refinement_net_init
from eval.mots_eval.mots_common.io import load_seqmap

import argparse
import skimage.io
import skimage.transform
import pandas
import pickle
import os
import numpy as np


# def compute_sceneflow(config, sequence):
#     assert os.path.exists(config.dir('point_imgs_savedir')), "Run point img computation before computing pointcloud."
#     assert os.path.exists(config.dir('skymask_savedir')), "Run sky estimation (deeplabv3) before computing pointcloud."
#     skymask_list = pickle.load(open(config.skymask_savedir + sequence + '/skymask.pkl', 'rb'))
#
#     if not os.path.exists(config.dir('sceneflow_savedir') + sequence + '/'):
#         os.makedirs(config.dir('sceneflow_savedir') + sequence + '/')
#
#     # get filelists
#     points_list = sorted(os.listdir(config.dir('point_imgs_savedir') + sequence + '/'))
#     flow_fwd_list = sorted(filter(lambda x: 'flow[0].fwd' in x, os.listdir(config.dir('flow_disp_savedir') + sequence + '/')))
#     occ_fwd_list = sorted(filter(lambda x: 'occ[0].fwd' in x, os.listdir(config.dir('flow_disp_savedir') + sequence + '/')))
#
#     for i in range(len(points_list)-1):
#         # load points
#         point_img_t0 = np.asarray(readFloat(config.dir('point_imgs_savedir') + sequence + '/' + points_list[i]))
#         point_img_t1 = np.asarray(readFloat(config.dir('point_imgs_savedir') + sequence + '/' + points_list[i + 1]))
#
#         # remove sky with deeplabv3 output
#         semsec_im = skymask_list[i]
#         semsec_mask = np.where(semsec_im != 10, 0, semsec_im).astype('uint16')
#         semsec_mask = skimage.transform.resize(semsec_mask, (point_img_t0.shape[0], point_img_t0.shape[1]), order=0)
#         semsec_mask = np.where(semsec_mask == 10, 1, semsec_mask).astype('bool')
#         point_img_t0[semsec_mask] = np.NaN
#
#         flow_t1_t0 = readFlow(config.dir('flow_disp_savedir') + sequence + '/' + flow_fwd_list[i])
#         occ_t1_t0 = readFloat(config.dir('flow_disp_savedir') + sequence + '/' + occ_fwd_list[i])[:, :, 0].astype(np.bool)
#
#         flow_img = flowwarp_pointimg(point_img_t1, point_img_t0, flow_t1_t0, occ_t1_t0)
#         scene_flow = np.subtract(point_img_t0, flow_img)
#         writeFloat(config.dir('sceneflow_savedir') + sequence + '/' + points_list[i], scene_flow)


def compute_point_imgs(config, sequence, calib_params):
    assert os.path.exists(config.dir('flow_disp_savedir')), "Run flow and disp estimation (netdef) before computing pointcloud."
    assert os.path.exists(config.dir('orb_pose_savedir')) or os.path.exists(config.geo_pose_savedir), "Run pose estimation (geonet, orbslam) before computing pointcloud."

    if not os.path.exists(config.dir('point_imgs_savedir') + sequence + '/'):
        os.makedirs(config.dir('point_imgs_savedir') + sequence + '/')

    if config.string('use_pose') == 'geonet':
        pose_list = pickle.load(open(config.dir('geo_pose_savedir') + sequence + '/out_poses.pkl', 'rb'))
    else:
        pose_list = np.genfromtxt(config.dir('orb_pose_savedir') + sequence + '/CameraTrajectory.txt', dtype='str')
    pose_list = process_poses(pose_list, config.string('use_pose'))

    disp_list = sorted(filter(lambda x: 'png.disp' in x, os.listdir(config.dir('flow_disp_savedir') + sequence + '/')))

    for i in range(len(disp_list)):

        disp = np.asarray(readFloat(config.dir('flow_disp_savedir') + sequence + '/' + disp_list[i]))
        depth = calib_params.bf / (disp * -1)
        point_img = computePointImage(depth, calib_params)
        point_img = transformPointImage(point_img, pose=pose_list[i])

        writeFloat(config.dir('point_imgs_savedir') + sequence + '/' + disp_list[i], point_img)


# def create_pointcloud(config, sequence):
#     assert os.path.exists(config.dir('point_imgs_savedir')), "Run point img computation before computing pointcloud."
#     assert os.path.exists(config.dir('sceneflow_savedir')), "Run scene flow computation before computing pointcloud."
#
#     if not os.path.exists(config.dir('pointcloud_savedir') + sequence + '/'):
#         os.makedirs(config.dir('pointcloud_savedir') + sequence + '/')
#
#     # img files
#     img_list = [skimage.io.imread(config.dir('data_dir') + 'images/' + sequence + '/image_2/' + file)
#                 for file in sorted(os.listdir(config.dir('data_dir') + 'images/' + sequence + '/image_2/'))]
#
#     # get filelists
#     points_list = sorted(os.listdir(config.dir('point_imgs_savedir') + sequence + '/'))
#     sceneflow_list = sorted(os.listdir(config.dir('sceneflow_savedir') + sequence + '/'))
#
#     all_coords = []
#     all_colors = []
#
#     for idx in reversed(range(len(img_list)-1)):
#         print(idx)
#         filtered_point_img = get_filtered_points(sequence, idx, config, points_list, sceneflow_list, img_list)
#
#         coords = filtered_point_img.reshape((filtered_point_img.shape[0] * filtered_point_img.shape[1],
#                                              filtered_point_img.shape[2]))
#
#         colors = computeColorMatrix(img_list[idx], (filtered_point_img.shape[1], filtered_point_img.shape[0]), coords)
#         coords, colors = filter_nans(coords, colors)
#
#         all_coords.extend(coords)
#         all_colors.extend(colors)
#
#     print('saving...')
#     pandas.DataFrame.from_records(all_coords).to_csv(config.dir('pointcloud_savedir') + sequence + '/' + 'all_coords.csv', index=False)
#     pandas.DataFrame.from_records(all_colors).to_csv(config.dir('pointcloud_savedir') + sequence + '/' + 'all_colors.csv', index=False)
#     print('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='path of config file to load', dest='config',
                        type=str, default='./configs/config_default')
    parser.add_argument('-compute_seg', help='compute segmentations', dest='compute_seg',
                        type=bool, default=True)
    parser.add_argument('-compute_flow_and_disp', help='compute flow and disp using netdef', dest='compute_flow_and_disp',
                        type=bool, default=False)
    parser.add_argument('-compute_poses', help='compute poses using ORB-SLAM', dest='compute_poses',
                        type=bool, default=False)
    # parser.add_argument('-compute_skymask', help='compute semsec for cutting sky using DeepLabv3', dest='compute_skymask',
    #                     type=bool, default=False)
    parser.add_argument('-compute_point_imgs', help='compute 3 channel point images using camera parameters',
                        dest='compute_point_imgs',
                        type=bool, default=False)
    # parser.add_argument('-compute_sceneflow', help='computation of scene flow', dest='compute_sceneflow',
    #                     type=bool, default=False)
    # parser.add_argument('-compute_pointclouds', help='computation of pointcloud', dest='compute_pointclouds',
    #                     type=bool, default=False)
    # parser.add_argument('-visualize_points', help='visualization of pointcloud', dest='visualize_points',
    #                     type=bool, default=False)
    args = parser.parse_args()
    config = Config(args.config)

    if args.compute_seg:
        refinement_net = refinement_net_init(config.str('segmentor_modelpath'))

    list_sequences, _ = load_seqmap(config.str('mots_seqmap_file'))

    for sequence in list_sequences:
        print(sequence)

        sequence_dir = config.dir('data_dir') + 'images/' + sequence + '/'

        calibration_params = CalibrationParameters(sequence_dir + sequence + '.txt')

        if args.compute_seg:
            print('computing segmentations..')
            detections = import_detections(config, sequence)
            compute_segmentations(refinement_net, sequence_dir, config.dir('segmentations_savedir') + sequence + '/', detections)
            print('done.')

        if args.compute_flow_and_disp:
            print('computing flow and disp..')
            compute_flow_and_disp(config.dir('flow_disp_savedir') + sequence + '/', sequence_dir, config.dir('netdef_disp_modeldir'),
                                  config.dir('netdef_flow_modeldir'))
            print('done.')

        if args.compute_poses:
            print('computing poses..')
            compute_poses_orb(sequence, calibration_params, config.dir('orb_pose_savedir') + sequence + '/', sequence_dir, config.dir('orbslam_modeldir'),
                              config.dir('orbslam_vocab_dir'))
            print('done.')

        # if args.compute_skymask:
        #     print('computing skymasks..')
        #     compute_sky_mask(config.dir('skymask_savedir') + sequence + '/', sequence_dir, config.dir('deeplab_modeldir'))
        #     print('done.')

        if args.compute_point_imgs:
            print('computing point images..')
            compute_point_imgs(config, sequence, calibration_params)
            print('done.')

        # if args.compute_sceneflow:
        #     print('computing scene flow..')
        #     compute_sceneflow(config, sequence)
        #     print('done.')
        #
        # if args.compute_pointclouds:
        #     print('computing pointclouds..')
        #     create_pointcloud(config, sequence)
        #     print('done.')
        #
        # if args.visualize_points:
        #     visualize_pointcloud(config.dir('pointcloud_savedir') + sequence + '/')
