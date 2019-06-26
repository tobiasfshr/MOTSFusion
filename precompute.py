from external.ORB_SLAM2.compute_poses import compute_poses_orb
from external.netdef_models.compute_disp_flow import compute_flow_and_disp
from utils.calibration_params import CalibrationParameters
from file_io.io_utils import readFloat, readFlow, writeFloat
from utils.geometry_utils import flowwarp_pointimg, computePointImage, transformPointImage, process_poses, filter_nans, computeColorMatrix
from config import Config
from file_io.import_utils import import_detections
from external.BB2SegNet.segment import compute_segmentations, refinement_net_init
from eval.mots_eval.mots_common.io import load_seqmap

import argparse
import pickle
import os
import numpy as np


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='path of config file to load', dest='config',
                        type=str, default='./configs/config_default')
    parser.add_argument('-compute_seg', help='compute segmentations', dest='compute_seg',
                        type=bool, default=True)
    parser.add_argument('-compute_flow_and_disp', help='compute flow and disp using netdef', dest='compute_flow_and_disp',
                        type=bool, default=True)
    parser.add_argument('-compute_poses', help='compute poses using ORB-SLAM', dest='compute_poses',
                        type=bool, default=True)
    parser.add_argument('-compute_point_imgs', help='compute 3 channel point images using camera parameters',
                        dest='compute_point_imgs',
                        type=bool, default=True)
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

        if args.compute_point_imgs:
            print('computing point images..')
            compute_point_imgs(config, sequence, calibration_params)
            print('done.')
