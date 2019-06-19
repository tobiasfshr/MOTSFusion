from subprocess import call
import pickle
import numpy as np
import os


def compute_poses_orb(sequence, calib_params, save_path, data_path, model_path, vocab_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    template_file = open(model_path + 'KITTI.yaml', 'r')

    calibration_file = template_file.readlines()
    for i in range(len(calibration_file)):
        calibration_file[i] = calibration_file[i].replace('$fx$', str(calib_params.fx))
        calibration_file[i] = calibration_file[i].replace('$fy$', str(calib_params.fy))
        calibration_file[i] = calibration_file[i].replace('$cx$', str(calib_params.cx))
        calibration_file[i] = calibration_file[i].replace('$cy$', str(calib_params.cy))
        calibration_file[i] = calibration_file[i].replace('$bf$', str(calib_params.bf))

    calibration_path = data_path + sequence + '.yaml'

    open(calibration_path, 'w').writelines(calibration_file)

    call([model_path + 'stereo_kitti', vocab_path, calibration_path, data_path[:-1], save_path])

