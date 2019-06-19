import numpy as np


class CalibrationParameters:

    def __init__(self, filepath):
        calib_params = open(filepath, 'r').readlines()
        self.cam_projection_matrix_l = self.parseLine(calib_params[2], (3, 4))
        self.cam_projection_matrix_r = self.parseLine(calib_params[3], (3, 4))

        self.baseline = 0.54
        self.fx = self.cam_projection_matrix_l[0, 0]
        self.fy = self.cam_projection_matrix_l[1, 1]
        self.cx = self.cam_projection_matrix_l[0, 2]
        self.cy = self.cam_projection_matrix_l[1, 2]
        self.bf =  self.baseline * self.fx

        self.camera_matrix = np.asarray([self.fx, 0., self.cx, 0., self.fy, self.cy, 0., 0., 1.]).reshape((3, 3))

    def parseLine(self, line, shape):
        data = line.split()
        data = np.array(data[1:]).reshape(shape).astype(np.float32)
        return data
