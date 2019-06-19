from file_io.io_utils import readFlow, readFloat
import json
import os
import numpy as np
import skimage.io
from utils.geometry_utils import process_poses
from tracker.tracked_sequence import TrackedSequence
import pickle
import glob


def import_point_imgs(config, sequence):
    return [readFloat(config.dir('point_imgs_savedir') + sequence + '/' + file)
            for file in sorted(os.listdir(config.dir('point_imgs_savedir') + sequence + '/'))]


def import_raw_imgs(config, sequence):
    return [skimage.io.imread(config.dir('data_dir') + 'images/' + sequence + '/image_2/' + file)
            for file in sorted(os.listdir(config.dir('data_dir') + 'images/' + sequence + '/image_2/'))]


def import_flow(config, sequence):
    if config.str('use_flow') == 'FlowNet':
        return import_flow_FlowNet(config.dir('flow_savedir'), sequence)
    elif config.str('use_flow') == 'PWCNet':
        return import_flow_PWCNet(config.dir('flow_savedir'), sequence)
    else:
        return None


def postprocess_flow(flow, occlusion):
    for i in range(len(flow)):
        if flow[i] is not None:
            occ_mask = np.asarray(occlusion[i], dtype=np.bool)
            occ_mask = np.concatenate((occ_mask, occ_mask), axis=-1)
            flow[i][occ_mask] = 0
    return flow


def import_flow_FlowNet(flow_import_path, sequence):
    flow_list = sorted(filter(lambda x: 'flow[0].fwd' in x, os.listdir(flow_import_path + sequence + '/')))
    optical_flow = [readFlow(flow_import_path + sequence + '/' + flow) for flow in flow_list]
    optical_flow = [None] + optical_flow
    occ_list = sorted(filter(lambda x: 'occ[0].fwd' in x, os.listdir(flow_import_path + sequence + '/')))
    occlusion = [readFloat(flow_import_path + sequence + '/' + occ) for occ in occ_list]
    occlusion = [None] + occlusion
    return postprocess_flow(optical_flow, occlusion)


def open_flow_png_file(file_path_list):
    # Funtion from Kilian Merkelbach.
    # Decode the information stored in the filename
    flow_png_info = {}
    assert len(file_path_list) == 2
    for file_path in file_path_list:
        file_token_list = os.path.splitext(file_path)[0].split("_")
        minimal_value = int(file_token_list[-1].replace("minimal", ""))
        flow_axis = file_token_list[-2]
        flow_png_info[flow_axis] = {'path': file_path,
                                    'minimal_value': minimal_value}

    # Open both files and add back the minimal value
    for axis, flow_info in flow_png_info.items():
        import png
        png_reader = png.Reader(filename=flow_info['path'])
        flow_2d = np.vstack(map(np.uint16, png_reader.asDirect()[2]))

        # Add the minimal value back
        flow_2d = flow_2d.astype(np.int16) + flow_info['minimal_value']

        flow_png_info[axis]['flow'] = flow_2d

    # Combine the flows
    flow_x = flow_png_info['x']['flow']
    flow_y = flow_png_info['y']['flow']
    flow = np.stack([flow_x, flow_y], 2)

    return flow


def import_flow_PWCNet(flow_import_path, sequence):
    if os.path.exists(flow_import_path + "/preprocessed_" + sequence):
        with open(flow_import_path + "/preprocessed_" + sequence, 'rb') as input:
            flows = pickle.load(input)
    else:
        flow_files_x = sorted(glob.glob(flow_import_path + "/" + sequence + "/*_x_minimal*.png"))
        flow_files_y = sorted(glob.glob(flow_import_path + "/" + sequence + "/*_y_minimal*.png"))
        assert len(flow_files_x) == len(flow_files_y)
        flows = [open_flow_png_file([x, y]) for x, y in zip(flow_files_x, flow_files_y)]
        with open(flow_import_path + "/preprocessed_" + sequence, 'wb') as output:
            pickle.dump(flows, output, pickle.HIGHEST_PROTOCOL)
    flows = [None] + flows
    return flows


def import_disparity(config, sequence):
    if config.str('use_depth') == 'DispNet':
        return import_disp_DispNet(config.dir('disp_savedir'), sequence)
    else:
        return None


def import_disp_DispNet(depth_import_path, sequence):
    disp_list = sorted(filter(lambda x: 'disp' in x, os.listdir(depth_import_path + sequence + '/')))
    disps = [readFloat(depth_import_path + sequence + '/' + disp) for disp in disp_list]
    return disps


def import_poses(config, sequence):
    if config.str('use_pose') == 'orbslam':
        return import_pose_orbslam(config.dir('orb_pose_savedir'), sequence)
    else:
        return None


def import_pose_orbslam(pose_import_path, sequence):
    pos_list = np.genfromtxt(pose_import_path + sequence + '/CameraTrajectory.txt', dtype='str')
    return process_poses(pos_list, 'orbslam')


def import_segmentations(config, sequence):
    if config.str('use_segmentations') == 'BB2SegNet':
        return import_segmentations_BB2SegNet(config, sequence)
    elif config.str('use_segmentations') == 'TrackRCNN':
        return import_segmentations_TrackRCNN(config, sequence)
    else:
        assert False, "Select appropriate segmentations"


def import_segmentations_BB2SegNet(config, sequence):
    # import pycocotools.mask as cm
    # import matplotlib.pyplot as plt
    # segs = json.load(open(config.dir('segmentations_savedir') + sequence + '/segmentations.json', 'r'))
    #
    # for segs_t in segs:
    #     for seg in segs_t:
    #         plt.imshow(cm.decode(seg))
    #         plt.show()

    return json.load(open(config.dir('segmentations_savedir') + sequence + '/segmentations.json', 'r'))


def import_segmentations_TrackRCNN(config, sequence):
    segmentations = []

    try:
        with open(config.dir('segmentations_savedir') + sequence + '.txt') as f:
            lines = f.readlines()
    except:
        print('WARNING: could not load ' + config.dir('segmentations_savedir') + sequence + '.txt')
        return None

    for line in lines:
        line = line.split(' ')
        segmentation = {}
        segmentation['counts'] = line[9]
        segmentation['score'] = line[5]
        segmentation['size'] = [int(line[7]), int(line[8])]

        t = int(line[0])
        while t >= len(segmentations):
            segmentations.append([])

        segmentations[t].append(segmentation)

    return segmentations


def import_detections(config, sequence):
    if config.str('use_detections') == 'RRC':
        return import_detections_RRC(config.dir('detections_savedir') + sequence + '/')
    if config.str('use_detections') == 'TrackRCNN':
        return import_detections_TrackRCNN(config.dir('detections_savedir') + sequence + '.txt')
    else:
        assert False, "Select appropriate detections"


def import_detections_RRC(detections_import_path):
    detections = []

    for t in range(len(os.listdir(detections_import_path))):
        file = detections_import_path + str(t).zfill(6) + '.txt'
        try:
            with open(file) as f:
                lines = f.readlines()
        except:
            detections.append([])
            print('WARNING: could not load ' + file)
            continue

        curr_detections = []
        for line in lines:
            line = line.split(' ')
            detection = {}
            detection['bbox'] = [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
            detection['score'] = float(line[4])
            detection['class'] = 1  # RRC works only for cars

            # import PIL.Image as Image
            # import numpy as np
            # import matplotlib.pyplot as plt
            # img = np.array(Image.open('/home/fischer_t/PycharmProjects/mots-dynslam/data/KITTI_tracking/unit_test/images/0002/image_2/' + os.path.split(file)[1].replace('.txt', '.png')), dtype="float32") / 255
            # fig = plt.figure()
            # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
            # ax = fig.subplots()
            # ax.set_axis_off()
            # ax.imshow(img)
            #
            # import matplotlib.patches as patches
            # rect = patches.Rectangle((float(line[0]), float(line[1])), float(line[2]) - float(line[0]), float(line[3]) - float(line[1]), linewidth=2, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)
            #
            # plt.show()

            curr_detections.append(detection)

        detections.append(curr_detections)

    return detections


def import_detections_TrackRCNN(detections_import_file):
    detections = []

    try:
        with open(detections_import_file) as f:
            lines = f.readlines()
    except:
        print('WARNING: could not load ' + detections_import_file)
        return None

    for line in lines:
        line = line.split(' ')
        detection = {}
        detection['bbox'] = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
        detection['score'] = float(line[5])
        detection['class'] = int(line[6])

        t = int(line[0])
        while t >= len(detections):
            detections.append([])

        detections[t].append(detection)

    return detections


def import_tracking_result(result_path):
    with open(result_path + 'saved_data.pkl', "rb") as f:
        tracked_sequence = pickle.load(f)
    return tracked_sequence



