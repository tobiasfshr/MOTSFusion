from config import Config
from file_io.import_utils import import_detections, import_segmentations, import_flow, import_poses, import_raw_imgs, import_point_imgs, import_tracking_result
from file_io.export_utils import export_tracking_result
from tracker.segmentation_tracking import create_tracklets
from eval.mots_eval.eval import run_mots_eval
from eval.mot_eval.evaluate_tracking import run_mot_eval
from eval.mots_eval.mots_common.io import load_seqmap
from visualization.visualize_mots import visualize_sequences
from utils.calibration_params import CalibrationParameters
from tracker.create_dynamic_transforms import create_dynamic_transforms
from tracker.reconstruction_tracking import merge_tracks, extrapolate_final_tracks
from external.BB2SegNet.segment import refinement_net_init
from visualization.visualize import visualize_sequence_3D
import argparse
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='path of config file to load',
                        dest='config',
                        type=str, default='./configs/config_default')
    args = parser.parse_args()

    config = Config(args.config)

    list_sequences, _ = load_seqmap(config.str('mots_seqmap_file'))
    refinement_net = refinement_net_init(config.str('segmentor_modelpath'))
    time_2d = 0
    time_dyn = 0
    time_merge = 0
    time_exp = 0
    for sequence in list_sequences:
        print('Perform tracking for sequence', sequence)
        # load pre-computed detections, optical flow and segmentations
        raw_detections = import_detections(config, sequence)
        segmentations = import_segmentations(config, sequence)
        flow = import_flow(config, sequence)

        # do 2D based tracking
        tic = time.time()
        tracked_sequence = create_tracklets(config, raw_detections, segmentations, flow)
        toc = time.time()
        print('  time elapsed', toc - tic)
        time_2d += toc - tic

        # write 2D result out to file
        tracked_sequence.fix_invalid_tracks()
        if config.str('mode') == 'MOTS':
            tracked_sequence.fix_mask_overlap()

        tracked_sequence.reorder_ids()
        export_tracking_result(tracked_sequence, config.dir('2D_tracking_result_savedir') + sequence + '/')

        print('  2D tracking done.')

        # load pre-computed poses, pointcloud images
        poses = import_poses(config, sequence)
        point_imgs = import_point_imgs(config, sequence)
        raw_imgs = import_raw_imgs(config, sequence)

        # load calibration parameters
        calibration_params = CalibrationParameters(config.dir('data_dir') + 'images/' + sequence + '/' + sequence + '.txt')

        # do 3D reconstruction based tracking
        # create dynamic transforms
        print('  Create dynamic transforms...')
        tic = time.time()
        tracked_sequence = create_dynamic_transforms(config, tracked_sequence, flow, point_imgs, raw_imgs, calibration_params)
        toc = time.time()
        print('  time elapsed', toc - tic)
        time_dyn += toc - tic
        print('  done.')

        # merge tracks
        print('  Merge tracks...')
        tic = time.time()
        tracked_sequence = merge_tracks(config, tracked_sequence, point_imgs, raw_imgs, poses, flow, calibration_params, refinement_net)
        toc = time.time()
        print('  time elapsed', toc - tic)
        time_merge += toc - tic
        print('  done.')

        # extrapolate final tracks
        print('  Extrapolate tracks...')
        tic = time.time()
        tracked_sequence = extrapolate_final_tracks(config, tracked_sequence, flow, poses, point_imgs, calibration_params, raw_imgs, refinement_net)
        toc = time.time()
        print('  time elapsed', toc - tic)
        time_exp += toc - tic
        print('  done.')

        # write 3D result out to file
        if config.str('mode') == 'MOTS':
            tracked_sequence.fix_mask_overlap()
        tracked_sequence.reorder_ids()
        export_tracking_result(tracked_sequence, config.dir('3D_tracking_result_savedir') + sequence + '/')
        print('  3D tracking done.')

        # visualize 3D tracking result
        if config.bool('debug'):
            print('  Visualize sequence reconstruction in Point Cloud...')
            visualize_sequence_3D(config, tracked_sequence, point_imgs, raw_imgs)
            print('  done.')

    print('time_2d', time_2d)
    print('time_dyn', time_dyn)
    print('time_merge', time_merge)
    print('time_exp', time_exp)

    # evaluate
    if not 'test' in args.config:
        print('Evaluating results...')
        print('2D tracking:')
        if config.str('mode') == 'MOTS':
            run_mots_eval(config.dir('2D_tracking_result_savedir'), list_sequences, config.dir('mots_gt_folder'), config.str('mots_seqmap_file'))
        else:
            run_mot_eval(config.dir('2D_tracking_result_savedir'), list_sequences, eval_modified=False)
            run_mot_eval(config.dir('2D_tracking_result_savedir'), list_sequences, eval_modified=True)
        print('')
        print('3D tracking:')
        if config.str('mode') == 'MOTS':
            run_mots_eval(config.dir('3D_tracking_result_savedir'), list_sequences, config.dir('mots_gt_folder'), config.str('mots_seqmap_file'))
        else:
            run_mot_eval(config.dir('3D_tracking_result_savedir'), list_sequences, eval_modified=False)
            run_mot_eval(config.dir('3D_tracking_result_savedir'), list_sequences, eval_modified=True)

    #visualize
    if config.str('mode') == 'MOTS':
        print('Visualizing results (3D)...')
        visualize_sequences(list_sequences, config.dir('3D_tracking_result_savedir'), config.dir('data_dir') + 'images/',
                            config.dir('3d_mots_vis_output_folder'), config.str('mots_seqmap_file'), draw_boxes=False,
                            create_video=False)

        print('Visualizing results (2D)...')
        visualize_sequences(list_sequences, config.dir('2D_tracking_result_savedir'), config.dir('data_dir') + 'images/',
                            config.dir('2d_mots_vis_output_folder'), config.str('mots_seqmap_file'), draw_boxes=False,
                            create_video=False)
