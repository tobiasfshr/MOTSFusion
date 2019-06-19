from file_io.import_utils import import_detections, import_segmentations
from file_io.export_utils import export_tracking_result_in_mots_format
from eval.mots_eval.eval import run_mots_eval
from eval.mots_eval.mots_common.io import load_seqmap
from config import Config
from tracker.tracked_sequence import TrackedSequence
import pycocotools.mask as cocomask
from eval.mots_eval.mots_common.io import load_txt


def import_gt_file(gt_path):
    objects_per_frame = load_txt(gt_path)
    print(sequence)
    for frame in objects_per_frame.keys():
        for object in objects_per_frame.get(frame):
            if not object.track_id == 10000:
                track_id = (object.track_id % 1000)
                det = {'class': object.class_id}
                while tracks_gt.get_num_ids() <= track_id:
                    tracks_gt.add_empty_track()

                tracks_gt.add_to_track(frame, track_id, det, object.mask)


if __name__ == '__main__':
    config = Config('./configs/config_default')

    list_sequences, max_frames = load_seqmap(config.str('mots_seqmap_file'))
    for sequence in list_sequences:
        tracks_gt = TrackedSequence(max_frames[sequence]+1)
        import_gt_file('./data/mots_gt/' + sequence + '.txt')

        raw_detections = import_detections(config, sequence)
        segmentations = import_segmentations(config, sequence)

        tracks_gt_seg = TrackedSequence(max_frames[sequence]+1)

        while max_frames[sequence]+1 > len(raw_detections):
            raw_detections.append([])

        while max_frames[sequence]+1 > len(segmentations):
            segmentations.append([])

        for step in range(tracks_gt.timesteps):
            combined_mask_per_frame = {}
            for gt_id in tracks_gt.get_active_tracks(step):
                ref_mask = tracks_gt.get_mask(step, gt_id, decode=False)
                ref_det = tracks_gt.get_detection(step, gt_id)
                ref_class = ref_det['class']

                for mask, det in zip(segmentations[step], raw_detections[step]):
                    # mask based (MOTS)
                    mask_iou = cocomask.area(cocomask.merge([mask, ref_mask], intersect=True)) / cocomask.area(cocomask.merge([mask, ref_mask]))
                    if mask_iou > 0.5:
                        while tracks_gt_seg.get_num_ids() <= gt_id:
                            tracks_gt_seg.add_empty_track()

                        tracks_gt_seg.add_to_track(step, gt_id, det, mask)

                        if step not in combined_mask_per_frame:
                            combined_mask_per_frame[step] = mask
                        else:
                            combined_mask_per_frame[step] = cocomask.merge([combined_mask_per_frame[step], mask],
                                                                            intersect=False)
        tracks_gt_seg.fix_mask_overlap()
        export_tracking_result_in_mots_format(tracks_gt_seg, './scripts/gt_mots_eval/' + sequence + '/')

    run_mots_eval('./scripts/gt_mots_eval/', list_sequences, config.dir('mots_gt_folder'), config.str('mots_seqmap_file'))
