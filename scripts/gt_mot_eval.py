from file_io.import_utils import import_detections, import_segmentations
from file_io.export_utils import export_tracking_result_in_mot_format
from eval.mot_eval.evaluate_tracking import run_mot_eval
from eval.mots_eval.mots_common.io import load_seqmap
from config import Config
from tracker.tracked_sequence import TrackedSequence
from utils.geometry_utils import bbox_iou
from eval.mots_eval.mots_common.io import SegmentedObject

def load_txt(path):
    objects_per_frame = {}
    track_ids_per_frame = {}  # To check that no frame contains two objects with same id
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(" ")

            if int(fields[1]) == -1:
                continue

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            if fields[2] == 'Car':
                class_id = 1
            elif fields[2] == 'Pedestrian':
                class_id = 2

            if not (class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            mask = {'bbox': [float(fields[6]), float(fields[7]), float(fields[8]), float(fields[9])], 'class': class_id}

            objects_per_frame[frame].append(SegmentedObject(
                mask,
                class_id,
                int(fields[1])
            ))

    return objects_per_frame


def import_gt_file(gt_path):
    objects_per_frame = load_txt(gt_path)
    print(sequence)
    for frame in objects_per_frame.keys():
        for object in objects_per_frame.get(frame):
            if not object.track_id == 10000:
                while tracks_gt.get_num_ids() <= object.track_id:
                    tracks_gt.add_empty_track()

                tracks_gt.add_to_track(frame, object.track_id, object.mask, None)


if __name__ == '__main__':
    config = Config('./configs/config_default')

    list_sequences, max_frames = load_seqmap(config.str('mots_seqmap_file'))
    for sequence in list_sequences:
        tracks_gt = TrackedSequence(max_frames[sequence]+1)
        import_gt_file('./eval/mot_eval/data/tracking/label_02/' + sequence + '/' + sequence + '.txt')

        raw_detections = import_detections(config, sequence)

        tracks_gt_det = TrackedSequence(max_frames[sequence]+1)

        while max_frames[sequence]+1 > len(raw_detections):
            raw_detections.append([])

        for step in range(tracks_gt.timesteps):
            for gt_id in tracks_gt.get_active_tracks(step):
                ref_det = tracks_gt.get_detection(step, gt_id)
                ref_bbox = ref_det['bbox']
                ref_class = ref_det['class']

                for det in raw_detections[step]:
                    # bbox based (MOT)
                    box_iou = bbox_iou(ref_bbox, det['bbox'])
                    if box_iou >= 0.5:
                        while tracks_gt_det.get_num_ids() <= gt_id:
                            tracks_gt_det.add_empty_track()

                        tracks_gt_det.add_to_track(step, gt_id, det, None)

                    else:
                        print(det['score'])

        export_tracking_result_in_mot_format(tracks_gt_det, './scripts/gt_mot_eval/' + sequence + '/')

    run_mot_eval('./scripts/gt_mot_eval/', list_sequences, eval_modified=False)
    run_mot_eval('./scripts/gt_mot_eval/', list_sequences, eval_modified=True)
