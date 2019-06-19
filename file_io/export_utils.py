import pickle
import os


def export_tracking_result(tracked_sequence, out_folder, full=False):
        if full:
            export_tracking_result_full(tracked_sequence, out_folder)

        export_tracking_result_in_mots_format(tracked_sequence, out_folder)
        export_tracking_result_in_mot_format(tracked_sequence, out_folder)


def export_tracking_result_full(tracked_sequence, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with open(out_folder + 'saved_data.pkl', "wb") as f:
        pickle.dump(tracked_sequence, f)


def export_tracking_result_in_mot_format(tracked_sequence, out_folder, class_names=['Car', 'Pedestrian']):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with open(out_folder + 'mot_result', "w") as f:
        for step in range(0, tracked_sequence.timesteps):
            for id in tracked_sequence.get_active_tracks(step):
                    detection = tracked_sequence.get_detection(step, id)
                    label = class_names[detection['class']-1]
                    print(step, id, label, -1, -1, -10, "{0:.2f}".format(detection['bbox'][0]), "{0:.2f}".format(detection['bbox'][1]),
                          "{0:.2f}".format(detection['bbox'][2]), "{0:.2f}".format(detection['bbox'][3]), -1, -1, -1, -1000, -1000, -1000, -10, file=f)


def export_tracking_result_in_mots_format(tracked_sequence, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with open(out_folder + 'mots_result', "w") as f:
        for step in range(0, tracked_sequence.timesteps):
            for id in tracked_sequence.get_active_tracks(step):
                detection = tracked_sequence.get_detection(step, id)
                mask = tracked_sequence.get_mask(step, id, decode=False)
                print(step, id, detection['class'], mask['size'][0], mask['size'][1], mask['counts'], file=f)

