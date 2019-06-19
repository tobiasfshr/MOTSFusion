import pycocotools.mask as rletools
import sys
from eval.mots_eval.mots_common.io import load_seqmap, load_sequences, load_gt_sequences
from eval.mots_eval.MOTS_metrics import compute_MOTS_metrics


IGNORE_CLASS = 10


def mask_iou(a, b, criterion="union"):
    is_crowd = criterion != "union"
    return rletools.iou([a.mask], [b.mask], [is_crowd])[0][0]


def evaluate_class(gt, results, max_frames, class_id):
    _, results_obj = compute_MOTS_metrics(gt, results, max_frames, class_id, IGNORE_CLASS, mask_iou)
    return results_obj


def run_mots_eval(result_dir, sequences, gt_folder, seqmap_filename):
    seqmap, max_frames = load_seqmap(seqmap_filename)
    print("Loading ground truth...")
    gt = load_gt_sequences(gt_folder, sequences)
    print("Loading results...")
    results = load_sequences(result_dir, sequences)
    print("Compute KITTI tracking eval with simplified matching and MOTSA")
    print("Evaluate class: Cars")
    results_cars = evaluate_class(gt, results, max_frames, 1)
    print("Evaluate class: Pedestrians")
    results_ped = evaluate_class(gt, results, max_frames, 2)
    print("Results for table (no *)")
    print("%.1f" % (results_cars.sMOTSA * 100.0), "%.1f" % (results_ped.sMOTSA * 100.0),
          "%.1f" % (results_cars.MOTSA * 100.0), "%.1f" % (results_ped.MOTSA * 100.0),
          "%.1f" % (results_cars.MOTSP * 100.0), "%.1f" % (results_ped.MOTSP * 100.0), results_cars.id_switches,
          results_ped.id_switches, sep="\t")
    print("Results for table (*)")
    print("%.1f" % (results_cars.sMOTSA_all_ids * 100.0), "%.1f" % (results_ped.sMOTSA_all_ids * 100.0),
          "%.1f" % (results_cars.MOTSA_all_ids * 100.0), "%.1f" % (results_ped.MOTSA_all_ids * 100.0),
          "%.1f" % (results_cars.MOTSP * 100.0), "%.1f" % (results_ped.MOTSP * 100.0), results_cars.id_switches_all,
          results_ped.id_switches_all, sep="\t")

