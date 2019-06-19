import pycocotools.mask as cocomask
import numpy as np
import cv2


def cut(mask, ref_mask):
    mask = cocomask.decode(mask)
    ref_mask = cocomask.decode(ref_mask)
    mask = np.where(mask == ref_mask, 0, mask)
    mask = cocomask.encode(np.asfortranarray(mask))
    mask['counts'] = mask['counts'].decode("UTF-8")
    return mask


class TrackedSequence:
    def __init__(self, timesteps):
        self.timesteps = timesteps

        self.track_ids = [[] for _ in range(timesteps)]
        self.detections = [[] for _ in range(timesteps)]
        self.masks = [[] for _ in range(timesteps)]

        self.attributes = {}

    def get_active_tracks(self, timestep):
        ids = []
        for id, is_tracked in enumerate(self.track_ids[timestep]):
            if is_tracked:
                ids.append(id)

        return ids

    def get_track(self, id):
        return list(map(lambda x: x[id], self.track_ids))

    def get_track_masks(self, id):
        return list(map(lambda x: x[id], self.masks))

    def get_track_detections(self, id):
        return list(map(lambda x: x[id], self.detections))

    def get_track_attribute(self, id, attr_name):
        return list(map(lambda x: x[id], self.attributes[attr_name]))

    def is_active(self, t, id):
        return self.track_ids[t][id]

    def get_mask(self, t, id, decode=True, postprocess=False):
        if decode:
            if postprocess:
                mask = cocomask.decode(self.masks[t][id])
                return cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
            else:
                return cocomask.decode(self.masks[t][id])
        else:
            if postprocess:
                mask = cocomask.decode(self.masks[t][id])
                return cocomask.encode(cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1))
            else:
                return self.masks[t][id]

    def get_detection(self, t, id):
        return self.detections[t][id]

    def get_attribute(self, t, id, attr_name):
        return self.attributes[attr_name][t][id]

    def get_num_ids(self):
        return len(self.track_ids[0])

    def set_attribute(self, t, id, attr_name, value):
        self.attributes[attr_name][t][id] = value

    def merge_tracks(self, ref_id, merge_id):
        for t in range(self.timesteps):
            if self.is_active(t, merge_id):
                self.track_ids[t][ref_id] = True
                self.track_ids[t][merge_id] = False

                self.detections[t][ref_id] = self.detections[t][merge_id]
                self.detections[t][merge_id] = None

                self.masks[t][ref_id] = self.masks[t][merge_id]
                self.masks[t][merge_id] = None

                for name in self.attributes.keys():
                    self.attributes[name][t][ref_id] = self.attributes[name][t][merge_id]
                    self.attributes[name][t][merge_id] = None

    def remove_track(self, id):
        for t in range(self.timesteps):
            if self.track_ids[t][id]:
                self.track_ids[t][id] = False
                self.detections[t][id] = None
                self.masks[t][id] = None

                for name in self.attributes.keys():
                    self.attributes[name][t][id] = None

    def remove_from_track(self, t, id):
        self.track_ids[t][id] = False
        self.detections[t][id] = None
        self.masks[t][id] = None

        for name in self.attributes.keys():
            self.attributes[name][t][id] = None

    def add_empty_track(self):
        for i in range(self.timesteps):
            self.track_ids[i].append(False)
            self.detections[i].append(None)
            self.masks[i].append(None)

    def start_new_track(self, t, detection, mask):
        for i in range(self.timesteps):
            if i != t:
                self.track_ids[i].append(False)
                self.detections[i].append(None)
                self.masks[i].append(None)

        self.track_ids[t].append(True)
        self.detections[t].append(detection)
        self.masks[t].append(mask)

        return len(self.track_ids[t])-1

    def add_new_attribute(self, name, values=None):
        if values is None:
            self.attributes[name] = [[None for _ in range(self.get_num_ids())] for _ in range(self.timesteps)]
        else:
            self.attributes[name] = [[values[t][id] for id in range(self.get_num_ids())] for t in range(self.timesteps)]

    def add_to_attribute(self, name, t, id, object):
        self.attributes[name][t][id] = object

    def add_to_track(self, t, id, detection, mask):
        self.track_ids[t][id] = True
        self.detections[t][id] = detection
        self.masks[t][id] = mask

    def reorder_ids(self):  # only call when save to change IDs!
        for id in reversed(range(self.get_num_ids())):
            current_track = self.get_track(id)

            if not any(current_track):
                for t in range(self.timesteps):
                    del self.track_ids[t][id]
                    del self.detections[t][id]
                    del self.masks[t][id]

                    for name in self.attributes.keys():
                        del self.attributes[name][t][id]

    def fix_invalid_tracks(self, sum_thresh=2.5, avg_thresh=0):
        for curr_id in range(self.get_num_ids()):
            if any(self.get_track(curr_id)):
                count = 0
                sum_box_score = 0
                for timestep, box in enumerate(self.get_track_detections(curr_id)):
                    if box is not None:
                        sum_box_score += self.get_detection(timestep, curr_id)['score']
                        count += 1

                if count:
                    avg_box_score = sum_box_score / count
                else:
                    avg_box_score = sum_box_score

                if sum_box_score < sum_thresh or (avg_box_score < avg_thresh):
                    self.remove_track(curr_id)

    def fix_mask_overlap(self, thresh=0.3):
        for t in range(self.timesteps):
            for id in range(self.get_num_ids()):
                if self.masks[t][id] is not None:
                    ref_mask = self.masks[t][id]

                    for k in range(id+1, self.get_num_ids()):
                        if self.masks[t][k] is not None:
                            overlap = cocomask.area(cocomask.merge([self.masks[t][k], ref_mask], intersect=True))
                            if overlap > 0.0:
                                if overlap > thresh * np.minimum(cocomask.area(ref_mask), cocomask.area(self.masks[t][k])):
                                    if cocomask.area(ref_mask) < cocomask.area(self.masks[t][k]):
                                        ref_mask = None
                                        break
                                    else:
                                        self.remove_from_track(t, k)

                                else:
                                    if cocomask.area(ref_mask) < cocomask.area(self.masks[t][k]):
                                        self.masks[t][k] = cut(self.masks[t][k], ref_mask)
                                    else:
                                        ref_mask = cut(ref_mask, self.masks[t][k])

                    self.masks[t][id] = ref_mask

                    if self.masks[t][id] is None:
                        self.remove_from_track(t, id)
