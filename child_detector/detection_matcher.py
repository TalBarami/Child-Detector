import numpy as np
from taltools.cv.bounding_boxes import xywh2xyxy, iou


class ChildMatcher:
    def __init__(self, iou_threshold, interpolation_threshold, tolerance):
        self.iou_threshold = iou_threshold
        self.interpolation_threshold = interpolation_threshold
        self.tolerance = tolerance

    def match(self, pboxes, detections):
        detections = detections.set_index('frame')
        T1, T2 = pboxes.shape[0], detections.index.max()+1
        if np.abs(T1-T2) > self.tolerance:
            raise IndexError(f'Length mismatch: skeleton({T1}) - detections({T2})')
        detections = detections.dropna()
        cboxes = detections[detections['label'] == 1]
        _aboxes = detections[detections['label'] == 0]
        cids = (np.ones(T1) * -1).astype(np.int8)
        for f, row in cboxes.iterrows():
            boxes = pboxes[f]
            cbox = row[['x1', 'y1', 'x2', 'y2']].values.squeeze()
            aboxes = _aboxes.loc[f:f][['x1', 'y1', 'x2', 'y2']].values if f in _aboxes.index else np.array([])

            best_idx, best_score = -1, 0
            for idx, pbox in enumerate(boxes):
                if np.all(pbox == 0):
                    continue
                iou_child = iou(cbox, pbox)
                iou_adult = max([iou(pbox, abox) for abox in aboxes]) if len(aboxes) > 0 else 0
                score = iou_child - iou_adult
                if score > best_score:
                    best_idx, best_score = idx, iou_child
            if best_idx != -1 and best_score > self.iou_threshold:
                cids[f] = best_idx
        return cids
