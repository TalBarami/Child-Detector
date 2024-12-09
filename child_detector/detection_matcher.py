from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from tqdm import tqdm
from os import path as osp
from taltools.cv.bounding_boxes import xywh2xyxy, iou
from taltools.io.files import read_pkl

from child_detector.detection_data import DetectionsData
from child_detector.child_detector import ChildDetector

class ChildMatcher:
    def __init__(self, iou_threshold, interpolation_threshold, tolerance):
        self.iou_threshold = iou_threshold
        self.interpolation_threshold = interpolation_threshold
        self.tolerance = tolerance

    def match(self, pboxes, _detections):
        if type(_detections) is str:
            _detections = DetectionsData.load(_detections)
        detections = _detections.detections
        T1, T2 = pboxes.shape[0], detections.index.max()+1
        if np.abs(T1-T2) > self.tolerance:
            raise IndexError(f'Length mismatch: skeleton({T1}) - detections({T2})')
        cboxes = detections[detections['label'] == 1]
        _aboxes = detections[detections['label'] == 0]
        cids = np.ones(T1) * -1
        for f, row in cboxes.iterrows():
            boxes = xywh2xyxy(pboxes[f])
            cbox = xywh2xyxy(row[['x', 'y', 'w', 'h']].values).squeeze()
            aboxes = xywh2xyxy(_aboxes.loc[f][['x', 'y', 'w', 'h']].values) if f in _aboxes.index else np.array([])

            best_idx, best_score = -1, -1
            for idx, pbox in enumerate(boxes):
                iou_child = iou(cbox, pbox)
                iou_adult = max([iou(pbox, abox) for abox in aboxes]) if len(aboxes) > 0 else 0
                score = iou_child - iou_adult
                if score > best_score:
                    best_idx, best_score = idx, iou_child
            if best_idx != -1 and best_score > self.iou_threshold:
                cids[f] = best_idx
        return cids
