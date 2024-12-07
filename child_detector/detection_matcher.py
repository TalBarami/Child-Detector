from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from tqdm import tqdm
from os import path as osp
from taltools.cv.bounding_boxes import xywh2xyxy, iou
from taltools.io.files import read_pkl

from child_detector.detection_data import DetectionsData
from child_detector.child_detector import ChildDetector

class ChildMatcher(ABC):
    def __init__(self, confidence_threshold, iou_threshold, interpolation_threshold, tolerance):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.interpolation_threshold = interpolation_threshold
        self.tolerance = tolerance

    # def _interpolate(self, _cboxes):
    #     cboxes = _cboxes.copy()
    #     missing = cboxes['x'].isna()
    #     gap_starts = missing & ~missing.shift(1, fill_value=False)
    #     gap_ends = missing & ~missing.shift(-1, fill_value=False)
    #     gap_start_indices = cboxes.index[gap_starts].tolist()
    #     gap_end_indices = cboxes.index[gap_ends].tolist()
    #     interpolated = cboxes[['x', 'y', 'w', 'h', 'confidence']].interpolate(method='spline', order=2, limit_direction='both')
    #     for start, end in zip(gap_start_indices, gap_end_indices):
    #         gap_length = end - start + 1
    #         if gap_length <= self.interpolation_threshold:
    #             cboxes.loc[start:end, ['x', 'y', 'w', 'h', 'confidence']] = interpolated.loc[start:end, ['x', 'y', 'w', 'h', 'confidence']].interpolate(method='linear', limit_direction='both')
    #     return cboxes

    def _match(self, pboxes, detections):
        if type(detections) is str:
            detections = DetectionsData.load(detections)
        detections['class'] = (detections['confidence'] > self.confidence_threshold).astype(int)
        T1, T2 = pboxes.shape[0], detections['frame'].max()+1
        if np.abs(T1-T2) > self.tolerance:
            raise IndexError(f'Length mismatch: skeleton({T1}) - detections({T2})')
        cboxes = detections[detections['class'] == 1]
        cids = cboxes.groupby('frame')['confidence'].idxmax()
        cboxes = cboxes.loc[cids].set_index('frame')
        # cboxes = self._interpolate(_cboxes)
        _aboxes = detections[detections['class'] == 0].set_index('frame')
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

    @abstractmethod
    def match(self, data, detections):
        pass



class SkeletonMatcher(ChildMatcher):
    def __init__(self, confidence_threshold, iou_threshold, interpolation_threshold, tolerance):
        super().__init__(confidence_threshold, iou_threshold, interpolation_threshold, tolerance)

    def match(self, skeleton, detections):
        if type(skeleton) is str:
            skeleton = SkeletonData.load(skeleton)
        sboxes = skeleton.bounding_boxes()
        cids = self._match(sboxes, detections)

class FacialMatcher(ChildMatcher):
    def __init__(self, confidence_threshold, iou_threshold, interpolation_threshold, tolerance):
        super().__init__(confidence_threshold, iou_threshold, interpolation_threshold, tolerance)

    def match(self, facial_data, detections):
        if type(facial_data) is str:
            facial_data = pd.read_csv(facial_data)
        if type(detections) is str:
            detections = read_pkl(detections)
        # TODO: match facial_data with children
        return facial_data
