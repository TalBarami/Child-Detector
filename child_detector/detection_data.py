import numpy as np
import pandas as pd
from taltools.cv.bounding_boxes import xywh2xyxy, iou

class DetectionsData:
    def __init__(self, detections, confidence_threshold=0.6, duplication_threshold=0.9):
        self._detections = detections
        self._detections_processed = None
        self.confidence_threshold = confidence_threshold
        self.duplication_threshold = duplication_threshold

    @property
    def detections(self):
        if self._detections_processed is None:
            self._detections_processed = self._process()
        return self._detections_processed

    @property
    def child(self):
        return self.detections[self.detections['label'] == 1]

    @property
    def adults(self):
        return self.detections[self.detections['label'] == 0]

    def save(self, detections_path):
        self._detections.to_csv(detections_path, index=False)

    @staticmethod
    def load(detections_path, confidence_threshold=0.6, duplication_threshold=0.9):
        return DetectionsData(pd.read_csv(detections_path).set_index('frame', drop=True), confidence_threshold, duplication_threshold)

    def _process(self):
        dfs = []
        frames = pd.DataFrame({'frame': np.arange(0, self._detections.index.max() + 1)})
        for frame, df in self._detections.groupby('frame'):
            df = df.reset_index(drop=True)
            df['confidence'] = (df['confidence_child'] - df['confidence_adult'] + 1) / 2
            df['label'] = 0
            max_conf, max_idx = df['confidence'].max(), df['confidence'].idxmax()
            df.loc[max_idx, 'label'] = 1 if max_conf >= self.confidence_threshold else 0
            to_remove = set()
            if len(df) > 1:
                boxes = xywh2xyxy(df[['x', 'y', 'w', 'h']].values)
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        boxes_iou = iou(boxes[i], boxes[j])
                        if boxes_iou > self.duplication_threshold:
                            to_remove.add(j if df['confidence'].iloc[i] > df['confidence'].iloc[j] else i)
            df = df.drop(list(to_remove)).reset_index(drop=True)
            df['frame'] = frame
            dfs.append(df)
        df = pd.concat(dfs).sort_values(by='frame')
        df = pd.merge(frames, df, on='frame', how='left').set_index('frame', drop=True)
        return df