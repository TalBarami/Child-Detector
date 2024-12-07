import numpy as np
import pandas as pd
from taltools.cv.bounding_boxes import xywh2xyxy, iou

class DetectionsData:
    def __init__(self, detections, duplication_threshold=0.9):
        self._detections = detections
        self._detections_processed = None
        self.duplication_threshold = duplication_threshold

    @property
    def detections(self):
        if self._detections_processed is None:
            self._detections_processed = self._process()
        return self._detections_processed

    def save(self, detections_path):
        self._detections.to_csv(detections_path, index=False)

    @staticmethod
    def load(detections_path, duplication_threshold=0.9):
        return DetectionsData(pd.read_csv(detections_path), duplication_threshold)

    def _process(self):
        dfs = []
        frames = pd.DataFrame({'frame': np.arange(0, self._detections['frame'].max() + 1)})
        for _, df in self._detections.groupby('frame'):
            df = df.reset_index(drop=True)
            df['confidence'] = (df['confidence_child'] - df['confidence_adult'] + 1) / 2
            to_remove = set()
            if len(df) > 1:
                boxes = xywh2xyxy(df[['x', 'y', 'w', 'h']].values)
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        boxes_iou = iou(boxes[i], boxes[j])
                        if boxes_iou > self.duplication_threshold:
                            to_remove.add(j if df['confidence'].iloc[i] > df['confidence'].iloc[j] else i)
            df = df.drop(list(to_remove)).reset_index(drop=True)
            dfs.append(df)
        df = pd.concat(dfs).sort_values(by='frame')
        df = pd.merge(frames, df, on='frame', how='left').reset_index(drop=True)
        return df