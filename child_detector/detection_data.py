import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from taltools.cv.bounding_boxes import xywh2xyxy, iou

def compute_iou_matrix(boxes):
    """Compute the IoU matrix for a set of boxes."""
    n = len(boxes)
    iou_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            iou_matrix[i, j] = iou(boxes[i], boxes[j])
            iou_matrix[j, i] = iou_matrix[i, j]  # Symmetric
    return iou_matrix

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
        self._detections.to_csv(detections_path)

    @staticmethod
    def load(detections_path, confidence_threshold=0.6, duplication_threshold=0.03):
        return DetectionsData(pd.read_csv(detections_path).set_index('frame', drop=True), confidence_threshold, duplication_threshold)

    def remove_duplicates(self, df):
        def filter_duplicates(group):
            x_diff = np.abs(group['x'].values[:, None] - group['x'].values)
            y_diff = np.abs(group['y'].values[:, None] - group['y'].values)
            diag = group['diag'].values
            dynamic_threshold = diag[:, None] + diag[None, :]
            dynamic_threshold *= self.duplication_threshold
            is_duplicate = (x_diff < dynamic_threshold) & (y_diff < dynamic_threshold)
            np.fill_diagonal(is_duplicate, False)
            to_remove = set()
            for i in range(len(group)):
                if i in to_remove:
                    continue
                duplicates = np.where(is_duplicate[i])[0]
                if len(duplicates) > 0:
                    indices = [i] + duplicates.tolist()
                    indices = [group.index[idx] for idx in indices]
                    max_idx = group.loc[indices]['confidence'].idxmax()
                    to_remove.update(idx for idx in indices if idx != max_idx)
            return to_remove

        remove_indices = set()
        for _, group in df.groupby('frame'):
            remove_indices.update(filter_duplicates(group))
        return df.drop(remove_indices).reset_index(drop=True)

    def _process(self):
        detections = self._detections.copy().reset_index()
        detections[['x1', 'y1', 'x2', 'y2']] = (detections[['x', 'y', 'x', 'y']] + (detections[['w', 'h', 'w', 'h']].values / 2 * [-1, -1, 1, 1])).values
        detections['diag'] = np.sqrt(detections['w'] ** 2 + detections['h'] **2)
        detections['confidence'] = (detections['confidence_child'] - detections['confidence_adult'] + 1) / 2
        detections = self.remove_duplicates(detections)
        detections['label'] = 0
        valid = detections[detections['confidence'] >= self.confidence_threshold]
        max_indices = valid.groupby('frame')['confidence'].idxmax()
        detections.loc[max_indices, 'label'] = 1
        # frames = pd.DataFrame({'frame': np.arange(0, self._detections.index.max() + 1)})
        # df = pd.concat(dfs, ignore_index=True).sort_values(by='frame')
        # df = pd.merge(frames, df, on='frame', how='left').set_index('frame', drop=True)
        return detections.set_index('frame')
