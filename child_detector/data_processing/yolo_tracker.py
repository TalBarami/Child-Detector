import os
import warnings

import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

class YOLOTracker:
    def __init__(self, model='yolov8n.pt'):
        self.model = YOLO(model, verbose=False)

    def track(self, video_path, reduce=True):
        results = self.model.track(video_path, verbose=False)
        if reduce:
            r = results[0]
            out = {
                'names': r.names,
                'orig_shape': r.orig_shape,
                'path': r.path
            }
            data = []
            for r in results:
                r = r.cpu()
                data.append({
                    'boxes': r.boxes,
                    'masks': r.masks,
                    'keypoints': r.keypoints,
                })
            out['data'] = data
            return out
        else:
            return results
