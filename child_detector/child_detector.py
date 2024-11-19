import logging
import os
from collections import Counter
from os import path as osp
from pathlib import Path

import numpy as np
import torch
from sympy.codegen.cnodes import static
from torch.utils.data import DataLoader
from ultralytics import YOLO
import pandas as pd
from taltools.cv.bounding_boxes import xywh2xyxy, iou
from taltools.cv.iterable_video_dataset import IterableVideoDataset
from taltools.io.files import read_pkl, write_pkl

from child_detector.confidence_overrider import override_conf


MODEL_PATH = osp.join(Path(__file__).parent.parent, 'resources', 'models', 'child_detector.pt')

class ChildDetector:
    def __init__(self, model_path=MODEL_PATH, confidence_threshold=0.25, duplication_threshold=0.9, batch_size=128, device=None):
        override_conf()
        handlers = list(logging.getLogger().handlers)
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)
        # self.confidence_threshold = confidence_threshold
        self.duplication_threshold = duplication_threshold
        self.batch_size = batch_size
        logging.getLogger().handlers = handlers

    def _detect(self, video_path):
        dataset = IterableVideoDataset(video_path)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda x: x)
        out = []
        for frames_batch in dataloader:
            detections = self.model(frames_batch)
            out += [pd.DataFrame(data=torch.cat((x.boxes.data[:, -3].unsqueeze(1),
                                                 x.boxes.data[:, -2].unsqueeze(1),
                                                 x.boxes.data[:, -1].unsqueeze(1),
                                                 x.boxes.xywh), dim=1).detach().cpu().numpy(), columns=['_class', 'confidence_adult', 'confidence_child', 'x', 'y', 'w', 'h']) for x in detections]
            # out += [pd.DataFrame(data=torch.cat(( x.boxes.cls.unsqueeze(1),
            #                                       x.boxes.conf.unsqueeze(1),
            #                                       x.boxes.xywh), dim=1).detach().cpu().numpy(), columns=['class', 'confidence', 'x', 'y', 'w', 'h']) for x in detections]
        return [(i, d) for i, d in enumerate(out)]

    def _process(self, detections):
        if type(detections) is str:
            detections = read_pkl(out_path)
        dfs = []
        for frame, df in detections:
            df['confidence'] = (df['confidence_child'] - df['confidence_adult'] + 1) / 2
            to_remove = set()
            if df.empty:
                df = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
            elif len(df) > 1:
                boxes = xywh2xyxy(df[['x', 'y', 'w', 'h']].values)
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        boxes_iou = iou(boxes[i], boxes[j])
                        if boxes_iou > self.duplication_threshold:
                            to_remove.add(j if df['confidence'].iloc[i] > df['confidence'].iloc[j] else i)
            df['frame'] = frame
            # if frame > n:
            #     _n = n // 2
            #     prev, curr, next = [d.dropna() for d in dfs[-n:-n//2]], dfs[-n//2].dropna(), [d.dropna() for d in dfs[-n//2+1:]]
            #     prev, next = [d for d in prev if not d.empty], [d for d in next if not d.empty]
            #     if not curr.empty and (len(prev) > 0 or len(next) > 0):
            #         curr = ChildDetector.temporal_consistency(curr, prev, next)
            #         dfs[-n//2] = curr

            dfs.append(df.drop(list(to_remove)).reset_index(drop=True))
        df = pd.concat(dfs).reset_index(drop=True)
        return df

    def detect(self, video_path, out_path):
        detections_path = out_path.replace('.csv', '.pkl')
        if osp.exists(out_path):
            return pd.read_csv(out_path)
        if osp.exists(detections_path):
            detections = read_pkl(detections_path)
        else:
            detections = self._detect(video_path)
            write_pkl(detections, detections_path)
        df = self._process(detections)
        df.to_csv(out_path, index=False)
        return df

    # @staticmethod
    # def temporal_consistency(curr, prev_frames, next_frames, iou_threshold=0.5):
    #     curr_boxes = xywh2xyxy(curr[['x', 'y', 'w', 'h']].values)
    #     for i, curr_row in curr.iterrows():
    #         label_votes = [curr_row['class']]
    #         for neighbor in prev_frames + next_frames:
    #             if neighbor.empty:
    #                 continue
    #             neighbor_boxes = xywh2xyxy(neighbor[['x', 'y', 'w', 'h']].values)
    #             for j, neighbor_row in neighbor.iterrows():
    #                 if iou(curr_boxes[i], neighbor_boxes[j]) > iou_threshold:
    #                     label_votes.append(neighbor_row['class'])
    #         label_counts = Counter(label_votes)
    #         majority_label = label_counts.most_common(1)[0][0]
    #         curr.at[i, 'class'] = majority_label
    #     return curr

