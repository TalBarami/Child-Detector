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
from child_detector.detection_data import DetectionsData


MODEL_PATH = osp.join(Path(__file__).parent.parent, 'resources', 'models', 'child_detector_241115.pt')

class ChildDetector:
    def __init__(self, model_path=MODEL_PATH, duplication_threshold=0.9, batch_size=128, device=None):
        override_conf()
        handlers = list(logging.getLogger().handlers)
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)
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
        return [(i, d) for i, d in enumerate(out)]

    def detect(self, video_path, out_path=None):
        if osp.exists(out_path):
            return self.load(out_path)
        detections = self._detect(video_path)
        data = DetectionsData(detections, duplication_threshold=self.duplication_threshold)
        # df = self._process(detections)
        # data = DetectionData(detections_raw=detections, detections_processed=df)
        if out_path is not None:
            data.save(out_path)
        return data

        # detections_path = out_path.replace('.csv', '.pkl')
        # if osp.exists(out_path):
        #     return pd.read_csv(out_path)
        # if osp.exists(detections_path):
        #     detections = read_pkl(detections_path)
        # else:
        #     detections = self._detect(video_path)
        #     write_pkl(detections, detections_path)
        # df = self._process(detections)
        # df.to_csv(out_path, index=False)
        # return df

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

