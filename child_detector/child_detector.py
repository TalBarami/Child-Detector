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

MODEL_PATH = read_json(Path.home().joinpath('.ancan', 'location_mapping.json'))['child_detector']

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
        df = pd.concat(out, keys=range(len(out)))
        df.reset_index(level=0, inplace=True)
        df.rename(columns={'level_0': 'frame'}, inplace=True)
        return df

    def detect(self, video_path, out_path=None):
        if osp.exists(out_path):
            return DetectionsData.load(out_path)
        detections = self._detect(video_path)
        data = DetectionsData(detections, duplication_threshold=self.duplication_threshold)
        if out_path is not None:
            data.save(out_path)
        return data
