import logging
from os import path as osp
from pathlib import Path

import cv2
import pandas as pd
import torch
from taltools.cv.iterable_video_dataset import IterableVideoDataset
from taltools.cv.videos import get_video_properties
from taltools.io.files import read_json
from torch.utils.data import DataLoader
from ultralytics import YOLO

from child_detector.confidence_overrider import override_conf
from child_detector.detection_data import DetectionsData

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
MODEL_PATH = read_json(Path.home().joinpath('.ancan', 'location_mapping.json'))['child_detector']

class ChildDetector:
    def __init__(self, model_path=MODEL_PATH, batch_size=128, device=None):
        override_conf()
        handlers = list(logging.getLogger().handlers)
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)
        self.batch_size = batch_size
        self.cols = ['_class', 'confidence_adult', 'confidence_child', 'x', 'y', 'w', 'h']

        logging.getLogger().handlers = handlers

    def _detect(self, video_path):
        (width, height), fps, _, _ = get_video_properties(video_path)
        dataset = IterableVideoDataset(video_path)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda x: x)
        out = []
        idx = 0
        for frames_batch in dataloader:
            detections, idx = self._detect_batch(frames_batch, idx)
            out += detections
        df = pd.concat(out, ignore_index=True).set_index('frame')
        _missing = pd.DataFrame(index=list(set(range(idx + 1)) - set(df.index)), columns=self.cols).rename_axis('frame')
        df = pd.concat([df, _missing], ignore_index=False).sort_index().reset_index()
        df.n_frames = idx+1
        df.fps = fps
        df.width = width
        df.height = height
        return df

    def _detect_batch(self, data_batch, idx=0):
        detections = self.model(data_batch)
        result = [pd.DataFrame(data=torch.cat((x.boxes.data[:, -3].unsqueeze(1),
                                             x.boxes.data[:, -2].unsqueeze(1),
                                             x.boxes.data[:, -1].unsqueeze(1),
                                             x.boxes.xywh), dim=1).detach().cpu().numpy(),
                             columns=self.cols).assign(frame=idx+i) for i, x in enumerate(detections)]
        new_idx = idx + len(detections)
        return result, new_idx

    def detect(self, video_path, out_path=None):
        if osp.exists(out_path):
            return DetectionsData.load(out_path)
        detections = self._detect(video_path)
        data = DetectionsData(detections)
        if out_path is not None:
            data.save(out_path)
        return data
