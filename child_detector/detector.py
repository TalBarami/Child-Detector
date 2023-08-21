import logging
import os
import random
from os import path as osp
from pathlib import Path

import torch
from skeleton_tools.openpose_layouts.body import COCO_LAYOUT
from skeleton_tools.utils.tools import read_pkl
from skeleton_tools.datasets.iterable_video_dataset import IterableVideoDataset

from child_detector.facial_matcher import FaceMatcher
from child_detector.skeleton_matcher import SkeletonMatcher

MODEL_PATH = osp.join(Path(__file__).parent.parent, 'resources', 'model.pt')


class ChildDetector:
    def __init__(self, model_path=MODEL_PATH, batch_size=128, num_worker=4, device=None):
        handlers = list(logging.getLogger().handlers)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.device = torch.device(device)
        if self.device is not None:
            self.model = self.model.to(self.device)
        logging.getLogger().handlers = handlers
        self.batch_size = batch_size
        self.num_worker = num_worker

    def detect(self, video_path):
        # dataset = IterableVideoDataset(video_path, self.batch_size)
        dataset = IterableVideoDataset(video_path, self.device)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_worker)
        dfs = []
        for frames_batch in ds:
            detections = self.model(frames_batch, size=640).detach().cpu()
            dfs += detections.pandas().xywh
        del ds
        return [(i, df) for i, df in enumerate(dfs)]

    def match_skeleton(self, skeleton, detections, iou_threshold=0.01, conf_threshold=0.1, similarity_threshold=0.85, grace_distance=20, tolerance=100):
        m = SkeletonMatcher(iou_threshold=iou_threshold, conf_threshold=conf_threshold, similarity_threshold=similarity_threshold, grace_distance=grace_distance, tolerance=tolerance)
        return m.match_skeleton(skeleton, detections)

    def match_face(self, faces, groups, detections, iou_threshold=1e-5):
        m = FaceMatcher(iou_threshold=iou_threshold)
        return m.match_face(faces, groups, detections)
