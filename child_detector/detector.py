# import logging
# from os import path as osp
# from pathlib import Path
#
# import torch
# from skeleton_tools.datasets.iterable_video_dataset import IterableVideoDataset
# from torch.utils.data import DataLoader
# from ultralytics import YOLO
# import pandas as pd
#
# from child_detector.facial_matcher import FaceMatcher
# from child_detector.skeleton_matcher import SkeletonMatcher
#
# # MODEL_PATH = osp.join(Path(__file__).parent.parent, 'resources', 'model.pt')
# MODEL_PATH = r'D:\repos\Child-Detector\child_detector\training\test_mid_train.pt'
#
# class ChildDetector:
#     def __init__(self, model_path=MODEL_PATH, batch_size=128, device=None):
#         handlers = list(logging.getLogger().handlers)
#         # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, _verbose=False)
#         self.model = YOLO(MODEL_PATH)
#         self.device = torch.device(device)
#         if self.device is not None:
#             self.model = self.model.to(self.device)
#         logging.getLogger().handlers = handlers
#         self.batch_size = batch_size
#
#     def detect(self, video_path):
#         # dataset = IterableVideoDataset(video_path, self.batch_size)
#         dataset = IterableVideoDataset(video_path)
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda x: x)
#         out = []
#         for frames_batch in dataloader:
#             # detections = self.model(frames_batch, size=640)
#             detections = self.model(frames_batch)
#             # dfs += detections.pandas().xywh
#             out += [pd.DataFrame(data=torch.cat(( x.boxes.cls.unsqueeze(1),
#                                                   x.boxes.conf.unsqueeze(1),
#                                                   x.boxes.xywh), dim=1).detach().cpu().numpy(), columns=['class', 'confidence', 'x', 'y', 'w', 'h']) for x in detections]
#         return [(i, d) for i, d in enumerate(out)]
#
#     def match_skeleton(self, skeleton, detections, iou_threshold=0.01, conf_threshold=0.1, similarity_threshold=0.85, grace_distance=20, tolerance=100):
#         m = SkeletonMatcher(iou_threshold=iou_threshold, conf_threshold=conf_threshold, similarity_threshold=similarity_threshold, grace_distance=grace_distance, tolerance=tolerance)
#         return m.match_skeleton(skeleton, detections)
#
#     def match_face(self, faces, groups, detections, iou_threshold=1e-5):
#         m = FaceMatcher(iou_threshold=iou_threshold)
#         return m.match_face(faces, groups, detections)
