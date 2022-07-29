import logging
import os
import random
import shlex
import shutil
import subprocess
from copy import deepcopy
from os import path

import pandas as pd
import torch
import numpy as np
from os import path as osp
import cv2
from skeleton_tools.pipe_components.openpose_initializer import OpenposeInitializer
from torch.utils.data import DataLoader

from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT, COCO_LAYOUT
from skeleton_tools.skeleton_visualization.numpy_visualizer import MMPoseVisualizer
from skeleton_tools.utils.skeleton_utils import bounding_box, box_distance, normalize_json, get_iou
from skeleton_tools.utils.tools import read_json, get_video_properties, read_pkl, write_pkl, init_logger

from child_detector.detection_dataset import ChildDetectionDataset
from pathlib import Path

MODEL_PATH = osp.join(Path(__file__).parent.parent, 'resources', 'model.pt')
class ChildDetector:
    def __init__(self, min_iou=0.01, model_path=MODEL_PATH, batch_size=128):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.batch_size = batch_size
        self.device = self.model.model.device
        self.min_iou = min_iou

    def find_nearest(self, box, kp, score):
        cb = (box['xcenter'], box['ycenter']), (box['width'] / 1.1, box['height'] / 1.3)
        M = kp.shape[0]
        boxes = [bounding_box(kp[i].T, score[i]) for i in range(M)]
        iou = [get_iou(cb, b) for b in boxes]
        nearest = np.argmax(iou)
        return nearest, np.max(iou)

    def detect(self, video_path):
        ds = ChildDetectionDataset(video_path, self.batch_size)
        dfs = []
        for frames_batch in ds:
            detections = self.model(frames_batch, size=640)
            dfs += detections.pandas().xywh
        del ds
        return dfs

    def _straight_match(self, detections, kp, kps, cids, detected, boxes):
        for i, df in enumerate(detections):
            children = df[df['class'] == 1]
            if children.shape[0] == 0:
                continue
            child_box = children.loc[children['confidence'].idxmax()]
            cid, iou = self.find_nearest(child_box, kp[:, i, :, :], kps[:, i, :])
            if iou < self.min_iou:
                continue
            cids[i] = cid
            boxes[i] = np.array([child_box['xcenter'], child_box['ycenter'], child_box['width'], child_box['height']])
            detected[i] = child_box['confidence']
            if children.shape[0] > 1:  # TODO: In this case, maybe think not only about max score, but also about previous / futural detections?
                print('Multiple Children???')

    def _interpolate(self, detections, kp, kps, cids, detected, boxes):
        env = [{} for _ in detected]
        c, d = 0, 10
        def scan(lst, key, reverse):
            tmp = None
            for i, child_conf in reversed(list(enumerate(lst))) if reverse else enumerate(lst):
                if child_conf > c:
                    tmp = i
                env[i][key] = tmp
        scan(detected, 'prev', reverse=False)
        scan(detected, 'next', reverse=True)

        for i, df in enumerate(detections):
            if detected[i] > c:
                continue
            prev, next = env[i]['prev'], env[i]['next']
            if not (prev and np.abs(prev - i) < d) or (next and np.abs(next - i) < d):
                continue
            j = prev if next is None else next if prev is None else prev if abs(i-prev) >= abs(next-i) else next
            _df = detections[j]
            children = _df[_df['class'] == 1]
            child_box = children.loc[children['confidence'].idxmax()]
            candidate, candidate_iou = self.find_nearest(child_box, kp[:, i, :, :], kps[:, i, :])
            if candidate_iou < self.min_iou:
                continue
            adults = df[df['class'] == 0]
            adults_matches = [self.find_nearest(adult_box, kp[:, i, :, :], kps[:, i, :]) for _, adult_box in adults.iterrows()]
            conflicts = [(a, iou) for a, iou in adults_matches if a == candidate and iou > self.min_iou]
            if len(conflicts) > 1:
                print('MULTIPLE CONFLICTS#!@$!@')
            # if any(conflicts):
            #     _, rival_iou = conflicts[0] #  TODO: what if rival is actually mis-annotated child? What about multiple rivals?
            #     if rival_iou > candidate_iou:
            #         continue
            cids[i] = candidate
            boxes[i] = np.array([child_box['xcenter'], child_box['ycenter'], child_box['width'], child_box['height']])

    def match_skeleton(self, skeleton, detections):
        skeleton = skeleton.copy()
        skeleton['child_ids'] = np.ones(skeleton['keypoint'].shape[1]) * -1
        skeleton['child_detected'] = np.zeros(skeleton['keypoint'].shape[1])
        skeleton['child_bbox'] = np.zeros((skeleton['keypoint'].shape[1], 4))
        cids = skeleton['child_ids']
        detected = skeleton['child_detected']
        boxes = skeleton['child_bbox']
        kp = skeleton['keypoint']
        kps = skeleton['keypoint_score']
        self._straight_match(detections, kp, kps, cids, detected, boxes)
        self._interpolate(detections, kp, kps, cids, detected, boxes) # TODO: Third pass, clear bounding boxes that are alone in a window of size d?
        # reverse_pass, child_box = False, None
        # first_box, first_detection = None, None
        # for i, df in enumerate(detections):
        #     children = df[df['class'] == 1]
        #     if children.shape[0] == 1:
        #         child_box = children.iloc[0]
        #         if first_detection is None:
        #             first_detection, first_box = i, child_box
        #         cid = self.find_nearest(child_box, kp[:, i, :, :], kps[:, i, :])
        #         skeleton['child_ids'][i] = cid
        #         skeleton['child_detected'][i] = 1
        #     elif child_box is not None:
        #         cid, iou = self.find_nearest(child_box, kp[:, i, :, :], kps[:, i, :], return_iou=True)
        #         if iou >= self.min_iou:
        #             skeleton['child_ids'][i] = cid
        #             skeleton['child_detected'][i] = 1
        #     else:
        #         reverse_pass = True
        # if reverse_pass and first_detection is not None:
        #     child_box = first_box
        #     for i in range(first_detection - 1, -1, -1):
        #         cid, iou = self.find_nearest(child_box, kp[:, i, :, :], kps[:, i, :], return_iou=True)
        #         if iou >= self.min_iou:
        #             skeleton['child_ids'][i] = cid
        #             skeleton['child_detected'][i] = 1
        #             (x, y), (w, h) = bounding_box(kp[cid, i].T, kps[cid, i].T)
        #             child_box = {'xcenter': x, 'ycenter': y, 'width': w * 1.1, 'height': h * 1.3, 'confidence': 0, 'class': 1, 'name': 'child'}
        # del ds
        return skeleton

    def filter(self, skeleton):
        kp = skeleton['keypoint']
        kps = skeleton['keypoint_score']
        out = skeleton.copy()
        out['keypoint'] = np.array([kp[cid, i] for i, cid in enumerate(skeleton['child_ids'])])
        out['keypoint_score'] = np.array([kps[cid, i] for i, cid in enumerate(skeleton['child_ids'])])
        return out

if __name__ == '__main__':
    random.seed(0)
    n = 20
    skeletons_dir = r'D:\datasets\lancet_submission_data\skeletons\raw'
    videos_dir = r'D:\datasets\lancet_submission_data\segmented_videos'
    out_dir = r'C:\Users\owner\Downloads\child_detect_tests'
    vis = MMPoseVisualizer(COCO_LAYOUT)
    cd = ChildDetector()
    skeletons = list(os.listdir(skeletons_dir))
    skeletons = random.sample(skeletons, n)
    videos = [v for v in os.listdir(videos_dir) if f'{osp.splitext(v)[0]}.pkl' in skeletons]
    videos.sort()
    skeletons.sort()
    for video, skeleton_name in zip(videos, skeletons):
        detections = cd.detect(osp.join(videos_dir, video))
        skeleton = read_pkl(osp.join(skeletons_dir, skeleton_name))
        if len(detections) != skeleton['keypoint'].shape[1]:
            raise IndexError(f'Frames mismatch for {video}')
        matched_skeleton = cd.match_skeleton(skeleton, detections)
        vis.create_video(osp.join(videos_dir, video), matched_skeleton, osp.join(out_dir, video))

    vids_root = r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\videos'
    jordi_root = r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\JORDIv3_detections'
    videos = ['664015048_ADOS_Clinical_011017_0000_3.mp4', '663981493_ADOS_Clinical_020216_0000_3.mp4', '1021400350_ADOS_Clinical_060617_0000_2.mp4']
    for v in videos:
        cid = v.split('_')[0]
        name, ext = osp.splitext(v)
        vpath = osp.join(vids_root, cid, v)
        detections = cd.detect(vpath)
        skeleton = read_pkl(osp.join(jordi_root, name, f'{name}_raw.pkl'))
        if len(detections) != skeleton['keypoint'].shape[1]:
            print(f'Frames mismatch: {len(detections)} detections for {skeleton["keypoint"].shape[1]} keypoints for {v}')
        matched_skeleton = cd.match_skeleton(skeleton, detections)
        vis.create_video(vpath, matched_skeleton, osp.join(out_dir, v))


# class ChildDetector:
#     def __init__(self,
#                  detector_root=r"C:\research\yolov5",
#                  model_path=r"C:\research\yolov5\runs\train\exp7\weights\best.pt",
#                  ffmpeg_root=r"C:\research\ffmpeg-N-101443-g74b5564fb5-win64-gpl\bin",
#                  resolution=(1280, 1024),
#                  data_centralized=False):
#         self.detector_root = detector_root
#         self.detection_dir = path.join(self.detector_root, 'runs', 'detect')
#         self.temp_dir = path.join(self.detector_root, 'runs', 'temp')
#         self.model_path = model_path
#         self.ffmpeg_root = ffmpeg_root
#         self.resolution = resolution
#         self.data_centralized = data_centralized
#
#     def _rescale_video(self, video_path, out_path):
#         width, height = self.resolution
#         subprocess.check_call(' '.join([path.join(self.ffmpeg_root, 'ffmpeg.exe'), '-i', f'"{video_path}"', '-vf', f'scale={width}:{height}', f'"{out_path}"']))
#
#     def _read_box(self, box_path):
#         with open(box_path, 'r') as f:
#             # children = [x.strip() for x in f.readlines() if x[0] == '1']
#             c_boxes = [[float(s) for s in x.strip().split(' ')[1:]] for x in f.readlines() if x[0] == '1']
#         return [(np.array((cx, cy)), np.array((w, h))) for cx, cy, w, h in c_boxes]
#
#     def _choose_box(self, boxes, prev_box=None):
#         new_box = []
#         if len(boxes) == 0 and prev_box is not None:
#             return prev_box
#         elif len(boxes) > 0:
#             if len(boxes) > 1 and prev_box is not None:
#                 distances = [box_distance(b, prev_box) for b in boxes]
#                 new_box = boxes[np.argmin(distances)]
#             else:
#                 new_box = boxes[0]
#         return new_box
#
#     def _collect_json(self, label_path):
#         data = []
#         box = None
#         last_known_box = box
#         for frame_index, file in enumerate(os.listdir(label_path)):
#             children = self._read_box(path.join(label_path, file))
#             if box is not None and len(box) > 0:
#                 last_known_box = box
#             box = self._choose_box(children, last_known_box)
#             if self.data_centralized:
#                 box[:2] -= 0.5
#             data.append({
#                 'frame_index': frame_index,
#                 'box': box
#             })
#         return data
#
#     def _detect_children_in_video(self, video_path, resolution=None):
#         name, ext = path.splitext(path.basename(video_path))
#         temp_scaled = path.join(self.temp_dir, f'{name}_scaled{ext}')
#         width, height = self.resolution if resolution is None else resolution
#         if path.exists(temp_scaled):
#             os.remove(temp_scaled)
#         if path.exists(path.join(self.detection_dir, name)):
#             shutil.rmtree(path.join(self.detection_dir, name))
#
#         try:
#             vid_res, _, _ = get_video_properties(video_path)
#             if vid_res != self.resolution:
#                 self._rescale_video(video_path, temp_scaled)
#                 video_path = temp_scaled
#             args = {
#                 'weights': f'"{self.model_path}"',
#                 'img': width,
#                 'source': f'"{video_path}"',
#                 'save-txt': '',
#                 'nosave': '',
#                 'project': f'"{self.detection_dir}"',
#                 'name': f'"{name}"'
#             }
#             python_path = r'C:\Users\owner\anaconda3\envs\yolo\python.exe'
#             cmd = f'"{python_path}" "{path.join(self.detector_root, "detect.py")}" {" ".join([f"--{k} {v}" for k, v in args.items()])}'
#             print(f'Executing: {cmd}')
#             subprocess.check_call(shlex.split(cmd), universal_newlines=True)
#             # subprocess.check_call(cmd)
#             box_json = self._collect_json(path.join(self.detection_dir, name, 'labels'))
#             return box_json
#         finally:
#             if path.exists(temp_scaled):
#                 os.remove(temp_scaled)
#             if path.exists(path.join(self.detection_dir, name)):
#                 shutil.rmtree(path.join(self.detection_dir, name))
#
#     def _match_skeleton(self, box, skeletons):
#         if box['box']:
#             (cx, cy), (w, h) = box['box']
#             distances = [box_distance((np.array([cx, cy]), np.array([w, h])), bounding_box((np.array([skel['pose'][0::2], skel['pose'][1::2]]).T / np.array(self.resolution)).T,
#                                                                                            np.array(skel['pose_score']))) for skel in skeletons]
#             return skeletons[np.argmin(distances)]['person_id']
#
#     def _match_video(self, box_json, video_json):
#         if not (any(box_json) or any(video_json)):
#             raise ValueError('Empty box / skeleton')
#         if box_json[-1]['frame_index'] != video_json[-1]['frame_index']:
#             print(f'Error - frames mismatch: Box: {len(box_json)}, Skeleton: {len(video_json)}')
#             length = np.min((len(box_json), len(video_json)))
#             box_json = box_json[:length]
#             video_json = video_json[:length]
#         pids = [self._match_skeleton(box, frame_info['skeleton']) if frame_info['skeleton'] else -1
#                 for box, frame_info in zip(box_json, video_json)]
#         return pids
#
#     def remove_adults(self, skeleton, video_path, resolution=None):
#         box_json = self._detect_children_in_video(video_path, resolution=resolution)
#         cids = self._match_video(box_json, skeleton)
#         data = [{'frame_index': frame_info['frame_index'], 'skeleton': [s for s in frame_info['skeleton'] if s['person_id'] == cid]}
#                 for frame_info, cid in zip(skeleton, cids)]
#         return data
