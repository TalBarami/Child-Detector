from os import path as osp

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from child_detector.data_processing.yolo_tracker import YOLOTracker
from taltools.io.files import init_directories, write_pkl
from taltools.logging.print_logger import PrintLogger

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

RESOURCES_ROOT = osp.join(Path(__file__).parent.parent, 'resources')

logger = PrintLogger(name='data_initializer')

class UniformSampler:
    def __init__(self, window_size, stride, p, root):
        self.window_size = window_size
        self.stride = stride
        self.p = p
        self.root = root
        self.ann_dir = osp.join(root, 'annotations')
        self.samples_dir = osp.join(root, 'data')
        self.tracker = YOLOTracker(osp.join(RESOURCES_ROOT, 'models', 'yolov8n.pt'))
        init_directories(self.ann_dir, self.samples_dir)

    def split(self, row):
        child_key, assessment, video_path, video_name, location = row['child_key'], row['assessment'], row['file_path'], row['basename'], row['location']
        out_file = osp.join(self.ann_dir, f'{video_name}_{self.window_size}_{self.stride}_{self.p}.csv')
        if osp.exists(out_file):
            return out_file
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        start_frames = list(range(0, total_frames - self.window_size, self.stride))
        df = pd.DataFrame(columns=['child_key', 'assessment', 'basename', 'location', 'start_frame', 'end_frame', 'fps', 'frame_count', 'tracked'])
        df['start_frame'] = start_frames
        df['child_key'] = row['child_key']
        df['assessment'] = row['assessment']
        df['basename'] = video_name
        df['location'] = location
        df['end_frame'] = df['start_frame'] + self.window_size
        df['fps'] = fps
        df['frame_count'] = total_frames
        df['tracked'] = -1
        step = int(1 / self.p)
        start = step // 2
        df.loc[start::step, 'tracked'] = 0
        df.to_csv(out_file, index=False)
        return out_file

    def write_n(self, cap, n, fps, clip_path):
        writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        for i in range(n):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()
        return ret

    def write(self, video_path, df_path):
        df = pd.read_csv(df_path)
        if df.empty:
            logger.error(f'Empty dataframe {df_path}')
            return
        video_name = df['basename'].iloc[0]
        fps = df['fps'].iloc[0]
        out_path = osp.join(self.samples_dir, f'{video_name}')
        _df = df[df['tracked'] == 0]
        cap = cv2.VideoCapture(video_path)
        for i, row in _df.iterrows():
            start_frame, end_frame = row['start_frame'], row['end_frame']
            clip_path = f'{out_path}_{start_frame}_{end_frame}.mp4'
            try:
                if not osp.exists(clip_path) or osp.getsize(clip_path) < 1e3:
                    logger.info(f'Writing {video_name}, frames {start_frame}-{end_frame}...')
                    ret = self.write_n(cap, end_frame-start_frame, fps, clip_path)
                    if not ret:
                        logger.error(f'Error writing {clip_path}.')
                        break
                out_track = f'{out_path}_{start_frame}_{end_frame}.pkl'
                if not osp.exists(out_track) or osp.getsize(out_track) < 1e3:
                    logger.info(f'Tracking {video_name}, frames {start_frame}-{end_frame}...')
                    track = self.tracker.track(clip_path)
                    logger.info(f'Writing {video_name} tracking results, frames {start_frame}-{end_frame}...')
                    write_pkl(track, out_track)
                df.loc[i, 'tracked'] = 1
            except Exception as e:
                logger.error(f'Error processing {video_name}, frames {start_frame}-{end_frame}: {e}')
                df.loc[i, 'tracked'] = -1
        df.to_csv(df_path, index=False)
        cap.release()

def collect_files(ckeys_path, out_path=None):
    db = read_db()
    children = pd.read_csv(ckeys_path).dropna().set_index('child_key')
    db = db[db['child_key'].isin(children.index)].reset_index(drop=True)
    db['location'] = db['child_key'].apply(lambda c: children.loc[c, 'Place_of_treatment'])

    if out_path is not None:
        db.to_csv(out_path, index=False)
    return db

if __name__ == '__main__':
    root = r'Z:\Users\TalBarami\ChildDetect'
    ckeys_path = osp.join(root, 'ckeys_list.csv')
    out_path = osp.join(root, 'annotations.csv')
    if osp.exists(out_path):
        files = pd.read_csv(out_path)
    else:
        files = collect_files(ckeys_path, out_path)
    logger.info(f'Found {files["basename"].nunique()} files, {files["assessment"].nunique()} assessments, {files["child_key"].nunique()} children.')

    sampler = UniformSampler(300, 300, p=0.1, root=root)
    # TODO: Remove later...
    files = files[files['location'].isin(["Shamir_medical_Center (Asaf Ha'rofeh)", "Judah's Lab"]) | files['child_key'].isin([1030000606, 1032641008, 683757931, 684000682, 699364708, 685885561, 684226150, 684212023, 707289343, 702304117])].reset_index(drop=True)
    n = files.shape[0]
    for i, row in files.iterrows():
        logger.info(f'({i}/{n}) Processing {row["basename"]}...')
        df_path = sampler.split(row)
        sampler.write(row['file_path'], df_path)