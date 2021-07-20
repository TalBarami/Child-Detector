import random

import pandas as pd
import cv2
import os
from os import path

class DataSampler:
    def __init__(self, videos_path, skeletons_path, tagged_df, out_path):
        self.videos_path = videos_path
        self.skeletons_path = skeletons_path
        self.df = tagged_df
        self.out_path = out_path

    def _sample(self, df, n):
        video_files = os.listdir(self.videos_path)
        i = 0
        while i < n:
            video_name, segment_name, start_time, end_time, start_frame, end_frame, status, action, child_ids, time, notes = df.sample(n=1).values[0]
            cap = cv2.VideoCapture(path.join(self.videos_path, [f for f in video_files if segment_name in f][0]))
            skel = read_json(path.join(self.skeletons_path, f'{segment_name}.json'))

            f = random.randint(0, len(skel['data']))
            candiadtes = [(arg, s['person_id']) for arg, s in enumerate(skel['data'][f]['skeleton'])]
            intersection = [(arg, c) for arg, c in candiadtes if c in child_ids]
            if any(intersection):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                if ret:
                    box = self.bounding_box(skel['data'][f]['skeleton'][intersection[0][0]])
                    cv2.imwrite(path.join(self.out_path, 'imgs', f'{segment_name}_{f}.png'), frame)

    def _bounding_box(self, skeleton):
        return 0

    def init_dataset(self, n, m):
        self._sample(self.df[(self.df['status'] == 'Status.OK') & (self.df['child_ids'] != "[-1]")], n)
        self._sample(self.df[(self.df['status'] == 'Status.OK') & self.df['child_ids'] == '[-1]'], m)


    def _sample_child(self, n):
        df = self.df[(self.df['status'] == 'Status.OK') & (self.df['child_ids'] != "[-1]")]
        self._sample(df, n)
        # Sample n from the valid df, then group by video and randomly select from this video from frames WHERE THE SKELETON IS AVAILABLE


def main():
    videos_path = r'D:\datasets\autism_center\segmented_videos'
    skeletons_path = r'D:\datasets\autism_center\skeletons\data'
    tagged_df = pd.read_csv(r'D:\datasets\autism_center\qa_dfs\merged.csv')
    out_path = r'D:\datasets\child_detector'
    ds = DataSampler(videos_path, skeletons_path, tagged_df, out_path)
    ds._sample_child(5)

if __name__ == '__main__':
    main()