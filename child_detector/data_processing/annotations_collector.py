import os
from os import path as osp
from datetime import datetime
import shutil

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as _train_test_split
from taltools.io.files import init_directories, read_pkl
from taltools.ancan.patients import add_patient_info

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class AnnotationsCollector:
    def __init__(self, df, root, out_dir, old_data_path=None):
        self.df = df
        self.df['video_path'] = self.df['video_path'].apply(lambda p: p.replace('\\', '/').split('ChildDetect/', 1)[-1])
        self.df['data_path'] = self.df['data_path'].apply(lambda p: p.replace('\\', '/').split('ChildDetect/', 1)[-1])
        self.root = root
        self.out_dir = out_dir
        self.old_data_path = old_data_path

    def collect_data(self, n_samples, visualize=False):
        images_dir = osp.join(self.out_dir, 'images')
        labels_dir = osp.join(self.out_dir, 'labels')
        images_labeled_dir = osp.join(self.out_dir, 'images_labeled')
        for set in ['train', 'val', 'test']:
            init_directories(*[osp.join(d, set) for d in [images_dir, labels_dir, images_labeled_dir]])
        df = self.df.reset_index(drop=True)
        for i, row in df.iterrows():
            print(f'{i}/{len(df)}')
            video_path = osp.join(self.root, row['video_path'])
            data_path = osp.join(self.root, row['data_path'])
            segment_name = row['segment_name']
            set = row['set']
            cids = [] if pd.isna(row['child_ids']) else eval(row['child_ids'])

            _n = row['end_frame'] - row['start_frame']
            _k = int(_n / n_samples)
            _js = np.arange(0, _n, _k).astype(int)
            if all(osp.exists(osp.join(images_dir, set, f'{segment_name}_{j}.jpg')) for j in _js) \
                    and all(osp.exists(osp.join(labels_dir, set, f'{segment_name}_{j}.txt')) for j in _js):
                continue

            cap = cv2.VideoCapture(video_path)
            data = read_pkl(data_path)
            n = len(data['data'])
            k = int(n / n_samples)
            for j in range(n):
                if j % k != 0:
                    cap.grab()
                    continue
                ret, frame = cap.read()
                if not ret:
                    break
                d = data['data'][j]['boxes']
                d = d[d.cls == 0]
                ids = d.id
                if ids is None:
                    label = []
                else:
                    boxes = d.xywhn.detach().cpu().numpy()
                    label = [(int(_id in cids), *box) for _id, box in zip(ids, boxes)]
                out_image = osp.join(images_dir, set, f'{segment_name}_{j}.jpg')
                out_label = osp.join(labels_dir, set, f'{segment_name}_{j}.txt')
                cv2.imwrite(out_image, frame)
                with open(out_label, 'w') as f:
                    for l in label:
                        f.write(' '.join(map(str, l)) + '\n')
                if visualize:
                    frame_labeled = frame.copy()
                    boxes_raw = d.xywh.detach().cpu().numpy()
                    for l, box in zip(label, boxes_raw):
                        color = (0, 0, 255) if l[0] else (255, 0, 0)
                        x, y, w, h = box.astype(int)
                        cv2.rectangle(frame_labeled, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
                        cv2.putText(frame_labeled, str(l[0]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.imwrite(osp.join(images_labeled_dir, f'{segment_name}_{j}.jpg'), frame_labeled)
            cap.release()

        if self.old_data_path:
            for set_type in ['train', 'val']:
                old_img_dir = osp.join(self.old_data_path, set_type, 'images')
                new_img_dir = osp.join(self.out_dir, 'images', set_type)
                old_lbl_dir = osp.join(self.old_data_path, set_type, 'labels')
                new_lbl_dir = osp.join(self.out_dir, 'labels', set_type)
                for f in os.listdir(old_img_dir):
                    if not osp.exists(osp.join(new_img_dir, f)):
                        shutil.copy(osp.join(old_img_dir, f), new_img_dir)
                for f in os.listdir(old_lbl_dir):
                    with open(osp.join(old_lbl_dir, f), 'r') as file:
                        boxes = np.array([list(map(float, l.strip().split())) for l in file.readlines()])[:]
                    try:
                        if len(boxes) > 0:
                            boxes[:, 3] *= 2
                            boxes[:, 4] *= 2.25
                            boxes[:, 2] -= boxes[:, 4] * 0.05
                            labels, boxes = boxes[:, 0].astype(int), boxes[:, 1:]
                        with open(osp.join(new_lbl_dir, f), 'w') as file:
                            if boxes.any():
                                for label, box in zip(labels, boxes):
                                    file.write(str(label) + ' ' + ' '.join(list(map(str, list(box)))) + '\n')
                    except Exception as e:
                        print(1)


def train_test_split(df, val_size=0.2, test_size=0.1, random_state=42):
    test_locations = ["Shamir_medical_Center (Asaf Ha'rofeh)", "Judah's Lab"]
    df = add_patient_info(df, video_name_col='basename')
    df['location'] = df['location'].apply(lambda l: l if l in test_locations else 'Soroka')
    cids = df[~df['location'].isin(test_locations)]['child_key'].unique()

    y = df[df['child_key'].isin(cids)].groupby('child_key').first()[['gender', 'age_bin']]
    y = y.dropna()
    cids = y.index

    _, cids = _train_test_split(cids, test_size=val_size+test_size, stratify=y, random_state=random_state)
    n_val = int(len(cids) * val_size / (val_size + test_size))
    cids_val, cids_test = cids[:n_val], cids[n_val:]
    df['set'] = 'train'
    df.loc[df['child_key'].isin(cids_val), 'set'] = 'val'
    df.loc[df['child_key'].isin(cids_test) | df['location'].isin(test_locations), 'set'] = 'test'
    return df


if __name__ == '__main__':
    root = r'Z:\Users\TalBarami\ChildDetect'
    out_dir = r'E:\datasets\child_detection'
    dataset_path = osp.join(out_dir, 'dataset.csv')
    # if osp.exists(dataset_path):
    #     df = pd.read_csv(dataset_path)
    # else:
    files = pd.read_csv(osp.join(root, 'annotations.csv'))
    files = files.drop_duplicates(subset='assessment').set_index('assessment')
    old_ann_root = r'Z:\Users\TalBarami\ChildDetect\deprecated'

    df = pd.concat([pd.read_csv(osp.join(root, 'shaked.csv')), pd.read_csv(osp.join(root, 'noa.csv'))] +
                   [pd.read_csv(osp.join(old_ann_root, f)) for f in os.listdir(old_ann_root) if f.endswith('.csv') and ('noa' in f or 'shaked' in f)])
    df = df[(df['status'] == 'OK') | (df['status'] == 'No child') | (df['status'] == 'No Child')]
    df = df.drop_duplicates(subset=['basename', 'start_frame'])
    df = df[df['segment_name'].apply(lambda p: osp.exists(osp.join(root, 'data', f'{p}.mp4')) and osp.exists(osp.join(root, 'data', f'{p}.pkl')))]
    # df['_id'] = df.apply(lambda row: f'{row["basename"]}_{row["start_frame"]}', axis=1)
    df['child_key'] = df['basename'].apply(lambda s: s.split('_')[0])
    df['assessment'] = df['basename'].apply(lambda s: '_'.join(s.split('_')[:-2]))
    df['location'] = df['assessment'].apply(lambda n: files.loc[n]['location'])
    df = train_test_split(df)
    df.to_csv(dataset_path, index=False)
    ann = AnnotationsCollector(df, root, out_dir, old_data_path=r'Z:\Users\TalBarami\ChildDetect\training\child_detect\old_data')
    ann.collect_data(10, visualize=True)
