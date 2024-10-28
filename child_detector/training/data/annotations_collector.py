from os import path as osp

import cv2
import numpy as np
import pandas as pd
from skeleton_tools.utils.tools import read_pkl
from sklearn.model_selection import train_test_split as _train_test_split

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class AnnotationsCollector:
    def __init__(self, df, root):
        self.df = df
        self.df['video_path'] = self.df['video_path'].apply(lambda p: p.replace('\\', '/').split('ChildDetect/', 1)[-1])
        self.df['data_path'] = self.df['data_path'].apply(lambda p: p.replace('\\', '/').split('ChildDetect/', 1)[-1])
        self.root = root
        self.out_dir = osp.join(root, 'training')

    def collect_data(self, percentage, visualize=False):
        out_dir = osp.join(self.out_dir, 'child_detect')
        images_dir = osp.join(out_dir, 'images')
        labels_dir = osp.join(out_dir, 'labels')
        images_labeled_dir = osp.join(out_dir, 'images_labeled')
        for i, row in self.df.iterrows():
            print(f'{i}/{len(self.df)}')
            video_path = osp.join(self.root, row['video_path'])
            data_path = osp.join(self.root, row['data_path'])
            segment_name = row['segment_name']
            set = row['set']
            cids = [] if pd.isna(row['child_ids']) else eval(row['child_ids'])

            _n = row['end_frame'] - row['start_frame']
            _js = np.arange(0, _n, _n * percentage)
            if all(osp.exists(osp.join(images_dir, set, f'{segment_name}_{j}.jpg')) for j in _js) \
                    and all(osp.exists(osp.join(labels_dir, set, f'{segment_name}_{j}.txt')) for j in _js):
                continue

            cap = cv2.VideoCapture(video_path)
            data = read_pkl(data_path)
            n = len(data['data'])
            k = int(n * percentage)
            for j in range(n):
                if j % k == 0:
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
                else:
                    cap.grab()
            cap.release()


def train_test_split(df, val_size=0.2, random_state=42):
    cids = df['cid'].unique()
    _, cids_val = _train_test_split(cids, test_size=val_size, random_state=random_state)
    df['set'] = 'train'
    df.loc[df['cid'].isin(cids_val), 'set'] = 'val'
    df.loc[df['location'].isin(["Shamir_medical_Center (Asaf Ha'rofeh)", "Judah's Lab"]), 'set'] = 'test'
    return df


if __name__ == '__main__':
    root = r'Z:\Users\TalBarami\ChildDetect'
    # names_path = osp.join(root, 'children_to_analyze.xlsx')
    # sheet_names = ['Beer_Sheva', 'Judah', 'Tachana', 'Assaf Harofeh']
    # files = pd.read_excel(names_path, sheet_name=sheet_names)
    # for sheet, df in files.items():
    #     df['location'] = sheet
    # files = pd.concat(files.values()).dropna()
    # files['assessment'] = files['basename'].apply(lambda v: '_'.join(v.split('_')[:-2]))
    # files['ckey'] = files['assessment'].apply(lambda a: a.split('_')[0])
    # files['assessment_dir'] = files.apply(lambda row: osp.join(r'Z:\recordings\videos', row['ckey'], row['assessment']), axis=1)
    # files['video_path'] = files.apply(lambda row: osp.join(row['assessment_dir'], f'{row["filename"]}'), axis=1)
    files = pd.read_csv(osp.join(root, 'annotations.csv'))
    files = files.drop_duplicates(subset='assessment').set_index('assessment')
    df = pd.concat([pd.read_csv(osp.join(root, 'shaked.csv')), pd.read_csv(osp.join(root, 'noa.csv'))])
    df = df[(df['status'] == 'OK') | (df['status'] == 'No child')]
    df['cid'] = df['basename'].apply(lambda s: s.split('_')[0])
    df['assessment'] = df['basename'].apply(lambda s: '_'.join(s.split('_')[:-2]))
    df['location'] = df['assessment'].apply(lambda n: files.loc[n]['location'])
    df = train_test_split(df)
    ann = AnnotationsCollector(df, root)
    ann.collect_data(0.05, visualize=True)
