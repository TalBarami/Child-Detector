import os
import shutil
from os import path as osp
from datetime import datetime

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

from skeleton_tools.utils.tools import write_pkl
from skeleton_tools.pipe_components.yolo_tracker import YOLOTracker
matplotlib.use('TkAgg')

def prepare_test_sample():
    root = r'Z:/Users/TalBarami/ChildDetect/training'
    test_root = osp.join(root, 'sample')
    test_images_dir = osp.join('child_detect', 'images', 'test')
    test_labels_dir = osp.join('child_detect', 'labels', 'test')
    files = [f for f in os.listdir(osp.join(root, test_images_dir))]
    test_images = np.random.choice(files, 10000, replace=False)
    test_labels = [f.replace('.jpg', '.txt') for f in test_images]
    for f in test_images:
        # print(f'Copy: {osp.join(test_images_dir, f)} -> {osp.join(test_root, "images", "test", f)}')
        shutil.copyfile(osp.join(root, test_images_dir, f), osp.join(test_root, test_images_dir, f))
    for f in test_labels:
        # print(f'Copy: {osp.join(test_labels_dir, f)} -> {osp.join(test_root, "labels", "test", f)}')
        shutil.copyfile(osp.join(root, test_labels_dir, f), osp.join(test_root, test_labels_dir, f))

def collect_labels(labels_dir):
    if osp.exists('test_labels.csv'):
        return pd.read_csv('test_labels.csv')
    out = {}
    for f in os.listdir(labels_dir):
        with open(osp.join(labels_dir, f), 'r') as file:
            lines = [line.strip().split() for line in file.readlines()]
        labels = [int(l[0]) for l in lines]
        boxes = [[float(l[1]), float(l[2]), float(l[3]), float(l[4])] for l in lines]
        out[osp.splitext(f)[0]] = {'labels': np.array(labels), 'y_boxes': np.array(boxes)}
    df = pd.DataFrame(out).T.reset_index().rename(columns={'index': 'file'})
    df.to_csv('test_labels.csv')
    return out

def collect_results(images_dir):
    if osp.exists('test_predictions.csv'):
        return pd.read_csv('test_predictions.csv')
    results = model.predict(images_dir)
    out = {}
    for r in results:
        f = osp.splitext(osp.basename(r.path))[0]
        boxes = r.boxes.xywhn.detach().cpu().numpy()
        preds = r.boxes.conf.detach().cpu().numpy()
        classes = r.boxes.cls.detach().cpu().numpy()
        out[f] = {'p_boxes': boxes, 'preds': preds, 'classes': classes}
    df = pd.DataFrame(out).T.reset_index().rename(columns={'index': 'file'})
    df.to_csv('test_predictions.csv', index=False)
    return df


def compute_iou(box1, box2):
    # box format: [x_min, y_min, width, height]
    x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

    # Calculate intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Intersection area
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Areas of the boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Union area
    union_area = box1_area + box2_area - inter_area

    # IoU calculation
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def map_boxes_by_iou(p_boxes, y_boxes):
    num_p = p_boxes.shape[0]
    num_y = y_boxes.shape[0]

    # Create a cost matrix where each entry is -IoU (because we want to maximize IoU)
    cost_matrix = np.zeros((num_p, num_y))
    for i in range(num_p):
        for j in range(num_y):
            cost_matrix[i, j] = -compute_iou(p_boxes[i], y_boxes[j])

    # Use the Hungarian algorithm to find the optimal mapping
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create the mapping: {prediction_index: ground_truth_index}
    mapping = {row: col for row, col in zip(row_ind, col_ind)}

    return mapping


def convert_to_datetime(date_str):
    # Case when the string is in "MDYYYY" format
    if len(date_str) == 6:
        # Check if it's likely a 'MMDDYY' format
        try:
            return datetime.strptime(date_str, '%d%m%y')
        except ValueError:
            pass

        return datetime.strptime(date_str, '%d%m%Y')

    # Case when the string is in "DDMMYYYY" format
    elif len(date_str) == 8:
        return datetime.strptime(date_str, '%d%m%Y')

    # Add other specific cases if necessary, or return None
    return None

if __name__ == '__main__':
    # from mmpose.apis import MMPoseInferencer
    #
    # inferencer = MMPoseInferencer('human')
    # result_generator = inferencer(img_path, show=True)
    # for r in result_generator:
    #     print(r)

    # prepare_test_sample()
    model = YOLO('test_mid_train.pt')
    root = r'Z:/Users/TalBarami/ChildDetect/training'
    test_root = osp.join(root, 'sample')
    labels = collect_labels(osp.join(test_root, 'child_detect', 'labels', 'test')).set_index('file')
    results = collect_results(osp.join(test_root, 'child_detect', 'images', 'test')).set_index('file')
    _df = pd.merge(labels, results, left_index=True, right_index=True)
    thresholds = np.arange(0.05, 1, 0.05)
    precisions, recalls = [], []

    def to_np(np_str):
        n_rows = np_str.count('\n') + 1
        np_arr = np.fromstring(np_str.replace('\n', '').replace('[', '').replace(']', '').strip(), sep=' ')
        return np_arr.reshape(n_rows, -1)

    df = pd.DataFrame(columns=['file', 'label', 'conf', 'prediction'])
    for name, row in _df.iterrows():
        p_boxes = to_np(row['p_boxes'])
        y_boxes = to_np(row['y_boxes'])
        if p_boxes.size == 0 or y_boxes.size == 0:
            continue
        labels = to_np(row['labels']).reshape(-1)
        preds = to_np(row['classes']).reshape(-1)
        conf = to_np(row['preds']).reshape(-1)
        mapping = map_boxes_by_iou(p_boxes, y_boxes)
        for i, j in mapping.items():
            df.loc[df.shape[0]] = [name, labels[j], conf[i], preds[i]]
    df['assessment'] = df['file'].apply(lambda x: '_'.join(x.split('_')[:4]))
    # df['cid'] = df['file'].apply(lambda x: int(x.split('_')[0]))
    r = pd.DataFrame(columns=['cid', 'date', 'threshold', 'tp', 'fp', 'tn', 'fn'])
    for assessment, g in df.groupby('assessment'):
        cid = int(assessment.split('_')[0])
        for t in thresholds:
            g = g[g['conf'] > t]
            tp = g[(g['label'] == 1) & (g['prediction'] == 1)].shape[0]
            fp = g[(g['label'] == 0) & (g['prediction'] == 1)].shape[0]
            tn = g[(g['label'] == 0) & (g['prediction'] == 0)].shape[0]
            fn = g[(g['label'] == 1) & (g['prediction'] == 0)].shape[0]
            r.loc[r.shape[0]] = [cid, assessment.split('_')[-1], t, tp, fp, tn, fn]

    for t, g in r.groupby('threshold'):
        precision, recall = g['tp'].sum() / (g['tp'].sum() + g['fp'].sum()), g['tp'].sum() / (g['tp'].sum() + g['fn'].sum())
        print(f'Threshold: {t:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')

    cinf = pd.read_csv(r'Z:\Users\TalBarami\lists_fromL\child_detection_children_with_db_info.csv')
    cinf = cinf[cinf['set'] == 'test']
    cinf['cid'] = cinf['child_key']
    cinf['date'] = cinf['date'].astype(str)
    cinf['date_of_birth'] = pd.to_datetime(cinf['Date_of_birth'], dayfirst=True)
    rm = pd.merge(r, cinf, on=['cid', 'date'], how='inner')
    rm['date'] = rm['date'].apply(convert_to_datetime)
    rm['age'] = (rm['date'] - rm['date_of_birth']).dt.days / 365.25

    # groupby age bins: [0-3], [3-5], [5,7], [7-9], [9+]
    bins = [0, 3, 5, 7, 9, 100]
    labels = ['0-3', '3-5', '5-7', '7-9', '9+'] # Note: too few data for 7+
    rm['age_bin'] = pd.cut(rm['age'], bins=bins, labels=labels)
    r_age = rm.groupby(['age_bin', 'threshold'])[['tp', 'fp', 'tn', 'fn']].sum().reset_index()
    r_age['precision'] = r_age['tp'] / (r_age['tp'] + r_age['fp'])
    r_age['recall'] = r_age['tp'] / (r_age['tp'] + r_age['fn'])
    r_age = r_age.sort_values(by=['age_bin', 'threshold'])
    # plot:
    fig, ax = plt.subplots()
    for age, g in r_age.groupby('age_bin'):
        ax.plot(g['recall'], g['precision'], label=age)
        for t in thresholds:
            ax.text(g['recall'].iloc[list(thresholds).index(t)], g['precision'].iloc[list(thresholds).index(t)], f'{t:.2f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve by Age')
    ax.legend()
    plt.show()

    r_gender = rm.groupby(['Sex', 'threshold'])[['tp', 'fp', 'tn', 'fn']].sum().reset_index()
    r_gender['precision'] = r_gender['tp'] / (r_gender['tp'] + r_gender['fp'])
    r_gender['recall'] = r_gender['tp'] / (r_gender['tp'] + r_gender['fn'])
    r_gender = r_gender.sort_values(by=['Sex', 'threshold'])
    # plot:
    fig, ax = plt.subplots()
    for gender, g in r_gender.groupby('Sex'):
        ax.plot(g['recall'], g['precision'], label=gender)
        for t in thresholds:
            ax.text(g['recall'].iloc[list(thresholds).index(t)], g['precision'].iloc[list(thresholds).index(t)], f'{t:.2f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve by Gender')
    ax.legend()
    plt.show()





    # plot precision_recall curve:
    all = r.groupby('threshold').sum()
    fig, ax = plt.subplots()
    precisions, recalls = all['tp'] / (all['tp'] + all['fp']), all['tp'] / (all['tp'] + all['fn'])
    ax.plot(recalls, precisions)
    for t in thresholds:
        ax.text(recalls[int(t*10)-1], precisions[int(t*10)-1], f'{t:.1f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    plt.show()

    #     results = model.val(data=osp.join(test_root, 'child_detect_test.yaml'), conf=t)
    #     precision = results.results_dict['metrics/precision(B)']
    #     recall = results.results_dict['metrics/recall(B)']
    #     precisions.append(precision)
    #     recalls.append(recall)
    # print(1)
    # data = {'thresholds': thresholds, 'precisions': precisions, 'recalls': recalls}
    # write_pkl(data, osp.join(test_root, 'results.pkl'))
    # fig, ax = plt.subplots()
    # # precision-recall curve:
    # ax.plot(recalls, precisions)
    # for t in thresholds:
    #     ax.text(recalls[int(t*10)-1], precisions[int(t*10)-1], f'{t:.1f}')
    # ax.set_xlabel('Recall')
    # ax.set_ylabel('Precision')
    # ax.set_title('Precision-Recall Curve')
    # fig.savefig(osp.join(test_root, 'precision_recall_curve.png'))
    # plt.show()
    #
    # train = pd.DataFrame({'file': list(set(['_'.join(x.split('_')[:4]) for x in os.listdir(osp.join(root, 'child_detect', 'labels', 'train'))])), 'set': 'train'})
    # val = pd.DataFrame({'file': list(set(['_'.join(x.split('_')[:4]) for x in os.listdir(osp.join(root, 'child_detect', 'labels', 'val'))])), 'set': 'val'})
    # test = pd.DataFrame({'file': list(set(['_'.join(x.split('_')[:4]) for x in os.listdir(osp.join(root, 'child_detect', 'labels', 'test'))])), 'set': 'test'})
    # df = pd.concat([train, val, test])
    # df['child_key'] = df['file'].apply(lambda x: x.split('_')[0])
    # df['date'] = df['file'].apply(lambda x: x.split('_')[3])
    # df['type'] = df['file'].apply(lambda x: x.split('_')[1])
    # df.to_csv(osp.join(test_root, 'child_detection_children.csv'), index=False)



    # precision = metrics['precision']
    # recall = metrics['recall']

    # Display the metrics for the current confidence threshold
    # print(f"Confidence threshold: {t}")
    # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    # print("="*50)