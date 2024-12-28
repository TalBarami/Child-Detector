import os
import shutil
from os import path as osp
from datetime import datetime

from tqdm import tqdm
import matplotlib
import pandas as pd
from child_detector.confidence_overrider import override_conf
from child_detector.data_processing.yolo_tracker import YOLOTracker
from matplotlib import pyplot as plt
import numpy as np
from taltools.io.files import write_pkl, init_directories
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

matplotlib.use('TkAgg')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


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

def prepare_test_data():
    data_dir = r'E:\datasets\child_detection'
    override_conf()
    model = YOLO(r'D:\repos\Child-Detector\runs\241115.pt')

    images_dir = osp.join(data_dir, 'images', 'test')
    labels_dir = osp.join(data_dir, 'labels', 'test')

    df = pd.DataFrame(columns=['file' ,'label', 'conf0', 'conf1'])
    images = list(os.listdir(images_dir))
    for image in tqdm(images):
        label_file = osp.join(labels_dir, image.replace('.jpg', '.txt'))
        label = pd.read_csv(label_file, header=None, sep=' ', names=['label', 'x', 'y', 'w', 'h'])
        pred = model.predict(osp.join(images_dir, image))[0].boxes
        y_boxes = label[['x', 'y', 'w', 'h']].values
        p_boxes = pred.xywhn.detach().cpu().numpy()
        p_conf = pred.data.detach().cpu().numpy()[:, -2:]
        pred = pd.DataFrame(np.concatenate((p_boxes, p_conf), axis=1), columns=['x', 'y', 'w', 'h', 'conf0', 'conf1'])
        if label.shape[0] == 0 or pred.shape[0] == 0:
            continue
        mapping = map_boxes_by_iou(p_boxes, y_boxes)
        for i, j in mapping.items():
            df.loc[df.shape[0]] = [image, label.loc[j, 'label'], pred.loc[i, 'conf0'], pred.loc[i, 'conf1']]
    return df

def plot_precision_recall(df, group_col, title, out_dir, t=None):
    fig, ax = plt.subplots()
    for n, g in df.dropna().groupby(group_col):
        ax.plot(g['recall'], g['precision'], label=n)
        if t is not None:
            ax.text(g['recall'].iloc[t], g['precision'].iloc[t], f'{thresholds[t]:.2f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve by {title}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, f'precision_recall_curve_{group_col}.png'))

def plot_metrics(df, group_col, title, out_dir, t=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['precision', 'recall', 'f1']
    df['f1'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'])
    for n, g in df.dropna().groupby(group_col):
        for i, metric in enumerate(metrics):
            axs[i].plot(g['threshold'], g[metric], label=n)
        if t is not None:
            for i, metric in enumerate(metrics):
                axs[i].text(g['threshold'].iloc[t], g[metric].iloc[t], f'{thresholds[t]:.2f}')
                # horizontal and vertical dotted lines at the best threshold
                axs[i].axvline(x=thresholds[t], color='gray', linestyle='--')
                axs[i].axhline(y=g[metric].iloc[t], color='gray', linestyle='--')
    for i, metric in enumerate(metrics):
        axs[i].set_xlabel('Threshold')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].set_title(f'{metric.capitalize()} by {title}')
        axs[i].legend()
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, f'metrics_{group_col}.png'))

def plot_conf(_df):
    for feature, dfs in [('Location', list(_df.groupby('location'))), ('Age', list(_df.groupby('age_bin')))]:
        fig, ax = plt.subplots(len(dfs), 2, figsize=(10, 2.5 * len(dfs)))
        for i, (name, df) in enumerate(dfs):
            for j, color in [(0, 'blue'), (1, 'red')]:
                conf = df[df['label'] == j]['conf']
                ax[i, j].hist(conf, bins=10, edgecolor='black', color=color, alpha=0.7)
                ax[i, j].set_title(f'(Label = {j}, {feature} = {name})')
                ax[i, j].set_xlabel('Confidence')
                ax[i, j].set_ylabel('Frequency')
                ax[i, j].grid(axis='y', linestyle='--', alpha=0.7)
        fig.suptitle(f'Confidence Distribution by Label and {feature}')
        fig.tight_layout()
        fig.savefig(osp.join(r'Z:\Users\TalBarami\ChildDetect\presentation', f'conf_{feature}.png'))
        plt.show()


if __name__ == '__main__':
    out_dir = r'Z:\Users\TalBarami\ChildDetect\presentation'
    dataset_file = osp.join(out_dir, 'test_results.csv')
    if osp.exists(dataset_file):
        df = pd.read_csv(dataset_file)
    else:
        df = prepare_test_data()
        df.to_csv(dataset_file, index=False)
    thresholds = np.arange(0.05, 1.00, 0.05)
    df['assessment'] = df['file'].apply(lambda x: '_'.join(x.split('_')[:4]))
    df['cid'] = df['file'].apply(lambda x: int(x.split('_')[0]))
    df['date'] = df['file'].apply(lambda x: x.split('_')[3])
    df['_conf'] = df['conf1'] - df['conf0']
    df['conf'] = (df['_conf'] + 1) / 2

    exclude = ['1021775038_ADOS_Clinical_060417', '683757931_ADOS_Clinical_150720']
    df = df[~df['assessment'].isin(exclude)]

    cinf = pd.read_csv(r'Z:\Users\TalBarami\241031_children_info.csv')
    cols = [0, 3, 4, 5]
    cinf = cinf.iloc[:, cols].dropna()
    cinf.columns = ['child_key', 'gender', 'date_of_birth', 'location']
    cinf['child_key'] = cinf['child_key'].astype(int)
    cinf['date_of_birth'] = pd.to_datetime(cinf['date_of_birth'], dayfirst=True)
    cinf.set_index('child_key', inplace=True)

    test_locations = {"Shamir_medical_Center (Asaf Ha'rofeh)": "Shamir Medical Center", "Judah's Lab": "Judah's Lab"}
    df['location'] = df['cid'].apply(lambda x: cinf.loc[x, 'location']).apply(lambda x: test_locations[x] if x in test_locations else 'Soroka')
    df['date'] = df['date'].apply(convert_to_datetime)
    df['date_of_birth'] = df['cid'].apply(lambda x: cinf.loc[x, 'date_of_birth'])
    df['age'] = (df['date'] - df['date_of_birth']).dt.days / 365.25
    df['gender'] = df['cid'].apply(lambda x: cinf.loc[x, 'gender'])
    # bins = [0, 2, 4, 6, 8, 10, 100]
    # labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10+']
    df = df[df['age'] <= 8]
    bins = [0, 2, 4, 6, 8]
    labels = ['0-2', '2-4', '4-6', '6-8']
    df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels)

    plot_conf(df)

    df_all = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'accuracy'])
    df_location = pd.DataFrame(columns=['location', 'threshold', 'precision', 'recall', 'accuracy'])
    df_age = pd.DataFrame(columns=['age_bin', 'threshold', 'precision', 'recall', 'accuracy'])
    df_gender = pd.DataFrame(columns=['gender', 'threshold', 'precision', 'recall', 'accuracy'])
    for t in thresholds:
        df['pred'] = df['conf'] > t
        precision, recall = (df['pred'] & df['label']).sum() / df['pred'].sum(), (df['pred'] & df['label']).sum() / df['label'].sum()
        accuracy = (df['pred'] == df['label']).mean()
        df_all.loc[df_all.shape[0]] = [t, precision, recall, accuracy]

        location_df = df.groupby('location')[['label', 'pred']]
        for location, g in location_df:
            precision, recall = (g['pred'] & g['label']).sum() / g['pred'].sum(), (g['pred'] & g['label']).sum() / g['label'].sum()
            accuracy = (g['pred'] == g['label']).mean()
            df_location.loc[df_location.shape[0]] = [location, t, precision, recall, accuracy]

        age_df = df[df['location'] == 'Soroka'].groupby('age_bin')[['label', 'pred']]
        for age, g in age_df:
            precision, recall = (g['pred'] & g['label']).sum() / g['pred'].sum(), (g['pred'] & g['label']).sum() / g['label'].sum()
            accuracy = (g['pred'] == g['label']).mean()
            df_age.loc[df_age.shape[0]] = [age, t, precision, recall, accuracy]

        gender_df = df[df['location'] == 'Soroka'].groupby('gender')[['label', 'pred']]
        for gender, g in gender_df:
            precision, recall = (g['pred'] & g['label']).sum() / g['pred'].sum(), (g['pred'] & g['label']).sum() / g['label'].sum()
            accuracy = (g['pred'] == g['label']).mean()
            df_gender.loc[df_gender.shape[0]] = [gender, t, precision, recall, accuracy]
    df_soroka = df_location[df_location['location'] == 'Soroka']
    f1 = 2 * df_soroka['precision'] * df_soroka['recall'] / (df_soroka['precision'] + df_soroka['recall'])
    best_f1 = f1.argmax()
    plot_precision_recall(df_location, 'location', 'Location', out_dir, t=best_f1)
    plot_precision_recall(df_gender, 'gender', 'Gender (Soroka - Test)', out_dir, t=None)
    plot_precision_recall(df_age, 'age_bin', 'Age (Soroka - Test)', out_dir, t=None)

    plot_metrics(df_location, 'location', 'Location', out_dir, t=None)
    plot_metrics(df_gender, 'gender', 'Gender', out_dir, t=None)
    plot_metrics(df_age, 'age_bin', 'Age', out_dir, t=None)
