# import numpy as np
# from skeleton_tools.utils.skeleton_utils import get_iou
#
#
# def get_box(row):
#     return np.array([row['x'], row['y'], row['w'], row['h']])
#
#
# def find_nearest(child_row, boxes):
#     cb = get_box(child_row)
#     iou = [get_iou(cb, b) for b in boxes]
#     nearest = np.argmax(iou)
#     return nearest, np.max(iou)