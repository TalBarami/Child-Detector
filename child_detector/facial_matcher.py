import numpy as np
from skeleton_tools.utils.skeleton_utils import get_iou

from child_detector.utils import get_box, find_nearest

def get_boxes(group):
    boxes = group.facebox.values[:, :4]
    boxes[:, 0] += boxes[:, 2] // 2
    boxes[:, 1] += boxes[:, 3] // 2
    return boxes[:, :4]

class FaceMatcher:
    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold

    def match_face(self, faces, detections):
        faces['is_child'] = 0
        groups = {i: g for i, g in faces.groupby('frame')}

        child_box = None
        for frame, df in detections:
            if frame not in groups.keys():
                continue
            children = df[df['class'] == 1]
            if children.shape[0] == 0:
                continue
            elif children.shape[0] > 1 and child_box is not None:
                print('Should do something here...')
                candidates = [(i, get_box(b)) for i, b in children.iterrows()]
                ious = [get_iou(get_box(child_box), b) for _, b in candidates]
                child_box = children.loc[candidates[np.argmax(ious)][0]]
                if not np.equal(child_box.values, children.loc[children['confidence'].idxmax()].values).all():
                    print('Different child box was chosen!')
            else:
                child_box = children.loc[children['confidence'].idxmax()]
            group = groups[frame]
            faceboxes = get_boxes(group)
            idx, iou = find_nearest(child_box, faceboxes)
            if iou < self.iou_threshold:
                continue
            loc = group.index[idx]
            faces.loc[loc, 'is_child'] = 1
        return faces