from torch.utils.data import Dataset
import cv2

class ChildDetectionDataset(Dataset):
    def __init__(self, video_path, skeleton_data):
        self.video_path = video_path
        self.skeleton = skeleton_data
        cap = cv2.VideoCapture(self.video_path)
        self.frames = []
        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                self.frames.append(frame)
        cap.release()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return i, self.frames[i], self.skeleton['keypoint'][:, i, :], self.skeleton['keypoint_score'][:, i, :]
