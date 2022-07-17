from torch.utils.data import Dataset
import cv2

class ChildDetectionDataset(Dataset):
    def __init__(self, video_path, skeleton_data):
        self.video_path = video_path
        self.skeleton = skeleton_data
        cap = cv2.VideoCapture(self.video_path)
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # cap = cv2.VideoCapture(self.video_path)
        # self.frames = []
        # ret = True
        # while ret:
        #     ret, frame = cap.read()
        #     if ret:
        #         self.frames.append(frame)
        # cap.release()

    def __len__(self):
        return self.num_frames

    def __getitem__(self, i):
        cap = cv2.VideoCapture(self.video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                raise IndexError(f'Out of bounds for frame {i}')
        finally:
            cap.release()
        return i, frame, self.skeleton['keypoint'][:, i, :], self.skeleton['keypoint_score'][:, i, :]
