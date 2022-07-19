from torch.utils.data import Dataset
import cv2

class ChildDetectionDataset(Dataset):
    def __init__(self, video_path, batch_size):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.i = 0
        self.batch_size = batch_size

    def __len__(self):
        return self.num_frames

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            ret, frame = self.cap.read()
            if ret:
                batch.append(frame)
                self.i += 1
            else:
                if self.num_frames <= self.i:
                    raise StopIteration
                else:
                    raise IndexError(f"Unable to read frame.")
        return batch

    def __del__(self):
        self.cap.release()
