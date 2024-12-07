class DetectionsData:
    def __init__(self, detections_raw, detections_processed):
        self.detections_raw = detections_raw
        self.detections_processed = detections_processed

    def save(self, detections_path):
        write_pkl(self, detections_path)

    @staticmethod
    def load(detections_path):
        return read_pkl(detections_path)
