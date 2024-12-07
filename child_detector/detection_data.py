from taltools.io.files import read_pkl, write_pkl

class DetectionsData:
    def __init__(self, detections, duplication_threshold=0.9):
        self._detections = detections
        self._detections_processed = None

    @property
    def detections(self):
        if self._detections_processed is None:
            self._detections_processed = self._process(self._detections)
        return self._detections_processed

    def save(self, detections_path):
        write_pkl(self, detections_path)

    @staticmethod
    def load(detections_path):
        return read_pkl(detections_path)

    def _process(self, detections):
        if type(detections) is str:
            detections = read_pkl(out_path)
        dfs = []
        for frame, df in detections:
            df['confidence'] = (df['confidence_child'] - df['confidence_adult'] + 1) / 2
            to_remove = set()
            if df.empty:
                df = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
            elif len(df) > 1:
                boxes = xywh2xyxy(df[['x', 'y', 'w', 'h']].values)
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        boxes_iou = iou(boxes[i], boxes[j])
                        if boxes_iou > self.duplication_threshold:
                            to_remove.add(j if df['confidence'].iloc[i] > df['confidence'].iloc[j] else i)
            df['frame'] = frame
            # if frame > n:
            #     _n = n // 2
            #     prev, curr, next = [d.dropna() for d in dfs[-n:-n//2]], dfs[-n//2].dropna(), [d.dropna() for d in dfs[-n//2+1:]]
            #     prev, next = [d for d in prev if not d.empty], [d for d in next if not d.empty]
            #     if not curr.empty and (len(prev) > 0 or len(next) > 0):
            #         curr = ChildDetector.temporal_consistency(curr, prev, next)
            #         dfs[-n//2] = curr

            dfs.append(df.drop(list(to_remove)).reset_index(drop=True))
        df = pd.concat(dfs).reset_index(drop=True)
        return df