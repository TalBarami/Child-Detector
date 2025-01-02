import numpy as np
import pandas as pd

class DetectionsData:
    def __init__(self, detections, confidence_threshold=0.6, duplication_threshold=0.9, brief_threshold=25):
        self._detections = detections
        self.confidence_threshold = confidence_threshold
        self.duplication_threshold = duplication_threshold
        self.brief_threshold = brief_threshold

        self._detections_processed = self._process()
        self.F, self.M = self.detections.attrs['n_frames'], self.detections['frame_offset'].max()+1
        self._numpy = self._create_numpy()
        self._brief_segments = None


    @property
    def xywh(self):
        return self._numpy[:, :, [0, 1, 2, 3]]

    @property
    def xyxy(self):
        return self._numpy[:, :, [4, 5, 6, 7]]

    @property
    def diag(self):
        return self._numpy[:, :, 8]

    @property
    def confidence(self):
        return self._numpy[:, :, [9, 10, 11]]

    @property
    def labels(self):
        return self._numpy[:, :, 12]

    @property
    def detections(self):
        return self._detections_processed

    @property
    def child(self):
        return self.detections[self.detections['label'] == 1]

    @property
    def adults(self):
        return self.detections[self.detections['label'] == 0]

    @property
    def brief_segments(self):
        if self._brief_segments is None:
            self._brief_segments = self._detect_brief_segments()
        return self._brief_segments


    def save(self, detections_path):
        self._detections.to_hdf(detections_path, key='detections', mode='w')
        with pd.HDFStore(detections_path) as store:
            store.get_storer('detections').attrs.metadata = self._detections.attrs

    @staticmethod
    def load(detections_path, confidence_threshold=0.6, duplication_threshold=0.03, brief_threshold=25):
        df = pd.read_hdf(detections_path)
        with pd.HDFStore(detections_path) as store:
            metadata = store.get_storer("detections").attrs.metadata
            df.attrs = metadata
        return DetectionsData(df, confidence_threshold, duplication_threshold, brief_threshold)

    # def remove_duplicates(self, df):
    #     def filter_duplicates(group):
    #         x_diff = np.abs(group['x'].values[:, None] - group['x'].values)
    #         y_diff = np.abs(group['y'].values[:, None] - group['y'].values)
    #         diag = group['diag'].values
    #         dynamic_threshold = diag[:, None] + diag[None, :]
    #         dynamic_threshold *= self.duplication_threshold
    #         is_duplicate = (x_diff < dynamic_threshold) & (y_diff < dynamic_threshold)
    #         np.fill_diagonal(is_duplicate, False)
    #         to_remove = set()
    #         for i in range(len(group)):
    #             if i in to_remove:
    #                 continue
    #             duplicates = np.where(is_duplicate[i])[0]
    #             if len(duplicates) > 0:
    #                 indices = [i] + duplicates.tolist()
    #                 indices = [group.index[idx] for idx in indices]
    #                 max_idx = group.loc[indices]['confidence'].idxmax()
    #                 to_remove.update(idx for idx in indices if idx != max_idx)
    #         return to_remove
    #
    #     remove_indices = set()
    #     for _, group in df.groupby('frame'):
    #         remove_indices.update(filter_duplicates(group))
    #     return df.drop(remove_indices).reset_index(drop=True)

    def _detect_brief_segments(self):
        df = self.detections
        frame_summary = df.groupby('frame').agg(child_detected=('label', 'any')).reset_index()
        frame_summary['segment'] = (frame_summary['child_detected'] != frame_summary['child_detected'].shift()).cumsum()
        segments = frame_summary.groupby('segment').agg(
            start_frame=('frame', 'first'),
            end_frame=('frame', 'last'),
            child_detected=('child_detected', 'first'),
            length=('child_detected', 'size')
        ).reset_index(drop=True)
        threshold = self.brief_threshold
        context_frames = 1
        brief_segments = []
        for idx, row in segments.iterrows():
            if row['length'] <= threshold:
                if idx > 0 and idx < len(segments) - 1:
                    prev_segment = segments.iloc[idx - 1]
                    next_segment = segments.iloc[idx + 1]
                    if (prev_segment['child_detected'] != row['child_detected'] and
                            next_segment['child_detected'] != row['child_detected'] and
                            prev_segment['length'] >= context_frames and
                            next_segment['length'] >= context_frames):
                        brief_segments.append(row)
        brief_segments = pd.DataFrame(brief_segments)
        return brief_segments

    def _create_numpy(self):
        cols = ['x', 'y', 'w', 'h', 'x1', 'y1', 'x2', 'y2', 'diag', 'confidence_adult', 'confidence_child', 'confidence', 'label']
        unique_frames = self.detections['frame'].unique()
        frame_indices = pd.Categorical(self.detections['frame'], categories=unique_frames).codes
        person_indices = self.detections['frame_offset'].values

        arr = np.empty((self.F, self.M, len(cols)))
        vals = self.detections[cols].to_numpy()
        arr[frame_indices, person_indices, :] = vals
        return arr

    def _process(self):
        detections = self._detections.copy().reset_index(drop=True)
        detections[['x', 'y', 'w', 'h']] = detections[['x', 'y', 'w', 'h']].astype(int)
        detections[['x1', 'y1', 'x2', 'y2']] = (detections[['x', 'y', 'x', 'y']] + (detections[['w', 'h', 'w', 'h']].values / 2 * [-1, -1, 1, 1])).values.astype(int)
        detections['diag'] = np.sqrt(detections['w'] ** 2 + detections['h'] **2)
        detections['confidence'] = (detections['confidence_child'] - detections['confidence_adult'] + 1) / 2
        # detections = self.remove_duplicates(detections)
        detections['frame_offset'] = detections.groupby('frame').cumcount()
        detections.loc[detections.dropna().index, 'label'] = 0
        valid = detections[detections['confidence'] >= self.confidence_threshold]
        max_indices = valid.groupby('frame')['confidence'].idxmax()
        detections.loc[max_indices, 'label'] = 1
        detections['label'] = detections['label'].astype(int)
        return detections
