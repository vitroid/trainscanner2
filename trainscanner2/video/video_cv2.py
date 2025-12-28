#!/usr/bin/env python3

"""
Wrapper for video systems

It does not fit the iterator framework.
"""

import cv2
import numpy as np


class VideoLoader(object):
    # In TrainScanner, the video frame starts from 0
    def __init__(self, filename: str, every: int = 1, duplicate_threshold: float = 0.1):
        self.cap = cv2.VideoCapture(filename)
        self._head = 0
        # 1 is the first frame
        self.every = every
        self.last_frame = None
        self.duplicate_threshold = duplicate_threshold

    def _read_raw_next(self):
        """重複を飛ばして次のユニークなフレームを読み込む内部メソッド"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                return False, None

            # 前のフレームとほぼ同じかチェック（圧縮ノイズを考慮）
            if self.last_frame is not None and frame.shape == self.last_frame.shape:
                # 差分の絶対値の平均を計算（全ピクセル・全チャンネルの平均）
                mean_diff = np.mean(cv2.absdiff(frame, self.last_frame))

                if mean_diff < self.duplicate_threshold:
                    continue  # 重複（またはほぼ静止）なので次を試す

            self.last_frame = frame
            return True, frame

    def next(self):
        ret, frame = self._read_raw_next()
        if not ret:
            return None
        self._head += 1
        for _ in range(self.every - 1):
            self.skip()
        return frame

    def skip(self):
        # 重複チェックのために read() を使う必要がある
        ret, _ = self._read_raw_next()
        if not ret:
            return None
        self._head += 1
        return self._head

    def total_frames(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) // self.every

    def seek(self, frame):
        if frame < self._head // self.every:
            assert False
        while frame != self._head // self.every:
            self.skip()
        return frame

    @property
    def head(self):
        return self._head // self.every


if __name__ == "__main__":
    vl = VideoLoader("../examples/sample3.mov")

    while True:
        nframe, frame = vl.next()
        if nframe == 0:
            break
        print(frame.shape, nframe)
