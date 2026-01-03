#!/usr/bin/env python3

"""
Wrapper for video systems

It does not fit the iterator framework.
"""

import cv2
import numpy as np


class VideoLoader(object):
    # In TrainScanner, the video frame starts from 0
    def __init__(self, filename: str, every: int = 1, duplicate_threshold: float = 0.5):
        self.cap = cv2.VideoCapture(filename)
        self._head = 0
        # 1 is the first frame
        self.every = every
        self.last_frame = None
        self.last_sig = None
        self.duplicate_threshold = duplicate_threshold

    def _read_raw_next(self):
        """重複を飛ばして次のユニークなフレームを読み込む内部メソッド"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                return False, None

            # 縮小・グレースケール化・平滑化により、圧縮ノイズ（ブロックノイズ等）の影響を排除した署名を作成
            # 64x64程度の解像度があれば、構造的な変化は維持しつつ高周波のブロックノイズを無効化できる
            small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            if self.last_sig is not None and blurred.shape == self.last_sig.shape:
                # 署名間の平均絶対誤差 (MAE) を計算
                mean_diff = np.mean(cv2.absdiff(blurred, self.last_sig))

                # YouTubeの重複フレームの場合、ノイズがあっても MAE は非常に小さくなる。
                # 経験的に、平滑化された64x64画像では 0.1〜0.5 程度が適切な重複しきい値。
                if mean_diff < self.duplicate_threshold:
                    # print(f"duplicate frame: {mean_diff}")
                    continue  # 重複（またはほぼ静止）なので次を試す

            self.last_sig = blurred
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
