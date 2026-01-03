#!/usr/bin/env python3

"""
Wrapper for video systems

"""

import cv2
import os
import numpy as np


class VideoLoader(object):
    def __init__(self, filename, duplicate_threshold: float = 0.1):
        self.dirname = filename
        self._file_index = 0
        self._head = 0
        self.last_frame = None
        self.last_sig = None
        self.duplicate_threshold = duplicate_threshold
        # ディレクトリ内のファイルをソートしておく
        self.filenames = sorted(
            [
                f"{self.dirname}/{f}"
                for f in os.listdir(self.dirname)
                if f.endswith(".png")
            ]
        )

    def _read_raw_next(self):
        """重複を飛ばして次のユニークな画像を読み込む内部メソッド"""
        while self._file_index < len(self.filenames):
            filename = self.filenames[self._file_index]
            self._file_index += 1
            frame = cv2.imread(filename)
            if frame is None:
                continue

            # 縮小・グレースケール化・平滑化により署名を作成
            small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            if self.last_sig is not None and blurred.shape == self.last_sig.shape:
                mean_diff = np.mean(cv2.absdiff(blurred, self.last_sig))
                if mean_diff < self.duplicate_threshold:
                    continue

            self.last_sig = blurred
            self.last_frame = frame
            return frame
        return None

    def next(self):
        frame = self._read_raw_next()
        if frame is not None:
            self._head += 1
        return frame

    def skip(self):
        frame = self._read_raw_next()
        if frame is not None:
            self._head += 1
        return self._head

    def total_frames(self):
        # ユニークなフレーム数は読み込まないと分からないため、ファイル数を最大値として返す
        return len(self.filenames)

    def seek(self, frame_idx):
        # ユニークなフレームに基づくシークは難しいため、
        # 最初から指定のユニークフレーム数分読み飛ばすことで擬似的に実装
        if frame_idx < self._head:
            # 前には戻れない仕様（必要なら再初期化が必要）
            # ここでは簡単のため、最初から読み直すかエラーにする
            self._file_index = 0
            self._head = 0
            self.last_frame = None

        while self._head < frame_idx:
            if self.skip() is None:
                break
        return self._head

    @property
    def head(self):
        return self._head
