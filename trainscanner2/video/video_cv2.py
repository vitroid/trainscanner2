#!/usr/bin/env python3

"""
Wrapper for video systems

It does not fit the iterator framework.
"""

import cv2


class VideoLoader(object):
    # In TrainScanner, the video frame starts from 0
    def __init__(self, filename: str, every: int = 1):
        self.cap = cv2.VideoCapture(filename)
        self._head = 0
        # 1 is the first frame
        self.every = every

    def next(self):
        ret, frame = self.cap.read()
        if ret == False:
            return None
        self._head += 1
        for i in range(self.every - 1):
            self.skip()
        return frame

    def skip(self):
        ret = self.cap.grab()
        self._head += 1
        if ret == False:
            return None
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
