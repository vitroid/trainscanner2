from dataclasses import dataclass
import cv2
import numpy as np

from trainscanner2.antishake import standardize


@dataclass
class PathItem:
    frame_index: int
    xy: tuple[float, float]
    value: list


class FIFO:
    def __init__(self, maxlen: int):
        self.queue = []
        self.maxlen = maxlen

    def append(self, item):
        self.queue.append(item)
        if len(self.queue) > self.maxlen:
            self.queue.pop(0)

    def fluctuation(self):
        return max(self.queue) - min(self.queue)

    @property
    def length(self):
        return len(self.queue)

    @property
    def filled(self):
        return len(self.queue) == self.maxlen


def std_hdr(image):
    # 既にグレースケールの場合は変換をスキップ
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Invalid image shape: {image.shape}")
    return standardize(
        np.log(gray.astype(np.float32) + 1)
    )
