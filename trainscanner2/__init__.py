from dataclasses import dataclass
import cv2
import numpy as np

from trainscanner2.image import standardize


@dataclass
class PathItem:
    frame_index: int
    xy: tuple[float, float]
    value: list
    hop: int = 1


def std_hdr(image):
    # 既にグレースケールの場合は変換をスキップ
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Invalid image shape: {image.shape}")
    return standardize(np.log(gray.astype(np.float32) + 1))
