#!/usr/bin/env python3

"""
Wrapper for video systems

It does not fit the iterator framework.
"""

import sys
from trainscanner2.video import video_cv2, image_dir
import os


def video_loader_factory(filename: str, every: int = 1):
    filename = filename.rstrip("/")
    if os.path.isdir(filename):
        return image_dir.VideoLoader(filename)
    ostype = sys.platform
    if ostype == "darwin":
        return video_cv2.VideoLoader(filename, every=every)
    elif 0 == ostype.find("linux"):
        return video_cv2.VideoLoader(filename, every=every)
    elif 0 == ostype.find("win"):
        return video_cv2.VideoLoader(filename, every=every)
    else:
        raise ValueError(f"Unsupported platform: {ostype}")


def video_iter(filename: str):
    video_loader = video_loader_factory(filename)
    while True:
        frame = video_loader.next()
        if frame is None:
            break
        yield frame


# assume it is executed as python -m trainscanner.video.__init__
if __name__ == "__main__":
    for frame in video_iter("examples/sample3.mov"):
        print(frame.shape)
