import cv2
import numpy as np
import sys
from trainscanner2.image import standardize


def shiftx(frame, dx):
    return np.roll(frame, dx, axis=1)


def shifty(frame, dy):
    return np.roll(frame, dy, axis=0)


def shift(frame, dx, dy):
    return shiftx(shifty(frame, dy), dx)


class AntiShaker2:
    # 直前のフレームとの差を返しつつ、変位の総量(最初のフレームからの累積変位)を記憶しておく。
    def __init__(self, velocity=1):
        self._absx = 0
        self._absy = 0
        self._last_frame = None
        self._velocity = velocity

    @property
    def abs_loc(self):
        return self._absx, self._absy

    @abs_loc.setter
    def abs_loc(self, value):
        self._absx, self._absy = value

    def add_frame(self, frame, mask=None):
        # subpixel matchingは要らないだろう。
        h, w = frame.shape[:2]
        frame_std = standardize(frame)

        if self._last_frame is None or self._velocity == 0:
            self._last_frame = frame_std.copy()
            return frame, (0, 0), (0, 0)

        frame0_extend = np.zeros(
            (h + 2 * self._velocity, w + 2 * self._velocity), dtype=np.float32
        )
        frame0_extend[
            self._velocity : -self._velocity, self._velocity : -self._velocity
        ] = self._last_frame
        # frame_std と mask を明示的にfloat32に変換してから掛け算
        if mask is None:
            template = frame_std.astype(np.float32)
        else:
            template = np.multiply(
                frame_std.astype(np.float32), mask.astype(np.float32)
            )
        # antishakeでは、整数未満の変位は無視する。精密照合はdetectにまかせる。
        scores = cv2.matchTemplate(frame0_extend, template, cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(scores)
        dx, dy = (max_loc[0] - self._velocity, max_loc[1] - self._velocity)
        self._absx += dx
        self._absy += dy
        diff_img = self._last_frame.copy()
        shifted_frame = shift(frame_std, dx, dy)
        # cv2.imshow("diff", diff_img - shifted_frame)
        self._last_frame = frame_std.copy()
        return shift(frame, self._absx, self._absy), (dx, dy), (self._absx, self._absy)


from trainscanner2.image import ImageRect, match_rect_expanded


class AntiShaker3:
    # 直前のフレームとの差を返しつつ、変位の総量(最初のフレームからの累積変位)を記憶しておく。
    def __init__(self, velocity=1, reference_frame=None):
        self._velocity = velocity
        if reference_frame is not None:
            self.reset(reference_frame)
        else:
            self._reference_imagerect = None

    def reset(self, frame):
        self._last_driftx = 0
        self._last_drifty = 0
        self._reference_imagerect = ImageRect(image=standardize(frame))

    def add_frame(self, frame, mask=None):
        if self._reference_imagerect is None:
            self.reset(frame)
        # subpixel matchingは要らないだろう。
        h, w = frame.shape[:2]
        frame_std = standardize(frame)
        frame_imagerect = ImageRect(
            image=shift(frame_std, self._last_driftx, self._last_drifty)
        )
        matchrect = match_rect_expanded(
            self._reference_imagerect, frame_imagerect, self._velocity
        )
        matchrect.validate()
        (driftx, drifty), _ = matchrect.peak(subpixel=False)
        # print(dx, dy)

        self._last_driftx += driftx
        self._last_drifty += drifty

        return (
            shift(frame, self._last_driftx, self._last_drifty),
            (self._last_driftx, self._last_drifty),
        )


if __name__ == "__main__":
    antishaker = AntiShaker3(velocity=10)
    # 動画を読み込む
    if len(sys.argv) < 2:
        videofile = "examples/sample3.mov"
        videofile = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/他人の動画/antishake test/Untitled.mp4"
        videofile = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/Czech Trams/00205/00205.MTS"
    else:
        videofile = sys.argv[1]
    cap = cv2.VideoCapture(videofile)
    _, frame = cap.read()
    height, width = frame.shape[:2]
    scaling_ratio = (512 * 512 / (width * height)) ** 0.5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
        frame, abs_loc = antishaker.add_frame(frame)
        print(abs_loc)
        cv2.imshow("frame", frame)
        cv2.imshow("diff", standardize(frame) - antishaker._reference_imagerect.image)
        cv2.waitKey(1)
