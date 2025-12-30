import cv2
import numpy as np
import sys
from logging import getLogger, DEBUG, basicConfig, INFO
import json

# from sklearn.mixture import GaussianMixture
from pyperbox import Rect
from trainscanner2.image import match_rect, MatchRect, ImageRect
from trainscanner2.video import video_loader_factory
from trainscanner2 import FIFO, std_hdr
from trainscanner2.antishake import AntiShaker2


# フレーム間の二乗差分を時間平均して、動きの大きい部分を抽出する。
class BlurMask:
    logger = getLogger(__name__)

    def __init__(self, lifetime=10):
        self.lifetime = lifetime
        self.masks = []
        self.sumask = None

    def add_frame(self, diff):
        # assert diff does not contain nan

        if np.isnan(diff).any():
            diff = np.zeros_like(diff)

        if self.sumask is None:
            self.sumask = diff
        else:
            self.sumask += diff

        self.masks.append(diff.copy())
        if len(self.masks) > self.lifetime:
            elim = self.masks.pop(0)
            self.sumask -= elim

        assert not np.isnan(self.sumask).any()
        return self.sumask / self.lifetime
        # return np.log(self.sumask + 1)


class BlurMask2:
    logger = getLogger(__name__)

    def __init__(self, lifetime=10):
        self.lifetime = lifetime
        self.mask = None

    def add_frame(self, diff):
        # assert diff does not contain nan

        if np.isnan(diff).any():
            diff = np.zeros_like(diff)
        # 型をfloat32に統一
        diff = diff.astype(np.float32, copy=False)
        if self.mask is None:
            self.mask = np.zeros_like(diff)

        # decay
        decayed = self.mask * (1 - 1 / self.lifetime)
        self.mask = cv2.max(diff, decayed)

        # if self.logger.getEffectiveLevel() <= INFO:
        #     cv2.imshow("diff", diff)
        #     cv2.imshow("mask", self.mask / (np.max(self.mask) + 1e-6))
        #     cv2.waitKey(0 if self.logger.getEffectiveLevel() == DEBUG else 1)
        self.logger.debug("----------")
        return self.mask


class BlurMask3:
    """
    内部で差分計算をするようにする。
    隣接フレームではなく、すこし遠いフレームを比較することで、遅い列車でも検出できるようにする。
    """

    logger = getLogger(__name__)

    def __init__(self, interval=30):
        self.images = FIFO(interval + 1)
        self.mask = None

    def add_frame(self, std_img):
        # assert diff does not contain nan
        self.images.append(std_img)

        diff = (self.images.queue[0] - self.images.queue[-1]) ** 2

        # 型をfloat32に統一
        diff = diff.astype(np.float32, copy=False)
        if self.mask is None:
            self.mask = np.zeros_like(diff)

        # decay
        decayed = self.mask * 0.9
        self.mask = cv2.max(diff, decayed)

        self.logger.debug("----------")
        return diff, self.mask


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def analyze_iter(vl, scaling_ratio=1.0, antishaker=None):
    """
    動画を読み込んで、各フレームをずらして自分自身と重ねあわせ、そのスコア(2次元行列)を返す。
    """
    logger = getLogger(__name__)

    # 背景の移動をもとにてぶれを検出し、最初のフレームの位置から視野が流れていかないようにする。
    if antishaker is None:
        antishaker = AntiShaker2(velocity=1)

    # 最初のフレームを読み、スケールして保管する。
    raw_frame = vl.next()
    scaled_frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
    height, width = scaled_frame.shape[:2]
    scaled_gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
    window = np.hanning(width) * np.hanning(height)[:, np.newaxis]
    scaled_std_frame = normalize(scaled_gray_frame) * window
    last_std_frame = scaled_std_frame
    origin_x = 0
    origin_y = 0
    # mask = np.ones(unblurred_scaled_frames.queue[0].shape[:2], dtype=np.float32)

    while True:
        frame_index = vl.head
        raw_frame = vl.next()
        if raw_frame is None:
            break

        scaled_frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
        scaled_gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        scaled_std_frame = normalize(scaled_gray_frame) * window
        del raw_frame

        # FFTを使い、unblurred_std_framesの2枚の画像をずらして比較する。
        # phase correlationを使う。
        # 周期境界の問題をなくすためにWindow関数をかける。
        fft0 = np.fft.fft2(last_std_frame)
        fft1 = np.fft.fft2(scaled_std_frame)
        # fft1 = np.fft.fft2(np.roll(last_unblurred_std_frame, 1, axis=1))
        scores = np.fft.ifft2(fft0 * np.conj(fft1))
        scores = np.abs(scores)
        scores = normalize(scores)
        scores = np.roll(scores, (width // 2, height // 2), axis=(1, 0))
        # center 3x3
        center = scores[
            height // 2 - 1 : height // 2 + 2, width // 2 - 1 : width // 2 + 2
        ]
        _, _, _, max_loc = cv2.minMaxLoc(center)
        peak_x = max_loc[0] - 1
        peak_y = max_loc[1] - 1
        # print(f"{peak_x=} {peak_y=} {center}")
        # sys.exit(0)
        # 例えばpeakが(x,y)=(-1,0)だったとしよう。
        # 新フレームの原点は旧フレームより(-1,0)だけ移動した。
        origin_x += peak_x
        origin_y += peak_y
        # 極大が原点になるようにscoreをずらす。(x,y)の順。
        scores = np.roll(scores, (-peak_x, -peak_y), axis=(1, 0))
        matchrect = MatchRect(
            value=scores,
            rect=Rect.from_bounds(
                left=-width // 2,
                right=width - width // 2,
                top=-height // 2,
                bottom=height - height // 2,
            ),
        )

        last_std_frame = scaled_std_frame
        unblurred_frame = np.roll(scaled_frame, (origin_x, origin_y), axis=(1, 0))

        # 返すもの。
        # 1. frame_index
        # 2. 新しいフレームの補正前の左上の絶対座標
        # 3. 照合スコア
        # 4. てぶれを補正した(つまり、旧フレームに位置をあわせた)縮小画像

        yield frame_index, (origin_x, origin_y), matchrect, unblurred_frame


def main():

    class Storer:
        # withで使えるようにしたい。
        def __init__(self, filename: str):
            self.filename = filename
            self.matchrects = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            with open(self.filename, "w") as f:
                json.dump(self.matchrects, f, indent=4)

        def append(self, frame_index, absolute_position, matchrect: MatchRect):
            self.matchrects[frame_index] = {}
            self.matchrects[frame_index]["absolute_position"] = absolute_position
            value = matchrect.value
            self.matchrects[frame_index]["value"] = value.tolist()
            rect = matchrect.rect
            self.matchrects[frame_index]["rect"] = [
                rect.left,
                rect.right,
                rect.top,
                rect.bottom,
            ]

    basicConfig(level=DEBUG)
    # 動画を読み込む
    if len(sys.argv) < 2:
        videofile = "../TrainScanner/examples/sample3.mov"
        # videofile = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/他人の動画/antishake test/Untitled.mp4"

    else:
        videofile = sys.argv[1]
    vl = video_loader_factory(videofile)
    frame = vl.next()
    scale = (300 * 300 / (frame.shape[0] * frame.shape[1])) ** 0.5
    if scale > 1.0:
        scale = 1.0

    vl = video_loader_factory(videofile)
    if "0835" in videofile:
        vl.seek(47 * 30)

    with Storer("motions_test.json") as storer:
        for frame_index, absolute_position, matchrect, _ in analyze_iter(
            vl, scaling_ratio=scale
        ):
            storer.append(frame_index, absolute_position, matchrect)


if __name__ == "__main__":
    main()
