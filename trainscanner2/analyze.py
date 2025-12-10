import cv2
import numpy as np
import sys
from logging import getLogger, DEBUG, basicConfig, INFO
import json

# from sklearn.mixture import GaussianMixture
from pyperbox import Rect
from trainscanner.image import match_rect, MatchRect, ImageRect
from trainscanner.video import video_loader_factory
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
        # print(f"{diff.shape=}, {decayed.shape=}")
        # pull up
        self.mask = cv2.max(diff, decayed)

        cv2.imshow("diff", diff)
        cv2.imshow("mask", self.mask / np.max(self.mask))
        self.logger.debug("----------")
        cv2.waitKey(0 if self.logger.getEffectiveLevel() == DEBUG else 1)
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
        # print(f"{diff.shape=}, {decayed.shape=}")
        # pull up
        self.mask = cv2.max(diff, decayed)

        cv2.imshow("diff", diff)
        cv2.imshow("mask", self.mask / np.max(self.mask))
        self.logger.debug("----------")
        cv2.waitKey(0 if self.logger.getEffectiveLevel() == DEBUG else 1)
        return self.mask


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def analyze_iter(vl, scaling_ratio=1.0):
    """
    動画を読み込んで、各フレームをずらして自分自身と重ねあわせ、そのスコア(2次元行列)を返す。
    """
    logger = getLogger(__name__)

    # diff画像をたくわえ、動きの大きい領域を検出する。
    blurmask = BlurMask3()

    # 背景の移動をもとにてぶれを検出し、最初のフレームの位置から視野が流れていかないようにする。
    antishaker = AntiShaker2(velocity=1)

    # 最初のフレームを読み、スケールして保管する。
    raw_frame = vl.next()
    raw_frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
    unblurred_scaled_frames = FIFO(2)
    estimate = 5
    unblurred_scaled_frame_history = FIFO(estimate)
    unblurred_scaled_frames.append(raw_frame)
    unblurred_scaled_frame_history.append(raw_frame)

    mask = np.ones(unblurred_scaled_frames.queue[0].shape[:2], dtype=np.float32)

    while True:
        frame_index = vl.head
        raw_frame = vl.next()
        if raw_frame is None:
            break

        scaled_frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
        del raw_frame

        height, width = scaled_frame.shape[:2]

        # antimask = np.exp(-mask)
        if np.max(mask) == np.min(mask):
            # 全部1にする。
            antimask = np.ones_like(mask)
        else:
            # normalizeは値の範囲を0〜1間におさめる
            antimask = 1 - normalize(mask)

        # 直前のフレームからの変位deltaを測定し、積算してフレームごとの絶対位置abs_locを求める。
        # unblurred_scaled_frameは位置あわせしたあとのフレーム。以後の処理はこれを基準とする。
        unblurred_scaled_frame, delta, abs_loc = antishaker.add_frame(
            scaled_frame, antimask
        )

        # unblurred_scaled_framesにはてぶれを修正し,最初のフレームの位置に背景がそろえられた画像が入る。
        unblurred_scaled_frames.append(unblurred_scaled_frame)
        unblurred_scaled_frame_history.append(unblurred_scaled_frame)

        logger.debug(f"{frame_index=} {delta=} {abs_loc=}")

        # 平均画像=背景
        averaged_background = np.zeros_like(
            unblurred_scaled_frames.queue[0], dtype=np.float32
        )
        for fh in unblurred_scaled_frame_history.queue:
            averaged_background += fh
        averaged_background /= len(unblurred_scaled_frame_history.queue)

        hdr_avg = std_hdr(averaged_background)
        # グレースケールに変換
        base_frame = unblurred_scaled_frames.queue[0]
        next_frame = unblurred_scaled_frames.queue[1]

        antimasked_hdr_base = std_hdr(base_frame) * antimask
        antimasked_hdr_next = std_hdr(next_frame) * antimask

        # 二乗差分画像を作る
        # diff = (antimasked_hdr_base - antimasked_hdr_next) ** 2
        # blurmaskに追加する。maskは平均化されたマスク
        # mask = blurmask.add_frame(diff)
        mask = blurmask.add_frame(antimasked_hdr_next)

        # maskは、diffの値が大きいピクセル。
        logger.debug(f"mask {np.min(mask)}, {np.max(mask)}")
        mask -= np.min(mask)

        # 平均背景をさしひいて、前景を強調する。
        # 今はマスクを使っていない。
        base_masked = antimasked_hdr_base.copy() - hdr_avg  # * mask
        next_masked = antimasked_hdr_next.copy() - hdr_avg  # * mask

        # こんどは移動量をたっぷりとる。
        max_shift = 100

        base_masked_extended = np.zeros(
            [height + 2 * max_shift, width + 2 * max_shift],
            dtype=np.float32,
        )
        base_masked_extended[max_shift:-max_shift, max_shift:-max_shift] = base_masked

        base_imagerect = ImageRect(
            image=base_masked_extended,
            lefttop=(-max_shift, -max_shift),
        )
        next_imagerect = ImageRect(image=next_masked, lefttop=(0, 0))
        # scoreとは、2つの画像のピクセル内積。1に近いほど画像が似ている=よく重なる。
        # match_rectはrect付き行列。
        matchrect = match_rect(base_imagerect, next_imagerect)

        yield frame_index, abs_loc, matchrect, unblurred_scaled_frame


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
