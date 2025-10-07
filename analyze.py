import cv2
import numpy as np
import sys
from logging import getLogger, DEBUG, basicConfig
import json

# from sklearn.mixture import GaussianMixture
from antishake import AntiShaker2
from trainscanner.image import match, standardize
from tiffeditor import Rect
from trainscanner.video import video_loader_factory
from trainscanner.image import MatchScore

# from dataclasses import dataclass


# @dataclass
# class FramePosition:
#     index: int
#     train_velocity: tuple[float, float]
#     absolute_location: tuple[float, float]  # of the frame


# 残像マスク


# フレーム間の二乗差分を時間平均して、動きの大きい部分を抽出する。
class BlurMask:
    def __init__(self, lifetime=10):
        self.lifetime = lifetime
        self.masks = []
        self.sumask = None

    def add_frame(self, diff):
        # assert diff does not contain nan
        assert not np.isnan(diff).any()

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


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# def alpha_mask(size, delta, width=20):
#     # 実際にはalphaを傾ける必要はなかった。
#     # 画像のほうを回すべきだった。
#     w, h = size
#     dx, dy = delta
#     L = (dx**2 + dy**2) ** 0.5
#     dx /= L
#     dy /= L
#     # 原点を通る平面。z = A x + B y
#     # (0、0、0)と(dx, dy, 1/w)を通るようにしたい。ただし1=dx^2+dy^2
#     # dx/A + dy/B = L/w
#     X, Y = np.meshgrid(np.arange(w), np.arange(h))
#     alpha = ((X - w / 2) * dx + (Y - h / 2) * dy) / width
#     print(alpha[int(dy * width) + h // 2, int(dx * width) + w // 2])
#     alpha[alpha > 1] = 1
#     alpha[alpha < 0] = 0
#     return alpha


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


def analyze_iter(vl, scaling_ratio=1.0):
    """
    動画を読み込んで、各フレームをずらして自分自身と重ねあわせ、そのスコア(2次元行列)を返す。
    """
    logger = getLogger(__name__)

    # diff画像をたくわえ、動きの大きい領域を検出する。
    blurmask = BlurMask(lifetime=20)

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

        std_log_gray_avg = standardize(
            np.log(
                cv2.cvtColor(averaged_background, cv2.COLOR_BGR2GRAY).astype(np.float32)
                + 1
            )
        )
        # グレースケールに変換
        base_frame = unblurred_scaled_frames.queue[0]
        next_frame = unblurred_scaled_frames.queue[1]

        antimasked_std_log_gray_base = (
            standardize(
                np.log(
                    cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1
                )
            )
            * antimask
        )
        antimasked_std_log_gray_next = (
            standardize(
                np.log(
                    cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1
                )
            )
            * antimask
        )

        # 二乗差分画像を作る
        diff = (antimasked_std_log_gray_base - antimasked_std_log_gray_next) ** 2
        # blurmaskに追加する。maskは平均化されたマスク
        mask = blurmask.add_frame(diff)

        # maskは、diffの値が大きいピクセル。
        logger.debug(f"mask {np.min(mask)}, {np.max(mask)}")
        mask += np.min(mask)

        # 平均背景をさしひいて、前景を強調する。
        # 今はマスクを使っていない。
        base_masked = antimasked_std_log_gray_base.copy() - std_log_gray_avg  # * mask
        next_masked = antimasked_std_log_gray_next.copy() - std_log_gray_avg  # * mask

        # こんどは移動量をたっぷりとる。
        max_shift = 100

        base_masked_extended = np.zeros(
            [height + 2 * max_shift, width + 2 * max_shift],
            dtype=np.float32,
        )
        base_masked_extended[max_shift:-max_shift, max_shift:-max_shift] = base_masked
        base_extended_rect = Rect.from_bounds(
            -max_shift,
            width + max_shift,
            -max_shift,
            height + max_shift,
        )
        next_rect = Rect.from_bounds(
            0,
            width,
            0,
            height,
        )
        # scoreとは、2つの画像のピクセル内積。1に近いほど画像が似ている=よく重なる。
        # matchscoreはtick付き行列。
        matchscore = match(
            base_masked_extended, base_extended_rect, next_masked, next_rect
        )

        # video frame index, absolute location of the frame, matchscore
        yield frame_index, abs_loc, matchscore, unblurred_scaled_frame


def main():

    class Storer:
        # withで使えるようにしたい。
        def __init__(self, filename: str):
            self.filename = filename
            self.matchscores = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            with open(self.filename, "w") as f:
                json.dump(self.matchscores, f)

        def append(self, frame_index, absolute_position, matchscore: MatchScore):
            self.matchscores[frame_index] = {}
            self.matchscores[frame_index]["absolute_position"] = absolute_position
            dx = str(matchscore.dx)
            dy = str(matchscore.dy)
            value = matchscore.value
            self.matchscores[frame_index]["value"] = value.tolist()
            self.matchscores[frame_index]["dx"] = dx
            self.matchscores[frame_index]["dy"] = dy

    basicConfig(level=DEBUG)
    # 動画を読み込む
    if len(sys.argv) < 2:
        videofile = "examples/sample3.mov"
        videofile = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/他人の動画/antishake test/Untitled.mp4"

    else:
        videofile = sys.argv[1]
    vl = video_loader_factory(videofile)
    if "0835" in videofile:
        vl.seek(47 * 30)

    with Storer("motions_test.json") as storer:
        for frame_index, absolute_position, matchscore in analyze_iter(
            vl, scaling_ratio=1.0
        ):
            storer.append(frame_index, absolute_position, matchscore)


if __name__ == "__main__":
    main()
