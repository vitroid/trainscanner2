import cv2
import numpy as np
import sys
from logging import getLogger, DEBUG, basicConfig, INFO
import json

# from sklearn.mixture import GaussianMixture
from pyperbox import Rect
from trainscanner2.image import match_rect, MatchRect, ImageRect, standardize
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


def get_phase_correlation_score_map(img1, img2, window=1.0):
    """
    2つの画像間の位相限定相関行列を計算する。
    """
    s1 = standardize(img1) * window
    s2 = standardize(img2) * window

    f1 = np.fft.fft2(s1)
    f2 = np.fft.fft2(s2)

    # Cross-power spectrum
    # f1 に対して f2 がどれだけ動いたか
    cross_power = f1 * np.conj(f2)
    normalized_cross_power = cross_power / (np.abs(cross_power) + 1e-15)

    score_map = np.abs(np.fft.ifft2(normalized_cross_power))
    score_map = np.fft.fftshift(score_map)

    return score_map


def analyze_iter(vl, scaling_ratio=1.0, antishaker=None):
    """
    動画を読み込んで、各フレームをずらして自分自身と重ねあわせ、そのスコア(2次元行列)を返す。
    """
    logger = getLogger(__name__)

    # 最初のフレームを読み、スケールして保管する。
    raw_frame = vl.next()
    scaled_frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
    height, width = scaled_frame.shape[:2]
    scaled_gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
    # OpenCVの標準的な窓関数を使用
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    last_gray_frame = scaled_gray_frame
    # 浮動小数点で累積しないと、微小な手振れでドリフトします
    origin_x = 0.0
    origin_y = 0.0
    # fftshift後の中心座標
    center_x, center_y = width // 2, height // 2

    while True:
        frame_index = vl.head
        raw_frame = vl.next()
        if raw_frame is None:
            break

        scaled_frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
        scaled_gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        del raw_frame

        # てぶれ補正のためには厳密な直流成分管理が必要。
        scores0 = get_phase_correlation_score_map(last_gray_frame, scaled_gray_frame)

        # 位相限定相関を使ってスコアマップを計算
        scores = get_phase_correlation_score_map(
            last_gray_frame, scaled_gray_frame, window
        )

        # center 3x3
        shake = 3
        center = scores0[
            center_y - shake : center_y + shake + 1,
            center_x - shake : center_x + shake + 1,
        ]
        _, _, _, max_loc = cv2.minMaxLoc(center)
        peak_x = max_loc[0] - shake
        peak_y = max_loc[1] - shake
        print(f"{peak_x=} {peak_y=}")
        # sys.exit(0)
        # 例えばpeakが(x,y)=(-1,0)だったとしよう。
        # 新フレームの原点は旧フレームより(-1,0)だけ移動した。
        origin_x += peak_x
        origin_y += peak_y
        # 極大が原点になるようにscoreをずらす。(x,y)の順。
        scores = np.roll(scores, (-peak_x, -peak_y), axis=(1, 0))
        matchrect = MatchRect(
            value=scores * 5,
            rect=Rect.from_bounds(
                left=-center_x,
                right=width - center_x,
                top=-center_y,
                bottom=height - center_y,
            ),
        )

        # np.roll の代わりに cv2.warpAffine を使用（サブピクセル精度、端の適切な処理）
        M = np.float32([[1, 0, -origin_x], [0, 1, -origin_y]])
        unblurred_frame = cv2.warpAffine(scaled_frame, M, (width, height))

        # 次のフレームとの比較のためにグレースケール画像を保存
        last_gray_frame = scaled_gray_frame

        # 返すもの。
        # 1. frame_index
        # 2. 新しいフレームの補正前の左上の絶対座標
        # 3. 照合スコア
        # 4. てぶれを補正した(つまり、旧フレームに位置をあわせた)縮小画像

        yield frame_index, (origin_x, origin_y), matchrect, unblurred_frame


def main0():

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


def main():
    # てぶれ補正のみをimshowで確認する。
    basicConfig(level=DEBUG)
    if len(sys.argv) < 2:
        videofile = "../TrainScanner/examples/sample3.mov"
    else:
        videofile = sys.argv[1]
    vl = video_loader_factory(videofile)
    frame = vl.next()
    scale = (300 * 300 / (frame.shape[0] * frame.shape[1])) ** 0.5
    if scale > 1.0:
        scale = 1.0
    vl = video_loader_factory(videofile)
    for frame_index, absolute_position, matchrect, unblurred_frame in analyze_iter(
        vl, scaling_ratio=scale
    ):
        cv2.imshow("unblurred_frame", unblurred_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
