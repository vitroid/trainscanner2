import cv2
import numpy as np
import sys
from scipy.signal import find_peaks
from logging import getLogger, DEBUG, basicConfig
import json

# from sklearn.mixture import GaussianMixture
import matplotlib as mpl
from antishake import AntiShaker2
from trainscanner.image import match, linear_alpha, standardize
from tiffeditor import Rect
from trainscanner.video import video_loader_factory
from trainscanner.image import MatchScore
from dataclasses import dataclass

# Stitchは問題なく動いているので、速度予測の精度を上げることにもうちょっと注力する。
# 今の考え方だけだと、もともとの画像が周期的だった場合に、それを速度と勘違いしてしまう。
# 背景画像から背景画像を引いてしまうのはどうか。
#
# あと、透視図に対応していない。これのためには、以前試したような、ブロック分割が良いとおもうが、まだ不完全。
#
# Rectを使うためだけにtiffeditorを読むのは不便。


@dataclass
class FramePosition:
    index: int
    train_velocity: tuple[float, float]
    absolute_location: tuple[float, float]  # of the frame


def find_2d_peaks(scores, num_peaks=4, min_distance=5):
    """
    2次元配列から高い順にピーク位置を検出する

    Args:
        scores: 2次元のスコア配列
        num_peaks: 検出するピーク数（デフォルト4）
        min_distance: ピーク間の最小距離

    Returns:
        peaks: [(y, x), ...] のリスト（高い順）
    """
    # 方法1: 各行でピークを検出
    all_peaks = []

    for i, row in enumerate(scores):
        # 1次元のピーク検出
        peaks_x, properties = find_peaks(
            row, height=np.mean(scores), distance=min_distance
        )
        heights = properties["peak_heights"]

        # 各ピークの座標と高さを記録
        for x, height in zip(peaks_x, heights):
            all_peaks.append((height, i, x))  # (高さ, y座標, x座標)

    # 各列でもピークを検出
    for j in range(scores.shape[1]):
        col = scores[:, j]
        peaks_y, properties = find_peaks(
            col, height=np.mean(scores), distance=min_distance
        )
        heights = properties["peak_heights"]

        # 各ピークの座標と高さを記録
        for y, height in zip(peaks_y, heights):
            all_peaks.append((height, y, j))  # (高さ, y座標, x座標)

    # 重複を除去（近い位置のピークを統合）
    unique_peaks = []
    for height, y, x in sorted(all_peaks, reverse=True):
        is_duplicate = False
        for _, uy, ux in unique_peaks:
            if abs(y - uy) <= min_distance and abs(x - ux) <= min_distance:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_peaks.append((height, y, x))

    # 高い順にソートして上位num_peaks個を取得
    top_peaks = [(y, x) for _, y, x in unique_peaks[:num_peaks]]

    return top_peaks


# 残像マスクはけっこううまくいくみたい。
# 昼間の撮影ならこれだけでほとんど解決するかも。
# もうすこしadaptiveにしたい。動く部分と動かない部分の判定とか。
# gmmで分ける? うまくいっているかどうかわからん。
# maskは、動く部分をうまく抽出しているので、本来なら極大をさがすだけで速度推定できる。
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


def peak_suppression(scores, center=(100, 100)):  # xy
    points = set()

    def explore(p):
        myscore = scores[p[1], p[0]]
        points.add(p)
        for nei in (
            (p[0] - 1, p[1]),
            (p[0] + 1, p[1]),
            (p[0], p[1] - 1),
            (p[0], p[1] + 1),
        ):
            if (
                nei[0] < 0
                or nei[0] >= scores.shape[1]
                or nei[1] < 0
                or nei[1] >= scores.shape[0]
            ):
                continue
            if nei not in points and scores[nei[1], nei[0]] < myscore:
                explore(nei)

    explore(center)
    smallest = min(scores[p[1], p[0]] for p in points)
    print(f"smallest {smallest}")
    # if smallest is not nan
    if not np.isnan(smallest):
        for p in points:
            scores[p[1], p[0]] = smallest


colors = ["navy", "turquoise"]  # , "darkorange"]


def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")


def intersection(a, b):
    return (max(a[0], b[0]), min(a[1], b[1]))


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def alpha_mask(size, delta, width=20):
    # 実際にはalphaを傾ける必要はなかった。
    # 画像のほうを回すべきだった。
    w, h = size
    dx, dy = delta
    L = (dx**2 + dy**2) ** 0.5
    dx /= L
    dy /= L
    # 原点を通る平面。z = A x + B y
    # (0、0、0)と(dx, dy, 1/w)を通るようにしたい。ただし1=dx^2+dy^2
    # dx/A + dy/B = L/w
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    alpha = ((X - w / 2) * dx + (Y - h / 2) * dy) / width
    print(alpha[int(dy * width) + h // 2, int(dx * width) + w // 2])
    alpha[alpha > 1] = 1
    alpha[alpha < 0] = 0
    return alpha


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


def rotated_placement(frame, sine, cosine, train_position):
    h, w = frame.shape[:2]
    rh = int(abs(h * cosine) + abs(w * sine))
    rw = int(abs(h * sine) + abs(w * cosine))
    halfw, halfh = w / 2, h / 2
    R = np.matrix(
        (
            (cosine, sine, -cosine * halfw - sine * halfh + rw / 2),
            (-sine, cosine, sine * halfw - cosine * halfh + rh / 2),
        )
    )
    alpha = linear_alpha(img_width=rw, mixing_width=20, slit_pos=0, head_right=True)
    rotated = cv2.warpAffine(frame, R, (rw, rh))
    # 画像中心をそろえる
    canvas.put_image(
        (int(train_position) - rw // 2, -rh // 2),
        rotated,
        linear_alpha=alpha,
    )


def analyze_iter(vl, scaling_ratio=1.0):
    logger = getLogger(__name__)

    blurmask = BlurMask(lifetime=20)
    antishaker = AntiShaker2(velocity=1)
    # # 背景のずれ
    # framepositions = {}

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
            antimask = np.ones_like(mask)
        else:
            antimask = 1 - normalize(mask)

        unblurred_scaled_frame, delta, abs_loc = antishaker.add_frame(
            scaled_frame, antimask
        )

        # framesにはてぶれを修正し,最初のフレームの位置に背景がそろえられた画像が入るので、あとの処理は列車の動きだけ考えればいい。
        unblurred_scaled_frames.append(unblurred_scaled_frame)
        unblurred_scaled_frame_history.append(unblurred_scaled_frame)

        logger.info(f"{frame_index=} {delta=} {abs_loc=}")
        # framepositions[frame_index] = FramePosition(
        #     index=frame_index, train_velocity=None, absolute_location=abs_loc
        # )

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

        diff = (antimasked_std_log_gray_base - antimasked_std_log_gray_next) ** 2
        mask = blurmask.add_frame(diff)

        # maskは、diffの値が大きいピクセル。
        logger.info(f"mask {np.min(mask)}, {np.max(mask)}")
        mask += np.min(mask)

        # 平均背景をさしひいて、前景を強調する。
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
        matchscore = match(
            base_masked_extended, base_extended_rect, next_masked, next_rect
        )

        # video frame index, absolute location of the frame, matchscore
        yield frame_index, abs_loc, matchscore


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


def main():
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
