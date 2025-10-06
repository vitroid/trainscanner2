# motions.jsonは2次元の数値配列の時間変化を含む。(実際にはMatchScore形式のデータクラス)
# これを読みこみ、極大を複数見付けだし、その移動を追跡する。、
# 極大の個数はとりあえず最大で3個。
# 逐次処理できることがわかった。
from logging import getLogger, basicConfig, INFO
import json
from trainscanner.image import MatchScore
import numpy as np
import matplotlib.pyplot as plt
from pyperbox import Rect
import pykalman
from dataclasses import dataclass


def find_peaks(arr: np.ndarray, rect: Rect, height: float = 0.5):
    assert rect.width == arr.shape[1]
    assert rect.height == arr.shape[0]
    centers = arr[1:-1, 1:-1]
    non_max = np.zeros_like(centers, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                cmp = (
                    arr[
                        1 + dy : centers.shape[0] + 1 + dy,
                        1 + dx : centers.shape[1] + 1 + dx,
                    ]
                    > centers
                )

                non_max |= cmp
    is_max = ~non_max
    for y, x in np.argwhere(is_max):
        if arr[y + 1, x + 1] > height:
            yield x - (rect.width - 1) // 2 + 1, y - (rect.height - 1) // 2 + 1, arr[
                y + 1, x + 1
            ]


@dataclass
class HistoryItem:
    xy: tuple[int, int]
    value: list


class Peak:
    def __init__(self, xy, value):
        self.mean = np.array(xy)
        self.covariance = np.eye(2)
        self.kf = pykalman.KalmanFilter(
            transition_matrices=np.eye(2),
            observation_matrices=np.eye(2),
            transition_covariance=np.eye(2),
            observation_covariance=np.eye(2),
            initial_state_mean=self.mean,
            initial_state_covariance=self.covariance,
        )
        self.missed_duration = 0
        self.history = [HistoryItem(xy=xy, value=value)]

    def predict(self):
        logger = getLogger(__name__)
        # logger.info(f"Predict from {self.mean=}")
        self.predicted = self.kf.transition_matrices @ self.mean
        return self.predicted

    def update(self, xy, value, missed=False):
        new_mean, new_covariance = self.kf.filter_update(
            self.mean, self.covariance, observation=np.array(xy)
        )
        self.history.append(HistoryItem(xy=xy, value=value))
        self.mean = new_mean
        self.covariance = new_covariance
        if missed:
            self.missed_duration += 1
        else:
            self.missed_duration = 0

    # 保留時間
    def missed(self, dummy_value):
        # 予測値でupdateする(?)
        xy = self.predicted
        self.update((int(xy[0]), int(xy[1])), value=dummy_value, missed=True)
        return self.missed_duration

    def closest(self, xy):
        # 速度変動の許容範囲
        d = np.linalg.norm(self.predicted - xy, axis=1)
        return xy[np.argmin(d)], np.min(d)


class MotionDetector:
    logger = getLogger(__name__)

    def __init__(self):
        self.flowing_points = {}
        self.next_label = 0

    def detect_iter(self, iterator, plot: bool = False):
        for frame_index, matchscore in iterator():
            yield from self._detect(matchscore, frame_index=frame_index, plot=plot)

    def _detect(self, matchscore: MatchScore, frame_index: int = None, plot=False):
        # persistentに直前までのピーク位置の履歴が保存されていて、
        # それぞれの新しい位置をカルマンフィルタで予測する。
        for p in self.flowing_points.keys():
            self.flowing_points[p].predict()

        # 高さが0.5以上の極大の位置を推定する。
        # random_scores = np.random.random([5, 5])
        maxima_values = {
            (int(x), int(y)): value
            for x, y, value in sorted(
                find_peaks(
                    matchscore.value,
                    Rect.from_bounds(
                        0, matchscore.value.shape[1], 0, matchscore.value.shape[0]
                    ),
                    height=0.3,
                ),
                key=lambda x: x[2],
                reverse=True,
            )[:3]
        }

        maxima = np.array(list(maxima_values.keys()))
        self.logger.debug(f"{maxima=}")

        unassigned_maxima = set([tuple(xy) for xy in maxima])
        missed_points = set(self.flowing_points.keys())

        if len(maxima) > 0:
            for p in self.flowing_points.keys():
                xy, d = self.flowing_points[p].closest(maxima)
                if xy is not None and d <= 3:
                    xy = tuple(xy)
                    self.flowing_points[p].update(
                        xy=xy, value=(frame_index, maxima_values[xy])
                    )
                    unassigned_maxima -= {xy}
                    missed_points -= {p}

        for p in missed_points:
            missed_duration = self.flowing_points[p].missed(
                dummy_value=(frame_index, 0)
            )
            if missed_duration > 3:
                self.logger.info(f"long missed {p=} {missed_duration=}")
                # 出力用に形をととのえる。長さ10フレーム以上のシーケンスに限定する。
                if len(self.flowing_points[p].history) >= 10:
                    yield p, self.flowing_points[p].history
                del self.flowing_points[p]

        for xy in unassigned_maxima:
            xy = tuple(xy)
            self.flowing_points[self.next_label] = Peak(
                xy=xy, value=(frame_index, maxima_values[xy])
            )
            self.next_label += 1

        # 個々のpeakについて、
        # 追跡しているpeak(
        # の延長線上に極めて近い場所にあるなら、
        # それを
        for p in self.flowing_points.keys():
            self.logger.debug(
                f"{p=}: {self.flowing_points[p].missed_duration=} {[h.xy for h in self.flowing_points[p].history]}"
            )
        self.logger.info("")
        if plot:
            self.plot(matchscore, frame_index=frame_index)

    def plot(self, matchscore: MatchScore, frame_index: int = None):
        # とりあえず、matchscore.valueを2次元の等高線で表示したい。
        # x軸とy軸の範囲はmatchscore.dxとmatchscore.dyから決める。
        x = np.linspace(matchscore.dx[0], matchscore.dx[-1], matchscore.value.shape[1])
        y = np.linspace(matchscore.dy[0], matchscore.dy[-1], matchscore.value.shape[0])
        # 値そのものをグラデーション表示
        # 位置をあわせる。
        X, Y = np.meshgrid(x, y)

        # imshowとcontourの座標系を統一
        plt.figure(figsize=(10, 8))

        # imshowで背景画像を表示（extentで座標範囲を指定）
        im = plt.imshow(
            matchscore.value,
            cmap="jet",
            extent=[
                matchscore.dx[0],
                matchscore.dx[-1],
                matchscore.dy[-1],
                matchscore.dy[0],
            ],  # y軸は反転
            aspect="auto",
            alpha=0.8,
        )

        # contourで等高線を重ねて描画
        contours = plt.contour(X, Y, matchscore.value, colors="white", linewidths=1.5)
        plt.clabel(contours, inline=True, fontsize=8)

        # カラーバーを追加（imshowの色情報を使用）
        plt.colorbar(im, label="Match Score Value")

        # 軸ラベルとタイトル
        plt.xlabel("dx")
        plt.ylabel("dy")
        plt.title(f"Motion Analysis {frame_index=}")

        plt.show()
        # print(np.min(matchscore.value), np.max(matchscore.value))


def __main__():
    basicConfig(level=INFO)
    with open("motions.json", "r") as f:
        motions = json.load(f)

    # 持続するpeak。番号は出現順でつける。
    def iterator():
        for motion in motions:
            matchscore = MatchScore(
                dx=eval(motions[motion]["dx"]),
                dy=eval(motions[motion]["dy"]),
                value=np.array(motions[motion]["value"]),
            )
            yield motion, matchscore

    motiondetector = MotionDetector()
    for history in motiondetector.detect_iter(iterator, plot=True):
        print(history)


if __name__ == "__main__":
    __main__()
