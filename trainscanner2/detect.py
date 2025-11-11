# motions.jsonは2次元の数値配列の時間変化を含む。(実際にはMatchScore形式のデータクラス)
# これを読みこみ、極大を複数見付けだし、その移動を追跡する。、
# 極大の個数はとりあえず最大で3個。
# 逐次処理できることがわかった。
from logging import getLogger, basicConfig, INFO, DEBUG
import json
import numpy as np
import matplotlib.pyplot as plt
import pykalman
import cv2
from pyperbox import Rect
from trainscanner.image import MatchRect
from trainscanner2 import PathItem
from trainscanner2.imagerect import ImageRect


class Path:
    """
    極大の位置と値を追跡する。欠測があってもカルマンフィルタが補う。
    """

    logger = getLogger(__name__)

    def __init__(self, id, xy, value):
        self.id = id
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
        # 連続する欠測の回数
        self.missed_duration = 0
        # 実測値の履歴。
        self.history = [PathItem(xy=xy, value=value)]

    # 予測し、結果は内部に保存する。
    def predict(self):
        # logger.debug(f"Predict from {self.mean=}")
        self.predicted = self.kf.transition_matrices @ self.mean
        return self.predicted

    # 実測値を記録する。
    def update(self, xy, value, missed=False):
        new_mean, new_covariance = self.kf.filter_update(
            self.mean, self.covariance, observation=np.array(xy)
        )
        self.history.append(PathItem(xy=xy, value=value))
        self.mean = new_mean
        self.covariance = new_covariance
        if missed:
            self.missed_duration += 1
        else:
            self.missed_duration = 0
        # if frame is not None:
        #     self._render(frame)

    # 欠測した場合の処理。予測値で補う。
    def missed(self, dummy_value):
        # 予測値でupdateする(?)
        xy = self.predicted
        self.update((int(xy[0]), int(xy[1])), value=dummy_value, missed=True)
        return self.missed_duration

    # 軌道に一番近い点と、それとの距離を返す。
    def closest(self, xy):
        # 速度変動の許容範囲
        d = np.linalg.norm(self.predicted - xy, axis=1)
        return xy[np.argmin(d)], np.min(d)


class MotionDetector:
    logger = getLogger(__name__)

    def __init__(self):
        self.paths = {}
        self.next_label = 0

    def done(self):
        # 最後まで生きのこったpathをpurgeする。
        # 辞書のコピーを作成してからイテレート（競合状態を回避）
        for path in list(self.paths.keys()):
            yield path, self.paths[path].history

    def _detect(
        self,
        matchrect: MatchRect,
        frame_index: int = None,
        plot: bool = False,
        max_miss: int = 5,
        min_score: float = 0.2,
        num_peaks: int = 3,
        velocity_uncertainty: float = 0.05,
    ):
        # self.pathsに直前までのピーク位置の履歴が保存されていて、
        # それぞれの新しい位置をカルマンフィルタで予測する。
        # 辞書のコピーを作成してからイテレート（競合状態を回避）
        for path_label in list(self.paths.values()):
            path_label.predict()

        # 高さが0.3以上の極大の位置を、スコアが大きい順に3つさがす。
        # maxima_values = {
        #     (int(x), int(y)): value
        #     for x, y, value in sorted(
        #         find_peaks(
        #             matchscore.value,
        #             Rect.from_bounds(
        #                 0, matchscore.value.shape[1], 0, matchscore.value.shape[0]
        #             ),
        #             height=min_score,
        #         ),
        #         key=lambda x: x[2],
        #         reverse=True,
        #     )[:num_peaks]
        # }
        if self.logger.getEffectiveLevel() == DEBUG:
            matchrect.plot(label=f"{frame_index=}")

        maxima = [
            (int(x), int(y), value)
            for (x, y), value in sorted(
                matchrect.peaks(
                    height=min_score,
                ),
                key=lambda x: x[1],
                reverse=True,
            )
            if int(x) != 0 or int(y) != 0
        ][:num_peaks]

        maxima_values = {(x, y): value for x, y, value in maxima}

        maxima_list = np.array(list(maxima_values.keys()))
        self.logger.info(f"{maxima_values=}")

        unassigned_maxima = {tuple(xy) for xy in maxima_list}
        missed_paths = set(self.paths.keys())
        dropped_paths = set()

        self.logger.debug(f"{self.paths.keys()=}")

        if len(maxima_list) > 0:
            # 追跡中の各pathについて
            # 辞書のコピーを作成してからイテレート（競合状態を回避）
            for path_label in list(self.paths.keys()):
                xy, d = self.paths[path_label].closest(maxima_list)
                L = np.linalg.norm(xy)
                threshold = L * velocity_uncertainty
                # これを大きめにしておかないと、分断されてしまうことがある。
                if threshold < 1.5:
                    threshold = 1.5
                # 一番近い極大がmax_shake以内にあれば (てぶれ等による多少のずれは許容する)
                if xy is not None and d <= threshold:
                    xy = tuple(xy)
                    # pathを更新する。
                    self.paths[path_label].update(
                        xy=xy, value=(frame_index, maxima_values[xy])
                    )
                    # 極大は割当て済み
                    unassigned_maxima -= {xy}
                    # パスも割当て済み
                    missed_paths -= {path_label}

        # まだ極大がみつかっていないパスについては、
        for path_label in missed_paths:
            # 予測値でごまかす
            missed_duration = self.paths[path_label].missed(
                dummy_value=(frame_index, 0)
            )
            # しかし連続でmax_miss回みのがした場合は、あきらめ、パスをyieldする処理に進む。
            if missed_duration >= max_miss:
                self.logger.debug(f"long missed {path_label=} {missed_duration=}")
                dropped_paths.add(path_label)

        # 野良極大
        for xy in unassigned_maxima:
            xy = tuple(xy)
            # 新しいパスを開始する
            self.paths[self.next_label] = Path(
                xy=xy,
                value=(frame_index, maxima_values[xy]),
                id=self.next_label,
            )
            self.next_label += 1

        # パスの合流を監視する。
        path_labels = list(self.paths.keys())
        self.logger.debug(f"{path_labels=}")

        final_path = {}

        for path_label in path_labels:
            if len(self.paths[path_label].history) < 3:
                continue
            tail = tuple(
                [
                    (int(h.xy[0]), int(h.xy[1]))
                    for h in self.paths[path_label].history[-3:]
                ]
            )
            # 2つのパスの間で、最後の3点の座標がまったく同じ場合は、パスが合流したとみなし、長い方(番号が若い方)を残し、短い方は抹消する。
            if tail in final_path:
                # 最後3frameの軌道が同じ場合は、新しいほうを廃止する。
                self.logger.debug(
                    f"The path {path_label} merges with final_path {final_path[tail]} {tail=}."
                )
                dropped_paths.add(path_label)
            else:
                final_path[tail] = path_label

        for tail, label in final_path.items():
            self.logger.debug(f"{label=} {tail=} {len(self.paths[label].history)=}")

        # 廃止処理は別ループ
        for path_label in dropped_paths:
            if path_label in self.paths:
                del self.paths[path_label]

        # 辞書のコピーを作成してからイテレート（競合状態を回避）
        for path_label in list(self.paths.keys()):
            self.logger.debug(
                f"{path_label=}: {self.paths[path_label].missed_duration=} {[h.xy for h in self.paths[path_label].history]}"
            )
        self.logger.debug("")
        if plot:
            matchrect.plot(label=f"{frame_index=}")

        active_paths = tuple(sorted(self.paths.keys()))
        return dict(self.paths), dropped_paths, active_paths


def __main__():
    basicConfig(level=INFO)
    with open("motions_test.json", "r") as f:
        motions = json.load(f)

    # 持続するpeak。番号は出現順でつける。
    def iterator():
        for motion in motions:
            left, right, top, bottom = motions[motion]["rect"]
            matchrect = MatchRect(
                rect=Rect.from_bounds(left, right, top, bottom),
                value=np.array(motions[motion]["value"]),
            )
            yield motion, matchrect

    motiondetector = MotionDetector()
    for _, matchrect in iterator():
        motiondetector._detect(matchrect, plot=True)


if __name__ == "__main__":
    __main__()
