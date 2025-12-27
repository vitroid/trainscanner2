# motions.jsonは2次元の数値配列の時間変化を含む。(実際にはMatchScore形式のデータクラス)
# これを読みこみ、極大を複数見付けだし、その移動を追跡する。、
# 極大の個数はとりあえず最大で3個。
# 逐次処理できることがわかった。
from logging import getLogger, basicConfig, INFO, DEBUG
import json
import numpy as np
import pykalman
from pyperbox import Rect
from trainscanner2.image import MatchRect, ImageRect
from trainscanner2 import PathItem


USE_KALMAN = False  # カルマンフィルタを有効にする場合はTrueにする


class Path:
    """
    極大の位置と値を追跡する。欠測があってもカルマンフィルタが補う。
    """

    logger = getLogger(__name__)

    def __init__(
        self, id: int, frame_index: int, xy: tuple[float, float], value: float
    ):
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
        self.history = [PathItem(frame_index=frame_index, xy=xy, value=value)]

    # 予測し、結果は内部に保存する。
    def predict(self):
        # logger.debug(f"Predict from {self.mean=}")
        self.predicted = self.kf.transition_matrices @ self.mean
        return self.predicted

    # 実測値を記録する。
    def update(
        self, frame_index: int, xy: tuple[float, float], value: float, missed=False
    ):
        if USE_KALMAN:
            new_mean, new_covariance = self.kf.filter_update(
                self.mean, self.covariance, observation=np.array(xy)
            )
            self.mean = new_mean
            self.covariance = new_covariance
        else:
            self.mean = np.array(xy)

        self.history.append(PathItem(frame_index=frame_index, xy=xy, value=value))
        if missed:
            self.missed_duration += 1
        else:
            self.missed_duration = 0
        # if frame is not None:
        #     self._render(frame)

    # 欠測した場合の処理。予測値で補う。
    def missed(self, frame_index: int, dummy_value: float):
        # 予測値でupdateする(?)
        xy = self.predicted
        self.update(frame_index=frame_index, xy=xy, value=dummy_value, missed=True)
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
        self.log = dict()

    def done(self):
        # 最後まで生きのこったpathをpurgeする。
        # 辞書のコピーを作成してからイテレート（競合状態を回避）
        for path in list(self.paths.keys()):
            yield path, self.paths[path].history

        # save self.log for debug.
        with open("peaks.log", "w") as f:
            for frame, value in self.log.items():
                for xy, remarks in value.items():
                    f.write(f"{frame} {xy[0]} {xy[1]} {remarks}\n")

    def _detect(
        self,
        matchrect: MatchRect,
        frame_index: int = None,
        plot: bool = False,
        max_miss: int = 10,
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

        self.log[frame_index] = dict()
        this_log = self.log[frame_index]

        maxima = [
            ((float(x), float(y)), value)
            for (x, y), value in sorted(
                matchrect.peaks(height=min_score, subpixel=True),
                key=lambda x: x[1],
                reverse=True,
            )
            if np.floor(x + 0.5) != 0.0 or np.floor(y + 0.5) != 0.0
        ][:num_peaks]

        maxima = dict(maxima)

        # top 3 peaksにマークする。
        for xy, value in maxima.items():
            this_log[xy] = f"top3:{value};"

        maxima_list = np.array(list(maxima.keys()))
        self.logger.info(f"maxima")
        for xy, value in maxima.items():
            self.logger.info(f"{xy=} {value=}")

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
                if threshold < 2.5:
                    threshold = 2.5
                # 一番近い極大がmax_shake以内にあれば (てぶれ等による多少のずれは許容する)
                if xy is not None and d <= threshold:
                    xy = tuple(xy)
                    # pathを更新する。
                    self.paths[path_label].update(
                        frame_index=frame_index, xy=xy, value=maxima[xy]
                    )
                    # 極大は割当て済み
                    unassigned_maxima -= {xy}
                    # パスも割当て済み
                    missed_paths -= {path_label}

                    this_log[xy] += f"assigned:{path_label};"

        # まだ極大がみつかっていないパスについては、
        for path_label in missed_paths:
            # 予測値でごまかす
            missed_duration = self.paths[path_label].missed(
                frame_index=frame_index,
                dummy_value=0,
            )
            this_path = self.paths[path_label]
            x, y = this_path.history[-1].xy
            x = float(x)
            y = float(y)
            assert (x, y) not in this_log
            this_log[x, y] = f"missed:{path_label};"
            # しかし連続でmax_miss回みのがした場合は、あきらめ、パスをyieldする処理に進む。
            if missed_duration >= max_miss:
                self.logger.debug(f"long missed {path_label=} {missed_duration=}")
                dropped_paths.add(path_label)

        # 野良極大
        for xy in unassigned_maxima:
            xy = tuple(xy)
            # 新しいパスを開始する
            self.paths[self.next_label] = Path(
                frame_index=frame_index,
                xy=xy,
                value=maxima[xy],
                id=self.next_label,
            )
            self.next_label += 1
            this_log[xy] += f"new:{self.next_label};"

        # パスの合流を監視する。
        path_labels = list(self.paths.keys())
        self.logger.debug(f"{path_labels=}")

        final_path = {}

        for path_label in path_labels:
            if len(self.paths[path_label].history) < 3:
                continue
            tail_intxy = tuple(
                [
                    (np.floor(h.xy[0] + 0.5), np.floor(h.xy[1] + 0.5))
                    for h in self.paths[path_label].history[-3:]
                ]
            )
            # 2つのパスの間で、最後の3点の座標がまったく同じ場合は、パスが合流したとみなし、長い方(番号が若い方)を残し、短い方は抹消する。
            if tail_intxy in final_path:
                # 最後3frameの軌道が同じ場合は、新しいほうを廃止する。
                self.logger.debug(
                    f"The path {path_label} merges with final_path {final_path[tail_intxy]} {tail_intxy=}."
                )
                dropped_paths.add(path_label)
                persist = final_path[tail_intxy]
                my_last_xy = self.paths[persist].history[-1].xy
                x = float(my_last_xy[0])
                y = float(my_last_xy[1])
                if (x, y) in this_log:
                    this_log[x, y] += f"merged:{path_label};"
                else:
                    this_log[x, y] = f"merged(?):{path_label};"
            else:
                final_path[tail_intxy] = path_label

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
