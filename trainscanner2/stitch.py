# 高精細画像のstitching
# analyze.pyをできるだけ流用する。

import cv2
import numpy as np
import sys
from logging import getLogger, DEBUG, INFO, basicConfig
import json
import os

# from sklearn.mixture import GaussianMixture
from trainscanner2.antishake import AntiShaker2
from trainscanner.image import match, standardize
from tiffeditor import Rect
from trainscanner.video import video_loader_factory
from trainscanner.image import MatchScore
from trainscanner2 import FIFO
from trainscanner2.analyze import normalize, BlurMask
from trainscanner2.render import PathItem, WindowManager

# DEBUG表示を有効にするかどうか（環境変数で制御）
SHOW_DEBUG_WINDOWS = os.environ.get("TRAINSCANNER_DEBUG", "0") == "1"


def analyze_iter(vl, tspos2: dict):
    """
    動画を読み込んで、各フレームをずらして自分自身と重ねあわせ、そのスコア(2次元行列)を返す。
    """
    logger = getLogger(__name__)

    magnify = int(1 / tspos2["scaling_factor"] + 1)
    unblurred_frames = FIFO(2)
    unblurred_frame_history = FIFO(5)
    # diff画像をたくわえ、動きの大きい領域を検出する。
    blurmask = BlurMask(lifetime=20)

    # 背景の移動をもとにてぶれを検出し、最初のフレームの位置から視野が流れていかないようにする。
    antishaker = AntiShaker2(velocity=magnify)

    mask = None
    for frame_info in tspos2["history"]:
        frame_index = frame_info["frame_index"]
        while vl.head < frame_index:
            vl.skip()
        raw_frame = vl.next()
        assert raw_frame is not None
        logger.debug(f"{frame_index=} {raw_frame.shape=}")

        height, width = raw_frame.shape[:2]

        if mask is None:
            mask = np.zeros(raw_frame.shape[:2], dtype=np.float32)
        if np.max(mask) == np.min(mask):
            # 全部1にする。
            antimask = np.ones_like(mask)
        else:
            # normalizeは値の範囲を0〜1間におさめる
            antimask = 1 - normalize(mask)

        # 直前のフレームからの変位deltaを測定し、積算してフレームごとの絶対位置abs_locを求める。
        # unblurred_scaled_frameは位置あわせしたあとのフレーム。以後の処理はこれを基準とする。
        unblurred_frame, delta, abs_loc = antishaker.add_frame(raw_frame, antimask)

        # unblurred_scaled_framesにはてぶれを修正し,最初のフレームの位置に背景がそろえられた画像が入る。
        unblurred_frames.append(unblurred_frame)
        unblurred_frame_history.append(unblurred_frame)

        if len(unblurred_frame_history.queue) < 2:
            continue
        # 平均画像=背景
        averaged_background = np.zeros_like(unblurred_frames.queue[0], dtype=np.float32)
        for fh in unblurred_frame_history.queue:
            averaged_background += fh
        averaged_background /= len(unblurred_frame_history.queue)

        std_log_gray_avg = standardize(
            np.log(
                cv2.cvtColor(averaged_background, cv2.COLOR_BGR2GRAY).astype(np.float32)
                + 1
            )
        )
        # グレースケールに変換
        base_frame = unblurred_frames.queue[0]
        next_frame = unblurred_frames.queue[1]

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
        if SHOW_DEBUG_WINDOWS:
            cv2.imshow("diff", diff)

        # maskは、diffの値が大きいピクセル。
        logger.debug(f"mask {np.min(mask)}, {np.max(mask)}")
        mask += np.min(mask)

        # 平均背景をさしひいて、前景を強調する。
        # 今はマスクを使っていない。
        base_masked = antimasked_std_log_gray_base.copy() - std_log_gray_avg  # * mask
        next_masked = antimasked_std_log_gray_next.copy() - std_log_gray_avg  # * mask

        # 照合する幅は、delta*magnifyの周囲±magnify
        max_shift = int(magnify) * 2
        # frame_infoには既に現在のhistory要素が入っている
        dx, dy = (
            frame_info["delta_x"] * magnify,
            frame_info["delta_y"] * magnify,
        )

        base_masked_extended = np.zeros(
            [height + 2 * max_shift, width + 2 * max_shift],
            dtype=np.float32,
        )
        base_masked_extended[max_shift:-max_shift, max_shift:-max_shift] = base_masked

        M = np.array([[1, 0, -dx], [0, 1, -dy]])
        base_masked_extended = cv2.warpAffine(
            base_masked_extended, M, (width + 2 * max_shift, height + 2 * max_shift)
        )
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
        _, max_val, _, max_loc = cv2.minMaxLoc(matchscore.value)
        dx += matchscore.dx[max_loc[0]]
        dy += matchscore.dy[max_loc[1]]

        logger.debug(f"frame_index={frame_index}")
        logger.debug(
            f"Rough: delta=({frame_info['delta_x'] * magnify:.2f}, "
            f"{frame_info['delta_y'] * magnify:.2f}), "
            f"score={frame_info['match_score']:.3f}, "
            f"abs_pos=({frame_info['abs_pos_x'] * magnify:.2f}, "
            f"{frame_info['abs_pos_y'] * magnify:.2f}), "
            f"magnify={magnify}"
        )
        logger.debug(
            f"Fine:  delta=({dx:.2f}, {dy:.2f}), score={max_val:.3f}, abs_loc={abs_loc}"
        )

        if SHOW_DEBUG_WINDOWS:
            diff2 = (
                base_masked_extended[
                    max_loc[1] : height + max_loc[1], max_loc[0] : width + max_loc[0]
                ]
                - next_masked
            )
            cv2.imshow("base_masked", base_masked)
            cv2.imshow("base_masked_extended", base_masked_extended)
            cv2.imshow("next_masked", next_masked)
            cv2.imshow("matchscore", matchscore.value)
            cv2.imshow("diff2", diff2)
            cv2.waitKey(0)

        yield frame_index, (dx, dy), abs_loc, max_val, unblurred_frame


def main():
    from trainscanner2.render import Render_one

    basicConfig(level=INFO)
    # 動画を読み込む
    if len(sys.argv) < 2:
        tspos2file = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/Kyushu/3397/IMG_3397_1.tspos2"
        tspos2file = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/Sapporo/C0085_trimmed_0.tspos2"
        tspos2file = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/Locals/Sanyo Yellow/IMG_0401_0.tspos2"
    else:
        tspos2file = sys.argv[1]
    with open(tspos2file, "r") as f:
        tspos2 = json.load(f)
    videofile = tspos2["video_path"]
    vl = video_loader_factory(videofile)

    scaling_factor = tspos2["scaling_factor"]
    render = Render_one(
        id=0, num_leading_frames=1, window_manager=WindowManager(video_base=videofile)
    )

    for frame_index, delta, absolute_position, max_val, unblurred_frame in analyze_iter(
        vl, tspos2=tspos2
    ):
        render.put(
            unblurred_frame,
            PathItem(xy=delta, value=(frame_index, max_val)),
            absolute_position=absolute_position,
        )
    # 最後に、stitchした大画像を保存する（メモリ効率的）
    # tspos2ファイル名から出力ファイル名を生成
    # 例: IMG_0401_0.tspos2 -> IMG_0401_0_hires.jpg
    base_path = os.path.splitext(tspos2file)[0] + "_hires"
    render.save(base_path=base_path)


if __name__ == "__main__":
    main()
