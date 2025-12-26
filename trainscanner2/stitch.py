# 高精細画像のstitching
# analyze.pyをできるだけ流用する。

import cv2
import numpy as np
import sys
import argparse
from logging import getLogger, DEBUG, INFO, basicConfig
import json
import os
from tqdm import tqdm
import time

from pyperbox import Rect
from trainscanner2.image import match_rect, diffImage, ImageRect
from trainscanner2.video import video_loader_factory
from trainscanner2 import FIFO, std_hdr
from trainscanner2.analyze import normalize, BlurMask
from trainscanner2.antishake import AntiShaker2
from trainscanner2.render import PathItem, WindowManager

# DEBUG表示を有効にするかどうか（環境変数で制御）
SHOW_DEBUG_WINDOWS = os.environ.get("TRAINSCANNER_DEBUG", "0") == "1"


def analyze_iter(vl, tspos2: dict, show_progress=False, progress_callback=None):
    """
    動画を読み込んで、各フレームをずらして自分自身と重ねあわせ、そのスコア(2次元行列)を返す。

    Args:
        vl: 動画ローダー
        tspos2: tspos2データ
        show_progress: 進捗表示するかどうか
        progress_callback: 進捗更新のコールバック関数 (current, total) -> None
    """
    logger = getLogger(__name__)

    magnify = int(1 / tspos2["scaling_factor"] + 1)
    unblurred_frames = FIFO(2)
    unblurred_frame_history = FIFO(5)
    # diff画像をたくわえ、動きの大きい領域を検出する。
    blurmask = BlurMask(lifetime=20)

    # 背景の移動をもとにてぶれを検出し、最初のフレームの位置から視野が流れていかないようにする。
    antishaker = AntiShaker2(velocity=magnify)
    damping = 0.05
    dumped = None

    mask = None

    # 進捗表示の設定
    history_items = tspos2["history"]
    total_frames = len(history_items)

    # プログレスバーの設定
    if show_progress and progress_callback is None:
        # コールバックがない場合はtqdmを使用
        progress_iter = enumerate(
            tqdm(history_items, desc="Processing frames", unit="frame")
        )
    else:
        # コールバックがある場合は通常のイテレータを使用
        progress_iter = enumerate(history_items)

    for i, frame_info in progress_iter:
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

        hdr_avg = std_hdr(averaged_background)
        # グレースケールに変換
        base_frame = unblurred_frames.queue[0]
        next_frame = unblurred_frames.queue[1]

        antimasked_hdr_base = std_hdr(base_frame) * antimask
        antimasked_hdr_next = std_hdr(next_frame) * antimask

        # 二乗差分画像を作る
        diff = (antimasked_hdr_base - antimasked_hdr_next) ** 2
        # blurmaskに追加する。maskは平均化されたマスク
        mask = blurmask.add_frame(diff)

        # maskは、diffの値が大きいピクセル。
        logger.debug(f"mask {np.min(mask)}, {np.max(mask)}")
        mask += np.min(mask)

        # 平均背景をさしひいて、前景を強調する。
        # 今はマスクを使っていない。
        base_masked = antimasked_hdr_base.copy() - hdr_avg  # * mask
        next_masked = antimasked_hdr_next.copy() - hdr_avg  # * mask

        # 照合する幅は、delta*magnifyの周囲±magnify
        max_shift = int(magnify)
        # frame_infoには既に現在のhistory要素が入っている
        dx, dy = (
            int(frame_info["delta_x"] * magnify),
            int(frame_info["delta_y"] * magnify),
        )

        base_masked_extended = np.zeros(
            [height + 2 * max_shift, width + 2 * max_shift],
            dtype=np.float32,
        )
        base_masked_extended[max_shift:-max_shift, max_shift:-max_shift] = base_masked

        # warpAffineを使わないほうが良い。
        # M = np.array([[1, 0, -dx], [0, 1, -dy]])
        # base_masked_extended = cv2.warpAffine(
        #     base_masked_extended, M, (width + 2 * max_shift, height + 2 * max_shift)
        # )
        base_masked_extended = np.roll(base_masked_extended, (-dy, -dx), axis=(0, 1))
        # print(dx, dy)
        # scoreとは、2つの画像のピクセル内積。1に近いほど画像が似ている=よく重なる。
        # matchscoreはtick付き行列。
        base_imagerect = ImageRect(
            image=base_masked_extended,
            lefttop=(-max_shift, -max_shift),
        )
        next_imagerect = ImageRect(image=next_masked, lefttop=(0, 0))
        matchrect = match_rect(base_imagerect, next_imagerect)

        # video frame index, absolute location of the frame, matchscore
        peak_result = matchrect.peak(subpixel=True)
        if peak_result is None:
            logger.warning(f"matchrect.peak() returned None for frame {frame_index}, using default values")
            vx, vy = 0, 0
            max_val = 0.0
        else:
            (vx, vy), max_val = peak_result
        dx += vx
        dy += vy
        # print(dx, dy)
        # print()

        if logger.getEffectiveLevel() == DEBUG:
            # デバッグ表示はメインスレッド以外ではクラッシュの原因になるため、
            # コマンドライン実行時のみ有効にするか、あるいは完全に無効化する
            pass

        if dumped is None:
            dumped = np.array((dx, dy))
        else:
            dumped = dumped * (1 - damping) + np.array((dx, dy)) * damping
            # if show_progress:
            #     print(
            #         f"Frame {frame_index}: raw=({dx:.2f}, {dy:.2f}), smoothed=({dumped[0]:.2f}, {dumped[1]:.2f})"
            #     )
            dx, dy = dumped

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
            # GUIスレッド以外でのimshowはクラッシュの原因になるため無効化
            pass

        # プログレスコールバックを呼び出し
        if progress_callback is not None:
            progress_callback(i + 1, total_frames)

        yield frame_index, (dx, dy), abs_loc, max_val, unblurred_frame


def stitch(tspos2file: str, verbose: bool = False, progress_callback=None):
    from trainscanner2.render import Render_one

    with open(tspos2file, "r") as f:
        tspos2 = json.load(f)
    
    # 動画ファイルのパスを決定するロジック
    original_videofile = tspos2["video_path"]
    
    if os.path.exists(original_videofile):
        # 1. まず、記録されているそのままのパス（絶対パス）を試す
        videofile = original_videofile
    else:
        # 2. 見つからない場合は、.tspos2ファイルと同じディレクトリを探す
        # (ファイルを移動した場合のフォールバック)
        basename = os.path.basename(original_videofile)
        tspos2path = os.path.dirname(tspos2file)
        videofile = os.path.join(tspos2path, basename)
        
        if not os.path.exists(videofile):
            # 3. それでも見つからない場合はエラー
            raise FileNotFoundError(
                f"動画ファイルが見つかりません。\n記録されたパス: {original_videofile}\n"
                f"または現在のディレクトリ: {videofile}"
            )

    vl = video_loader_factory(videofile)

    scaling_factor = tspos2["scaling_factor"]

    # ウィンドウ表示の設定
    window_manager = None
    if verbose:
        # -vオプションがある場合のみウィンドウを表示（ボタンは不要）
        window_manager = WindowManager(
            video_base=videofile,
            show_buttons=False,  # ボタンは表示しない
        )

    render = Render_one(
        id=0,
        num_leading_frames=1,
        window_manager=window_manager,
        scaling_factor=1.0 / scaling_factor,  # 高解像度への変換係数
        video_path=videofile,
        cache=True,
    )

    # 処理開始のメッセージ
    # if verbose:
    #     print(f"Starting high-resolution stitching...")
    #     print(f"Total frames to process: {len(tspos2['history'])}")

    # last_time = time.time()
    for frame_index, delta, absolute_position, max_val, unblurred_frame in tqdm(analyze_iter(
        vl, tspos2=tspos2, show_progress=verbose, progress_callback=progress_callback
    ), total=len(tspos2["history"])):
        render.put(
            unblurred_frame,
            PathItem(frame_index=frame_index, xy=delta, value=max_val),
            absolute_position=absolute_position,
        )
        # now = time.time()
        # print(f"Frame {frame_index} processed in {now - last_time:.2f} seconds")
        # last_time = now

    return render


def main():

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="高精細画像のstitching")
    parser.add_argument("tspos2file", nargs="?", help="tspos2ファイルのパス")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="詳細な進捗表示とウィンドウ表示"
    )
    args = parser.parse_args()

    # ログレベルを設定
    if args.verbose:
        basicConfig(level=DEBUG)
    else:
        basicConfig(level=INFO)

    # 動画を読み込む
    if args.tspos2file:
        tspos2file = args.tspos2file
    else:
        # デフォルトファイル（開発用）
        tspos2file = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/Kyushu/3397/IMG_3397_1.tspos2"
        tspos2file = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/Sapporo/C0085_trimmed_0.tspos2"
        tspos2file = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/Locals/Sanyo Yellow/IMG_0401_0.tspos2"

    render = stitch(tspos2file, verbose=args.verbose)
    # 最後に、stitchした大画像を保存する（メモリ効率的）
    # tspos2ファイル名から出力ファイル名を生成
    # 例: IMG_0401_0.tspos2 -> IMG_0401_0_hires.png
    base_path = os.path.splitext(tspos2file)[0] + "_hires"
    render.save(base_path=base_path)


if __name__ == "__main__":
    main()
