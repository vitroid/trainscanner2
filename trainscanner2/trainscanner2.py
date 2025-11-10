import sys
import os
import numpy as np
from logging import getLogger, INFO, DEBUG, basicConfig
from tqdm import tqdm

# PyQt6をoffscreenモードで実行（macOSのシステムサービスエラーを回避）
# os.environ["QT_QPA_PLATFORM"] = "offscreen"  # 一時的に無効化

from trainscanner2.video import video_loader_factory
from trainscanner2.analyze import analyze_iter
from trainscanner2.detect import MotionDetector
from trainscanner2.render import Render

# 縮小画像で照合したあと、GUiで選んで完全解像度のものを再スキャンするか。
# 縮小画像は30万pixel上限にする。それだな。
# ただ、あまりに小さいと変位が見えなくなる。
# 完全解像度の時は、縮小画像で推定したpathのそばだけ見ればいいので爆速。しかも、その時にスリットの設定などを行えばなおいい。

# quality至上主義にすると、同じ動画のなかに異なる品質のものがあった時に見逃す可能性がある。
# 時刻時刻での最良品質に対して比較するのが良い。
# あと、カメラのてぶれの許容範囲はもうちょっと大きくてもいいだろう。
# 00205.MTSからどれだけ成功を抽出できるかを評価基準にする。→205の場合は極端にカメラが動くので、antishakeを切ったほうがまし。参考にならない。


def main():
    basicConfig(level=INFO)
    logger = getLogger(__name__)

    # 動画を読み込む
    if len(sys.argv) < 2:
        videofile = "examples/sample3.mov"
        videofile = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/他人の動画/antishake test/Untitled.mp4"
        videofile = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/Czech Trams/00205/00205.MTS"
        videofile = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/Czech Trams/00199 8732/00199.MTS"
    else:
        videofile = sys.argv[1]

    vl = video_loader_factory(videofile)
    total_frames = vl.total_frames()
    frame = vl.next()
    scale = (300 * 300 / (frame.shape[0] * frame.shape[1])) ** 0.5
    if scale > 1.0:
        scale = 1.0

    frame_positions = {}

    # PyQt6ウィンドウで表示（OpenCVウィンドウを使う場合は use_pyqt=False）
    # マルチビューウィンドウを使用する場合は use_multiview=True
    renderer = Render(video_path=videofile, scaling_factor=scale, use_multiview=True)

    def iterator():
        # 実際の動画処理を使用
        for frame_index, absolute_position, matchscore, scaled_frame in analyze_iter(
            vl, scaling_ratio=scale
        ):
            frame_positions[frame_index] = absolute_position
            yield frame_index, absolute_position, matchscore, scaled_frame

    # 実際の動画処理
    logger.info("Starting video processing with MultiView window...")

    motiondetector = MotionDetector()

    for frame_index, absolute_position, matchscore, frame in iterator():
        paths, dropped_paths, active_path_ids = motiondetector._detect(
            matchscore, frame_index=frame_index
        )

        # デバッグ情報を出力
        logger.info(
            f"Frame {frame_index}: Found {len(paths)} paths, {len(dropped_paths)} dropped"
        )
        for path_id in paths.keys():
            logger.info(
                f"  Path {path_id}: {len(paths[path_id].history)} history items"
            )

        for id, path in paths.items():
            renderer.put(
                id, frame, path.history[-1], absolute_position=absolute_position
            )

        # アクティブなPath IDの一覧をMultiViewへ伝達（更新対象を限定）
        if hasattr(renderer, "multiview_manager") and renderer.multiview_manager:
            renderer.multiview_manager.set_active_paths(active_path_ids)

        for path_id in dropped_paths:
            renderer.mark_inactive(id=path_id)

        # GUIの更新を許可（非ブロッキング処理）
        if hasattr(renderer, "multiview_manager") and renderer.multiview_manager:
            renderer.multiview_manager.app.processEvents()

        # 処理の間隔を空けて、GUIの応答性を保つ
        # import time

        # time.sleep(0.1)

    # 残りのパスを完了として処理
    ## これ要るのか?
    for path_id, history in motiondetector.done():
        renderer.done(id=path_id)

    # 処理完了後、低品質ウィンドウを最終確認（処理中も随時閉じられている）
    logger.info("Processing complete. Final check for low-quality windows...")
    renderer.close_low_quality_windows(quality_ratio=0.5)

    # PyQt6ウィンドウが全て閉じられるまで待機
    logger.info("Close remaining windows to exit.")
    renderer.wait_for_windows_close()


if __name__ == "__main__":
    main()
