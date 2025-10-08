import sys
from logging import getLogger, INFO, basicConfig
from trainscanner.video import video_loader_factory
from trainscanner.image import linear_alpha
from detect import MotionDetector
from analyze import analyze_iter
import numpy as np
import cv2
from tiledimage.cachedimage import CachedImage
from tiledimage.simpleimage import SimpleImage
from render import Render

# 縮小画像で照合したあと、GUiで選んで完全解像度のものを再スキャンするか。
# 縮小画像は30万pixel上限にする。それだな。
# ただ、あまりに小さいと変位が見えなくなる。
# 完全解像度の時は、縮小画像で推定したpathのそばだけ見ればいいので爆速。しかも、その時にスリットの設定などを行えばなおいい。

# Rendererウィンドウに、捨てる/中止ボタン、保存するボタン、最高解像度で再レンダリングするボタンを準備する。
# 最高解像度でのレンダリングはたぶん別プログラムになるだろう。
# そいつには、historyとscaleを渡す。誤差の範囲で最高の照合をさせる。

# DONE
# てぶれ補正
# めっちゃ短いのにスコアがとても高いのは、たぶん背景なんだが、背景(0,0)をどの時点で排除するか。
# 同一軌道へ収束している軌道は併合してよい。treeの解析がほしい。
# なぜ平均スコアが0.3を大幅に下回る絵ができるのか=最後の0スコア点が平均を下げている?


def main():
    basicConfig(level=INFO)
    logger = getLogger(__name__)
    # 動画を読み込む
    if len(sys.argv) < 2:
        videofile = "examples/sample3.mov"
        videofile = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/他人の動画/antishake test/Untitled.mp4"

    else:
        videofile = sys.argv[1]
    vl = video_loader_factory(videofile)
    frame = vl.next()
    scale = (512 * 512 / (frame.shape[0] * frame.shape[1])) ** 0.5
    if scale > 1.0:
        scale = 1.0

    frame_positions = dict()

    renderer = Render()

    def iterator():
        for frame_index, absolute_position, matchscore, scaled_frame in analyze_iter(
            vl, scaling_ratio=scale
        ):
            logger.info(f"{frame_index=} {absolute_position=}")
            frame_positions[frame_index] = absolute_position
            yield frame_index, matchscore, scaled_frame

    motiondetector = MotionDetector()
    best_score = 0.0
    # def detect_iter(self, iterator, plot: bool = False):
    # iterator()からスコア行列をとりだし、pathをたどり、pathがとぎれたら鎖(移動ベクトルの列挙)を返す。
    for frame_index, matchscore, frame in iterator():
        paths = motiondetector._detect(matchscore, frame_index=frame_index)

        for id, path in paths.items():
            renderer.put(id, frame, path.history[-1])

    motiondetector.done()


if __name__ == "__main__":
    main()
