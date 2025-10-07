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


# 自動スケールがほしい。SimpleImageを拡張する。
# どの絵を使うかの判断基準を再考する。
# やはりstitch過程を見たいぞ。別threadにまかせればいい。
# 毎ステップ、完全な照合を行うのではなく、10frameに一回照合を行い、あとは前後にpath追跡してつないでいくのはどうか。
# 短いpathは見落すが問題ない。
# あるいは、縮小画像で完全照合したあと、GUiで選んで完全解像度のものを再スキャンするか。
# 縮小画像は30万pixel上限にする。それだな。
# ただ、あまりに小さいと変位が見えなくなる。
# 完全解像度の時は、縮小画像で推定したpathのそばだけ見ればいいので爆速。しかも、その時にスリットの設定などを行えばなおいい。

# 途中経過を表示する場合にも、scoreによる足切りは必要。
# それを実装するには、path間の情報交換が必要になる。ちょっと難しい。逐次レンダラーはPathではないオブジェクトにまかせたい。
# 逐次レンダラーを使うのであれば、iteratorの設計指針はもっと変わる。

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

    def iterator():
        for frame_index, absolute_position, matchscore, scaled_frame in analyze_iter(
            vl, scaling_ratio=scale
        ):
            logger.info(f"{frame_index=} {absolute_position=}")
            frame_positions[frame_index] = absolute_position
            yield frame_index, matchscore, scaled_frame

    motiondetector = MotionDetector()
    best_score = 0.0
    for p_index, history in motiondetector.detect_iter(iterator, plot=False):
        # re-open the video
        # vl = video_loader_factory(videofile)
        # avg_score = np.mean([h.value[1] for h in history])
        # if best_score * 0.8 < avg_score:
        #     if best_score < avg_score:
        #         best_score = avg_score
        #     render(vl, frame_positions, history, id=p_index, scale=scale)
        # storer.append(frame_index, absolute_position, matchscore)
        pass


if __name__ == "__main__":
    main()
