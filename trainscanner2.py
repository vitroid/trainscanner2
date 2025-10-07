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

scale = 1.0


# 自動スケールがほしい。SimpleImageを拡張する。
# どの絵を使うかの判断基準を再考する。
# やはりstitch過程を見たいぞ。別threadにまかせればいい。
# 毎ステップ、完全な照合を行うのではなく、10frameに一回照合を行い、あとは前後にpath追跡してつないでいくのはどうか。
# 短いpathは見落すが問題ない。
# あるいは、縮小画像で完全照合したあと、GUiで選んで完全解像度のものを再スキャンするか。
# 縮小画像は30万pixel上限にする。それだな。
# ただ、あまりに小さいと変位が見えなくなる。
# 完全解像度の時は、縮小画像で推定したpathのそばだけ見ればいいので爆速。しかも、その時にスリットの設定などを行えばなおいい。

# DONE
# てぶれ補正
# めっちゃ短いのにスコアがとても高いのは、たぶん背景なんだが、背景(0,0)をどの時点で排除するか。
# 同一軌道へ収束している軌道は併合してよい。treeの解析がほしい。
# なぜ平均スコアが0.3を大幅に下回る絵ができるのか=最後の0スコア点が平均を下げている?


def rotated_placement(canvas, frame, sine, cosine, train_position, first=False):
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
    # cv2.imshow("rotated", rotated)
    # cv2.waitKey(0)
    # 画像中心をそろえる
    if first:
        canvas.put_image(
            (int(train_position) - rw // 2, -rh // 2),
            rotated,
        )
    else:
        canvas.put_image(
            (int(train_position) - rw // 2, -rh // 2),
            rotated,
            linear_alpha=alpha,
        )


def render(vl, frame_positions, history, id=0):
    logger = getLogger(__name__)
    train_position = 0

    # まず、画像全体の配置を調査する。短かすぎる画像は生成するまでもない。
    for h in history:
        delta = h.xy
        frame_index, value = h.value
        dx, dy = delta
        dd = (dx**2 + dy**2) ** 0.5
        train_position += dd

    if train_position < len(history) * 2:
        # trainscanner2はあまりにも遅い列車の動きには対応していないので、100フレームで200pixelも進んでいないならそれは背景にすぎない。
        logger.debug(f"Ignore static image {id=} {train_position=} {len(history)=}")
        return

    train_position = 0
    scores = []
    first = True

    # with CachedImage(mode="new", dir=f"test{id}.pngs") as canvas:
    with SimpleImage() as canvas:
        for h in history:
            delta = h.xy
            frame_index, value = h.value
            logger.debug(f"{id=} {frame_index=} {delta=} ")
            scores.append(value)
            dx, dy = delta
            dd = -((dx**2 + dy**2) ** 0.5)
            if dd != 0:
                if first:
                    vl.seek(frame_index)
                raw_frame = vl.next()
                scaled_frame = cv2.resize(raw_frame, (0, 0), fx=scale, fy=scale)
                scaled_frame = np.roll(
                    scaled_frame, frame_positions[frame_index], axis=(1, 0)
                )
                train_position += dd
                cosine = dx / dd
                sine = dy / dd
                rotated_placement(
                    canvas, scaled_frame, sine, cosine, train_position, first
                )
                first = False
                # rotated_placement(canvas, scaled_frame, 0, 1.0, 0)
        img = canvas.get_image()
        if img is not None:
            cv2.imshow(f"{id} {np.mean(scores)}", img)
            cv2.waitKey(0)


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
    if "0835" in videofile:
        vl.seek(47 * 30)

    frame_positions = dict()

    def iterator():
        for frame_index, absolute_position, matchscore in analyze_iter(
            vl, scaling_ratio=scale
        ):
            logger.info(f"{frame_index=} {absolute_position=}")
            frame_positions[frame_index] = absolute_position
            yield frame_index, matchscore

    motiondetector = MotionDetector()
    best_score = 0.0
    for p_index, history in motiondetector.detect_iter(iterator, plot=False):
        # re-open the video
        vl = video_loader_factory(videofile)
        avg_score = np.mean([h.value[1] for h in history])
        if best_score * 0.8 < avg_score:
            if best_score < avg_score:
                best_score = avg_score
            render(vl, frame_positions, history, id=p_index)
        # storer.append(frame_index, absolute_position, matchscore)


if __name__ == "__main__":
    main()
