import sys
from logging import getLogger, INFO, basicConfig
from trainscanner.video import video_loader_factory
from trainscanner.image import linear_alpha
from detect import MotionDetector
from analize import analyze_iter
import numpy as np
import cv2
from tiledimage.cachedimage import CachedImage

scale = 0.25

# DONE
# てぶれ補正できてない。

# 同一軌道へ収束している軌道は併合してよい。treeの解析がほしい。
# なぜ平均スコアが0.3を大幅に下回る絵ができるのか=最後の0スコア点が平均を下げている?
# めっちゃ短いのにスコアがとても高いのは、たぶん背景なんだが、背景(0,0)をどの時点で排除するか。
# どの絵を使うかの判断基準を再考する。


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
    train_position = 0
    scores = []
    first = True
    with CachedImage(mode="new", dir=f"test{id}.pngs") as canvas:
        for h in history:
            delta = h.xy
            frame_index, value = h.value
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
