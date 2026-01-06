import cv2
import numpy as np
import sys
from logging import getLogger, DEBUG, basicConfig
from trainscanner2.video import video_loader_factory
from trainscanner2.image import standardize


def analyze_iter_with_full_scores(vl, scaling_ratio=1.0):
    raw_frame = vl.next()
    if raw_frame is None:
        return

    frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
    height, width = frame.shape[:2]

    # ハニング窓の作成（位相限定相関の精度向上のため）
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)

    last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    abs_dx, abs_dy = 0.0, 0.0

    # 最初のフレームをそのまま返す
    # yield 0, (0, 0), None, frame

    while True:
        frame_index = vl.head
        raw_frame = vl.next()
        if raw_frame is None:
            break

        curr_frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # cv2.phaseCorrelate を使用して 2 フレーム間のズレを推定
        # (dx, dy), response = cv2.phaseCorrelate(src1, src2, window)
        shift, response = cv2.phaseCorrelate(last_gray, curr_gray, window)
        dx, dy = shift

        # 累積移動量（カメラが動いた総量）
        abs_dx += dx
        abs_dy += dy

        if dx != 0 or dy != 0:
            print(f"Rel: ({dx:.2f}, {dy:.2f}), Abs: ({abs_dx:.2f}, {abs_dy:.2f})")

        # 補正: カメラの動き (abs_dx) を打ち消す方向に画像をずらす
        M = np.float32([[1, 0, -abs_dx], [0, 1, -abs_dy]])
        stabilized = cv2.warpAffine(curr_frame, M, (width, height))

        # cv2.phaseCorrelate はスコアマップを返さないため None を渡す
        yield frame_index, (abs_dx, abs_dy), None, stabilized
        last_gray = curr_gray


def main():
    basicConfig(level=DEBUG)
    videofile = (
        sys.argv[1] if len(sys.argv) > 1 else "../TrainScanner/examples/sample3.mov"
    )
    vl = video_loader_factory(videofile)
    scale = 0.5

    print("Press 'q' to quit.")

    for f_idx, pos, scores, stabilized in analyze_iter_with_full_scores(
        vl, scaling_ratio=scale
    ):
        cv2.imshow("Stabilized View", stabilized)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
