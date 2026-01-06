import cv2
import numpy as np
import sys
from logging import getLogger, DEBUG, basicConfig
from trainscanner2.video import video_loader_factory


# cv2.phasecorrelationを使う場合、極大を1つしか検出できないせいで、列車にひきずられる場合がある。
# オプションないので制御できない。


def analyze_iter_with_full_scores(vl, scaling_ratio=1.0):
    raw_frame = vl.next()
    if raw_frame is None:
        return

    frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
    height, width = frame.shape[:2]

    # ハニング窓の作成
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)

    first_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
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

        # 暫定の累積移動量で今のフレームをずらす
        M = np.float32([[1, 0, -abs_dx], [0, 1, -abs_dy]])
        shifted_gray = cv2.warpAffine(curr_gray, M, (width, height))

        # 最初のフレームとの残りのズレを推定
        shift, response = cv2.phaseCorrelate(first_gray, shifted_gray, window)
        dx, dy = shift

        # 累積移動量を更新
        abs_dx += dx
        abs_dy += dy

        if dx != 0 or dy != 0:
            print(
                f"Index: {frame_index}, peak=({dx:.2f}, {dy:.2f}), abs=({abs_dx:.2f}, {abs_dy:.2f}), response={response:.4f}"
            )

        # 補正: カメラの動き (abs_dx) を打ち消す方向に画像をずらす
        M_final = np.float32([[1, 0, -abs_dx], [0, 1, -abs_dy]])
        stabilized = cv2.warpAffine(curr_frame, M_final, (width, height))

        # 最初のフレームとの差分を表示
        cv2.imshow(
            "diff",
            cv2.absdiff(
                first_gray.astype(np.uint8),
                cv2.warpAffine(curr_gray, M_final, (width, height)).astype(np.uint8),
            ),
        )

        yield frame_index, (abs_dx, abs_dy), None, stabilized


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
