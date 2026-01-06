import cv2
import numpy as np
import sys
from logging import getLogger, DEBUG, basicConfig
from trainscanner2.video import video_loader_factory
from trainscanner2.image import standardize

# 回転にも対応する位置合わせをCursorに書かせてみたが、まったく駄目。動いていない。


def analyze_iter_with_full_scores(vl, scaling_ratio=1.0):
    raw_frame = vl.next()
    if raw_frame is None:
        return

    frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
    height, width = frame.shape[:2]
    center = (width / 2, height / 2)
    max_radius = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)

    # ハニング窓の作成
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG

    def get_polar_image(gray_img):
        # 振幅スペクトルを計算
        f = np.fft.fftshift(np.fft.fft2(standardize(gray_img) * window))
        mag = np.log(np.abs(f) + 1e-15)
        # 対数極座標変換
        polar = cv2.warpPolar(mag, (width, height), center, max_radius, flags)
        return polar.astype(np.float32)

    last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    last_polar = get_polar_image(last_gray)

    abs_dx, abs_dy = 0.0, 0.0
    abs_angle = 0.0

    # 最初のフレームをそのまま返す
    yield 0, (0, 0, 0), None, frame

    while True:
        frame_index = vl.head
        raw_frame = vl.next()
        if raw_frame is None:
            break

        curr_frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_polar = get_polar_image(curr_gray)

        # 1. 回転の推定 (last_gray と curr_gray の間)
        # 極座標画像間で位相限定相関
        shift_p, response_p = cv2.phaseCorrelate(last_polar, curr_polar)

        # 垂直方向(y)が角度に対応
        d_angle_idx = shift_p[1]
        # 0付近のズレとして解釈 (height//2 が 0度に対応するよう fftshift されている場合を考慮)
        # cv2.phaseCorrelate は直接ズレを返す。極座標変換の特性上、yのズレが角度。
        # warpPolarのデフォルトでは 360度 = height
        d_angle = -(d_angle_idx * 360.0) / height

        # 180度反転の曖昧さ回避（手ぶれ補正なので小さい値を採用）
        if d_angle > 180:
            d_angle -= 360
        if d_angle < -180:
            d_angle += 360

        abs_angle += d_angle

        # 2. 推定された回転を適用して、平行移動を推定
        # curr_gray を今回の回転分だけ戻して last_gray と比較する
        R = cv2.getRotationMatrix2D(center, d_angle, 1.0)
        rotated_curr_gray = cv2.warpAffine(curr_gray, R, (width, height))

        shift_t, response_t = cv2.phaseCorrelate(last_gray, rotated_curr_gray, window)
        dx, dy = shift_t

        # 累積移動量を更新
        abs_dx += dx
        abs_dy += dy

        if dx != 0 or dy != 0 or d_angle != 0:
            print(
                f"Index: {frame_index}, d_angle: {d_angle:.2f}, d_pos: ({dx:.2f}, {dy:.2f}), response: {response_t:.4f}"
            )

        # 最終的な安定化画像を生成 (累積の回転と平行移動を適用)
        M = cv2.getRotationMatrix2D(center, abs_angle, 1.0)
        M[0, 2] += abs_dx
        M[1, 2] += abs_dy

        stabilized = cv2.warpAffine(curr_frame, M, (width, height))

        yield frame_index, (abs_dx, abs_dy, abs_angle), None, stabilized

        last_gray = curr_gray
        last_polar = curr_polar


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
