import cv2
import numpy as np
import sys
from logging import getLogger, DEBUG, basicConfig
from trainscanner2.video import video_loader_factory
from trainscanner2.image import standardize


def normalize_for_display(x):
    """スコアマップを見やすくするために非線形にスケーリングして正規化する"""
    # 非常に鋭いピークを抑え、周囲の分布を見えるようにする
    x = np.maximum(x, 0)
    x = np.sqrt(x)  # ガンマ補正的な効果
    v_min = np.min(x)
    v_max = np.max(x)
    if v_max > v_min:
        return (x - v_min) / (v_max - v_min)
    return x


def get_phase_correlation_score_map(img1, img2, window):
    """
    2つの画像間の位相限定相関行列を計算する。
    """
    s1 = standardize(img1) * window
    s2 = standardize(img2) * window

    f1 = np.fft.fft2(s1)
    f2 = np.fft.fft2(s2)

    # Cross-power spectrum
    # f1 に対して f2 がどれだけ動いたか
    cross_power = f2 * np.conj(f1)  # 順序を f2 * conj(f1) に変更
    normalized_cross_power = cross_power / (np.abs(cross_power) + 1e-15)

    score_map = np.abs(np.fft.ifft2(normalized_cross_power))
    score_map = np.fft.fftshift(score_map)

    return score_map


def analyze_iter_with_full_scores(vl, scaling_ratio=1.0):
    raw_frame = vl.next()
    if raw_frame is None:
        return

    frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
    height, width = frame.shape[:2]
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)

    last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    abs_dx, abs_dy = 0.0, 0.0

    # 最初のフレームをそのまま返す
    yield 0, (0, 0), None, frame

    while True:
        frame_index = vl.head
        raw_frame = vl.next()
        if raw_frame is None:
            break

        curr_frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        scores = get_phase_correlation_score_map(last_gray, curr_gray, window)

        _, max_val, _, max_loc = cv2.minMaxLoc(scores)

        # ズレの計算（符号を反転）
        # fftshift後の中心 (w//2, h//2) からの相対位置
        dx = max_loc[0] - (width // 2)
        dy = max_loc[1] - (height // 2)

        # デバッグ出力（最初の数フレームのみ）
        if frame_index <= 5:
            diff = cv2.absdiff(last_gray, curr_gray)
            diff_sum = np.sum(diff)
            print(
                f"[analyze_test] Frame {frame_index}: peak=({dx:.2f}, {dy:.2f}), "
                f"score={max_val:.4f}, "
                f"diff_sum={diff_sum:.0f}, "
                f"abs_pos=({abs_dx:.2f}, {abs_dy:.2f})"
            )

        # 相関がある程度高い場合のみ移動を更新
        if max_val > 0.03:
            # 累積移動量（カメラが動いた総量）
            abs_dx += dx
            abs_dy += dy

        # 補正: カメラの動き (abs_dx) を打ち消す方向に画像をずらす
        # cv2.warpAffine は「出力画像の各ピクセルが入力のどこから来るか」を指定するため
        # カメラの移動がプラスなら、補正量はマイナス
        M = np.float32([[1, 0, -abs_dx], [0, 1, -abs_dy]])
        stabilized = cv2.warpAffine(curr_frame, M, (width, height))

        yield frame_index, (abs_dx, abs_dy), scores, stabilized
        last_gray = curr_gray


def main():
    basicConfig(level=DEBUG)
    videofile = (
        sys.argv[1] if len(sys.argv) > 1 else "../TrainScanner/examples/sample3.mov"
    )
    vl = video_loader_factory(videofile)
    scale = 0.5

    show_scores = True
    print("Press 'q' to quit. 's' to toggle score map.")

    for f_idx, pos, scores, stabilized in analyze_iter_with_full_scores(
        vl, scaling_ratio=scale
    ):
        cv2.imshow("Stabilized View", stabilized)

        if scores is not None and show_scores:
            # スコアマップを強調表示
            score_view = normalize_for_display(scores)
            h, w = score_view.shape
            # 十字線を表示（中心位置）
            cv2.line(score_view, (w // 2 - 10, h // 2), (w // 2 + 10, h // 2), (1.0), 1)
            cv2.line(score_view, (w // 2, h // 2 - 10), (w // 2, h // 2 + 10), (1.0), 1)
            cv2.imshow("Phase Correlation Scores (Enhanced)", score_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            show_scores = not show_scores

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
