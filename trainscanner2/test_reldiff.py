import cv2
import numpy as np
import sys
from logging import getLogger, DEBUG, basicConfig
from trainscanner2.video import video_loader_factory
from trainscanner2.image import standardize, MatchRect
from trainscanner2.analyze import get_phase_correlation_score_map
from pyperbox import Rect


# 直前のフレームを基準に位置合わせ。
# 安定している(大きな移動を許容しないので)一方、誤差の蓄積が起こる。
# 誤差の蓄積はsubpixel処理である程度回避できる。この方法も回転には対応できない。
#
def normalize_for_display(x):
    """スコアマップを見やすくするために非線形にスケーリングして正規化する"""
    # 非常に鋭いピークを抑え、周囲の分布を見えるようにする
    x = np.maximum(x, 0)
    # 対数スケーリングにより、1ピクセルの鋭いピークと周囲の低スコアの差を圧縮
    x = np.log10(x * 1000 + 1)

    # 1ピクセルのピークを「太らせる」ためにガウシアンブラーを適用
    x = cv2.GaussianBlur(x, (9, 9), 0)

    v_min = np.min(x)
    v_max = np.max(x)
    if v_max > v_min:
        return (x - v_min) / (v_max - v_min)
    return x


def analyze_iter_with_full_scores(vl, scaling_ratio=1.0):
    raw_frame = vl.next()
    if raw_frame is None:
        return

    frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
    height, width = frame.shape[:2]

    last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    abs_dx, abs_dy = 0.0, 0.0

    # 最初のフレームをそのまま返す
    yield 0, (0, 0), None, frame

    # Hanning窓の作成（一度だけ作成して再利用）
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)

    while True:
        frame_index = vl.head
        raw_frame = vl.next()
        if raw_frame is None:
            break

        curr_frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 窓関数を適用してスコアマップ計算（エッジの影響を排除）
        scores = get_phase_correlation_score_map(last_gray, curr_gray, window=window)

        # MatchRectを使用してサブピクセル精度のピークを検出
        # スコアマップの中心を(0,0)とするRectを設定
        score_rect = Rect.from_bounds(
            -width // 2, width - (width // 2), -height // 2, height - (height // 2)
        )
        mr = MatchRect(value=scores, rect=score_rect)
        (dx, dy), max_val = mr.peak(subpixel=True)

        # ズレの累積（背景の揺れを追跡）
        abs_dx += dx
        abs_dy += dy

        # デバッグ出力（最初の数フレームのみ）
        if frame_index <= 5:
            print(
                f"[analyze_test] Frame {frame_index}: peak=({dx:.3f}, {dy:.3f}), "
                f"score={max_val:.4f}, "
                f"abs_pos=({abs_dx:.3f}, {abs_dy:.3f})"
            )

        if abs(dx) > 0.1 or abs(dy) > 0.1:
            print(f"dx={dx:.3f}, dy={dy:.3f}")

        # 補正: カメラの動きを打ち消す方向に画像をずらす
        M = np.float32([[1, 0, abs_dx], [0, 1, abs_dy]])
        stabilized = cv2.warpAffine(curr_frame, M, (width, height))

        # スコアマップをそのまま返し、メインループで表示できるようにします
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

            # カラーマップ（JET）を適用して温度図のように表示
            score_view = (score_view * 255).astype(np.uint8)
            score_view = cv2.applyColorMap(score_view, cv2.COLORMAP_JET)

            h, w = score_view.shape[:2]
            # 十字線を表示（中心位置）
            cv2.line(
                score_view,
                (w // 2 - 15, h // 2),
                (w // 2 + 15, h // 2),
                (255, 255, 255),
                1,
            )
            cv2.line(
                score_view,
                (w // 2, h // 2 - 15),
                (w // 2, h // 2 + 15),
                (255, 255, 255),
                1,
            )
            cv2.imshow("Phase Correlation Scores (Enhanced)", score_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            show_scores = not show_scores

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
