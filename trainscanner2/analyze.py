import cv2
import numpy as np
import sys
from logging import getLogger, DEBUG, basicConfig, INFO
import json

# from sklearn.mixture import GaussianMixture
from pyperbox import Rect
from trainscanner2.image import match_rect, MatchRect, ImageRect, standardize
from trainscanner2.video import video_loader_factory
from trainscanner2 import FIFO, std_hdr
from trainscanner2.antishake import AntiShaker2


# フレーム間の二乗差分を時間平均して、動きの大きい部分を抽出する。
class BlurMask:
    logger = getLogger(__name__)

    def __init__(self, lifetime=10):
        self.lifetime = lifetime
        self.masks = []
        self.sumask = None

    def add_frame(self, diff):
        # assert diff does not contain nan

        if np.isnan(diff).any():
            diff = np.zeros_like(diff)

        if self.sumask is None:
            self.sumask = diff
        else:
            self.sumask += diff

        self.masks.append(diff.copy())
        if len(self.masks) > self.lifetime:
            elim = self.masks.pop(0)
            self.sumask -= elim

        assert not np.isnan(self.sumask).any()
        return self.sumask / self.lifetime
        # return np.log(self.sumask + 1)


class BlurMask2:
    logger = getLogger(__name__)

    def __init__(self, lifetime=10):
        self.lifetime = lifetime
        self.mask = None

    def add_frame(self, diff):
        # assert diff does not contain nan

        if np.isnan(diff).any():
            diff = np.zeros_like(diff)
        # 型をfloat32に統一
        diff = diff.astype(np.float32, copy=False)
        if self.mask is None:
            self.mask = np.zeros_like(diff)

        # decay
        decayed = self.mask * (1 - 1 / self.lifetime)
        self.mask = cv2.max(diff, decayed)

        # if self.logger.getEffectiveLevel() <= INFO:
        #     cv2.imshow("diff", diff)
        #     cv2.imshow("mask", self.mask / (np.max(self.mask) + 1e-6))
        #     cv2.waitKey(0 if self.logger.getEffectiveLevel() == DEBUG else 1)
        self.logger.debug("----------")
        return self.mask


class BlurMask3:
    """
    内部で差分計算をするようにする。
    隣接フレームではなく、すこし遠いフレームを比較することで、遅い列車でも検出できるようにする。
    """

    logger = getLogger(__name__)

    def __init__(self, interval=30):
        self.images = FIFO(interval + 1)
        self.mask = None

    def add_frame(self, std_img):
        # assert diff does not contain nan
        self.images.append(std_img)

        diff = (self.images.queue[0] - self.images.queue[-1]) ** 2

        # 型をfloat32に統一
        diff = diff.astype(np.float32, copy=False)
        if self.mask is None:
            self.mask = np.zeros_like(diff)

        # decay
        decayed = self.mask * 0.9
        self.mask = cv2.max(diff, decayed)

        self.logger.debug("----------")
        return diff, self.mask


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


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
    cross_power = f2 * np.conj(f1)
    normalized_cross_power = cross_power / (np.abs(cross_power) + 1e-15)

    score_map = np.abs(np.fft.ifft2(normalized_cross_power))
    score_map = np.fft.fftshift(score_map)

    return score_map


def analyze_iter(vl, scaling_ratio=1.0, antishaker=None):
    """
    動画を読み込んで、各フレームをずらして自分自身と重ねあわせ、そのスコア(2次元行列)を返す。
    """
    logger = getLogger(__name__)

    # 最初のフレームを読み、スケールして保管する。
    raw_frame = vl.next()
    if raw_frame is None:
        return

    scaled_frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
    height, width = scaled_frame.shape[:2]
    scaled_gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
    # OpenCVの標準的な窓関数を使用
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    last_gray_frame = scaled_gray_frame
    # 浮動小数点で累積しないと、微小な手振れでドリフトします
    origin_x = 0.0
    origin_y = 0.0
    # fftshift後の中心座標
    center_x, center_y = width // 2, height // 2

    # 最初のフレームをそのまま返す（analyze_test.pyと同じ）
    yield 0, (0.0, 0.0), None, scaled_frame

    while True:
        frame_index = vl.head
        raw_frame = vl.next()
        if raw_frame is None:
            break

        scaled_frame = cv2.resize(raw_frame, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
        scaled_gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        del raw_frame

        # 位相限定相関を使ってスコアマップを計算（analyze_test.pyと同じ実装）
        scores = get_phase_correlation_score_map(
            last_gray_frame, scaled_gray_frame, window
        )

        # 背景の手振れ補正のため、中心付近（原点付近）のピークのみを探す
        # 動体が大きく動いている場合、最大ピークは動体の移動を表す可能性があるため、
        # 中心付近の範囲内で最大値を探す
        shake = 20  # 手振れの最大範囲（ピクセル単位）

        # 中心領域を抽出（境界チェック付き）
        y_start = max(0, center_y - shake)
        y_end = min(height, center_y + shake + 1)
        x_start = max(0, center_x - shake)
        x_end = min(width, center_x + shake + 1)
        center_region = scores[y_start:y_end, x_start:x_end]

        # 中心領域内で最大値を探す
        _, max_val, _, max_loc = cv2.minMaxLoc(center_region)

        # 中心領域内での相対位置を元の座標系に変換
        # cv2.minMaxLoc は (x, y) = (列, 行) の順序で返す
        peak_x = float((max_loc[0] + x_start) - center_x)
        peak_y = float((max_loc[1] + y_start) - center_y)

        # デバッグ: スコアマップ全体の最大値も確認（動体の移動を検出しているか）
        # analyze_test.pyと同じ方法で全体からも検出してみる
        _, global_max_val, _, global_max_loc = cv2.minMaxLoc(scores)
        global_peak_x = float(global_max_loc[0] - center_x)
        global_peak_y = float(global_max_loc[1] - center_y)

        # デバッグ出力（最初の数フレームのみ）
        if frame_index <= 5:
            diff = cv2.absdiff(last_gray_frame, scaled_gray_frame)
            diff_sum = np.sum(diff)
            print(
                f"[analyze] Frame {frame_index}: center_peak=({peak_x:.2f}, {peak_y:.2f}), "
                f"center_score={max_val:.4f}, "
                f"global_peak=({global_peak_x:.2f}, {global_peak_y:.2f}), "
                f"global_score={global_max_val:.4f}, "
                f"diff_sum={diff_sum:.0f}"
            )

        # 相関が低すぎる場合はノイズとみなして移動を無視（ドリフト・ぶっ飛び防止）
        # analyze_test.pyと同じ閾値を使用
        # ただし、中心領域のピークが低い場合は、グローバルピークを使用（動体の移動を検出している場合）
        if max_val > 0.03:
            origin_x += peak_x
            origin_y += peak_y
            logger.debug(
                f"Frame {frame_index}: peak=({peak_x:.2f}, {peak_y:.2f}), abs=({origin_x:.2f}, {origin_y:.2f}), score={max_val:.3f}"
            )
        elif (
            global_max_val > 0.03
            and abs(global_peak_x) < shake
            and abs(global_peak_y) < shake
        ):
            # 中心領域のピークが低いが、グローバルピークが中心付近にある場合はそれを使用
            origin_x += global_peak_x
            origin_y += global_peak_y
            logger.debug(
                f"Frame {frame_index}: using global_peak=({global_peak_x:.2f}, {global_peak_y:.2f}), abs=({origin_x:.2f}, {origin_y:.2f}), score={global_max_val:.3f}"
            )
        else:
            logger.debug(
                f"Frame {frame_index}: low correlation ({max_val:.3f}), ignoring movement"
            )
            print(f"  → 相関値が低いため、移動を無視しました")
        # 極大が原点になるようにscoreをずらす。(x,y)の順。
        scores = np.roll(scores, (-peak_x, -peak_y), axis=(1, 0))
        matchrect = MatchRect(
            value=scores,
            rect=Rect.from_bounds(
                left=-center_x,
                right=width - center_x,
                top=-center_y,
                bottom=height - center_y,
            ),
        )

        # np.roll の代わりに cv2.warpAffine を使用（サブピクセル精度、端の適切な処理）
        M = np.float32([[1, 0, -origin_x], [0, 1, -origin_y]])
        unblurred_frame = cv2.warpAffine(scaled_frame, M, (width, height))

        # 次のフレームとの比較のためにグレースケール画像を保存
        last_gray_frame = scaled_gray_frame

        # 返すもの。
        # 1. frame_index
        # 2. 新しいフレームの補正前の左上の絶対座標
        # 3. 照合スコア
        # 4. てぶれを補正した(つまり、旧フレームに位置をあわせた)縮小画像

        yield frame_index, (origin_x, origin_y), matchrect, unblurred_frame


def main0():

    class Storer:
        # withで使えるようにしたい。
        def __init__(self, filename: str):
            self.filename = filename
            self.matchrects = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            with open(self.filename, "w") as f:
                json.dump(self.matchrects, f, indent=4)

        def append(self, frame_index, absolute_position, matchrect: MatchRect):
            self.matchrects[frame_index] = {}
            self.matchrects[frame_index]["absolute_position"] = absolute_position
            value = matchrect.value
            self.matchrects[frame_index]["value"] = value.tolist()
            rect = matchrect.rect
            self.matchrects[frame_index]["rect"] = [
                rect.left,
                rect.right,
                rect.top,
                rect.bottom,
            ]

    basicConfig(level=DEBUG)
    # 動画を読み込む
    if len(sys.argv) < 2:
        videofile = "../TrainScanner/examples/sample3.mov"
        # videofile = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/TrainScannerWorkArea/他人の動画/antishake test/Untitled.mp4"

    else:
        videofile = sys.argv[1]
    vl = video_loader_factory(videofile)
    frame = vl.next()
    scale = (300 * 300 / (frame.shape[0] * frame.shape[1])) ** 0.5
    if scale > 1.0:
        scale = 1.0

    vl = video_loader_factory(videofile)
    if "0835" in videofile:
        vl.seek(47 * 30)

    with Storer("motions_test.json") as storer:
        for frame_index, absolute_position, matchrect, _ in analyze_iter(
            vl, scaling_ratio=scale
        ):
            storer.append(frame_index, absolute_position, matchrect)


def main():
    # てぶれ補正のみをimshowで確認する。
    basicConfig(level=DEBUG)
    if len(sys.argv) < 2:
        videofile = "../TrainScanner/examples/sample3.mov"
    else:
        videofile = sys.argv[1]
    vl = video_loader_factory(videofile)
    frame = vl.next()
    scale = (300 * 300 / (frame.shape[0] * frame.shape[1])) ** 0.5
    if scale > 1.0:
        scale = 1.0
    vl = video_loader_factory(videofile)

    # 補正前のフレームも保持するために、analyze_iterを少し変更するか、
    # ここで補正前のフレームを再取得する必要がある
    # 簡易的に、最初のフレームを保存して比較表示
    first_frame = None

    for frame_index, absolute_position, matchrect, unblurred_frame in analyze_iter(
        vl, scaling_ratio=scale
    ):
        if first_frame is None:
            first_frame = unblurred_frame.copy()

        abs_x, abs_y = absolute_position
        print(f"Frame {frame_index}: 累積移動量=({abs_x:.2f}, {abs_y:.2f})")

        # 補正後のフレームを表示
        cv2.imshow("Stabilized (補正後)", unblurred_frame)

        # スコアマップを表示（デバッグ用）
        if matchrect is not None:
            score_display = matchrect.plot_image()
            cv2.imshow("Score Map", score_display)

        # 累積移動量が大きい場合、黒い帯が見えるはず
        if abs(abs_x) > 5 or abs(abs_y) > 5:
            print(f"  → 大きな移動が検出されました！黒い帯が見えるはずです。")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
