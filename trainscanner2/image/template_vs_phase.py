"""
Template MatchingとPhase Correlationの比較検証モジュール

このモジュールは、cv2.matchTemplateとPhase Correlation法（FFTベース）の違いを
徹底的に検証するためのツールを提供します。

Phase Correlation法は、畳み込み積分をFFTで高速化したものであり、
逆FFT後の相関マップから複数のピークを検出すれば、
Template Matchingと同等の機能を実現できます。
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from logging import getLogger

from trainscanner2.image import _find_peaks

logger = getLogger(__name__)


def template_matching(
    image: np.ndarray, template: np.ndarray, method=cv2.TM_CCOEFF_NORMED
) -> Tuple[Tuple[int, int], float]:
    """
    Template Matchingを実行し、最良のマッチ位置とスコアを返す。

    Args:
        image: 探索対象画像
        template: テンプレート画像
        method: マッチング手法（デフォルト: TM_CCOEFF_NORMED）

    Returns:
        ((x, y), score): 最良のマッチ位置とスコア
    """
    result = cv2.matchTemplate(image, template, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc, max_val


def fft_cross_correlation_full(
    image: np.ndarray, template: np.ndarray, use_phase_only: bool = False
) -> np.ndarray:
    """
    FFTベースの相関（畳み込み積分）を実装し、相関マップを返す。

    use_phase_only=Trueの場合、Phase Correlation（位相情報のみ）を計算
    use_phase_only=Falseの場合、通常の相関（Template Matchingと同等）を計算

    【重要】相関（correlation）と畳み込み（convolution）の違い:
    - 相関: R(x,y) = Σ I(x',y') * T(x'-x, y'-y)  [テンプレートを反転しない]
    - 畳み込み: C(x,y) = Σ I(x',y') * T(x-x', y-y')  [テンプレートを反転する]

    FFTで相関を計算するには、テンプレートを反転してからFFTする必要があります。
    しかし、実際には np.conj(template_fft) を使うことで、反転を回避できます。

    Args:
        image: 探索対象画像
        template: テンプレート画像
        use_phase_only: True=Phase Correlation, False=通常の相関

    Returns:
        相関マップ（Template Matchingの結果と同等のサイズ）
    """
    # 画像とテンプレートをfloat32に変換
    img = image.astype(np.float32)
    tmpl = template.astype(np.float32)

    h_img, w_img = img.shape[:2]
    h_tmpl, w_tmpl = tmpl.shape[:2]

    # テンプレートマッチングと同じ出力サイズ
    result_h = h_img - h_tmpl + 1
    result_w = w_img - w_tmpl + 1

    # 線形相関を実現するためのパディングサイズ
    # FFTで線形相関を得るには、画像サイズ + テンプレートサイズ - 1 が必要
    fft_h = h_img + h_tmpl - 1
    fft_w = w_img + w_tmpl - 1

    # 画像をパディング（左上に配置）
    img_padded = np.zeros((fft_h, fft_w), dtype=np.float32)
    img_padded[:h_img, :w_img] = img

    # テンプレートをパディング（左上に配置、反転はしない）
    # 相関を計算するには、np.conj()を使うことで反転を回避できる
    template_padded = np.zeros((fft_h, fft_w), dtype=np.float32)
    template_padded[:h_tmpl, :w_tmpl] = tmpl

    # FFTを実行
    img_fft = np.fft.fft2(img_padded)
    template_fft = np.fft.fft2(template_padded)

    # 相関を計算: I * conj(T) は相関を計算する（反転なし）
    # 畳み込みを計算する場合は I * T を使う（反転が必要）
    cross_power = img_fft * np.conj(template_fft)

    if use_phase_only:
        # Phase Correlation: 位相情報のみを使用
        cross_power_magnitude = np.abs(cross_power)
        epsilon = 1e-10
        cross_power_normalized = cross_power / (cross_power_magnitude + epsilon)
    else:
        # 通常の相関: 振幅と位相の両方を使用（Template Matchingと同等）
        cross_power_normalized = cross_power

    # 逆FFTで相関マップを取得
    correlation_map = np.real(np.fft.ifft2(cross_power_normalized))

    # 線形相関の結果を切り出し（左上の result_h x result_w 部分）
    # 注意: FFTの結果は、相関値が正しい位置に配置されている
    result = correlation_map[:result_h, :result_w]

    # 注意: 正規化は行わない（生の相関値を返す）
    # Template MatchingのTM_CCOEFF_NORMEDは各位置での正規化相関を計算しているため、
    # FFTベースの実装では生の相関値を返し、後で必要に応じて正規化する
    # 正規化すると最大値が常に1.0になり、実際の相関値の意味が失われる

    return result.astype(np.float32)


def phase_correlation_full(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Phase Correlation法を完全に実装し、相関マップ（逆FFT後の結果）を返す。

    これは位相情報のみを使用するPhase Correlationです。
    Template Matchingと同等の機能を得るには、fft_cross_correlation_full(use_phase_only=False)を使用してください。

    Args:
        image: 探索対象画像
        template: テンプレート画像

    Returns:
        相関マップ（Template Matchingの結果と同等のサイズ）
    """
    return fft_cross_correlation_full(image, template, use_phase_only=True)


def phase_correlation_peaks(
    image: np.ndarray,
    template: np.ndarray,
    min_score: float = 0.5,
    num_peaks: int = 5,
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Phase Correlation法で複数のピークを検出する。

    Template Matchingと同等の機能を提供し、FFTベースで高速化できる。

    Args:
        image: 探索対象画像
        template: テンプレート画像
        min_score: 最小スコア閾値
        num_peaks: 検出する最大ピーク数

    Returns:
        [((x, y), score), ...]: ピーク位置とスコアのリスト（スコア降順）
    """
    # 相関マップを取得
    correlation_map = phase_correlation_full(image, template)

    # 既存の_find_peaks関数を使用してピークを検出
    peak_positions = _find_peaks(correlation_map)

    # スコアでフィルタリングとソート
    peaks = []
    for x, y in peak_positions:
        score = float(correlation_map[y, x])
        if score >= min_score:
            peaks.append(((x, y), score))

    # スコアでソート（降順）
    peaks.sort(key=lambda x: x[1], reverse=True)

    return peaks[:num_peaks]


def phase_correlation(
    image1: np.ndarray, image2: np.ndarray
) -> Tuple[Tuple[float, float], float]:
    """
    Phase Correlationを実行し、位置ずれと信頼度を返す。
    （cv2.phaseCorrelateのラッパー、後方互換性のため）

    Args:
        image1: 基準画像
        image2: 比較対象画像

    Returns:
        ((dx, dy), response): 位置ずれと信頼度（0-1の範囲）
    """
    # 画像サイズが異なる場合は調整が必要
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # 同じサイズにする（大きい方に合わせる）
    h = max(h1, h2)
    w = max(w1, w2)

    # パディング
    img1_padded = np.zeros((h, w), dtype=np.float32)
    img2_padded = np.zeros((h, w), dtype=np.float32)

    img1_padded[:h1, :w1] = image1.astype(np.float32)
    img2_padded[:h2, :w2] = image2.astype(np.float32)

    # Phase Correlationを実行
    (dx, dy), response = cv2.phaseCorrelate(img1_padded, img2_padded)

    return (dx, dy), response


def compare_methods(
    image: np.ndarray,
    template: np.ndarray,
    expected_position: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    Template MatchingとPhase Correlation（複数ピーク検出版）の結果を比較する。

    Args:
        image: 探索対象画像
        template: テンプレート画像
        expected_position: 期待される位置（検証用、オプション）

    Returns:
        比較結果の辞書
    """
    results = {}

    # Template Matching (TM_CCOEFF_NORMED)
    tm_loc, tm_score = template_matching(image, template, cv2.TM_CCOEFF_NORMED)
    results["template_matching"] = {
        "position": tm_loc,
        "score": tm_score,
        "method": "TM_CCOEFF_NORMED",
    }

    # Template Matching (TM_CCORR_NORMED) - antishakeで使用
    tm_ccorr_loc, tm_ccorr_score = template_matching(
        image, template, cv2.TM_CCORR_NORMED
    )
    results["template_matching_ccorr"] = {
        "position": tm_ccorr_loc,
        "score": tm_ccorr_score,
        "method": "TM_CCORR_NORMED",
    }

    # FFTベースの相関（通常の相関、Template Matchingと同等）
    try:
        fft_corr_map = fft_cross_correlation_full(image, template, use_phase_only=False)

        # デバッグ情報: 相関マップの統計
        fft_map_min = float(np.min(fft_corr_map))
        fft_map_max = float(np.max(fft_corr_map))
        fft_map_mean = float(np.mean(fft_corr_map))
        fft_map_std = float(np.std(fft_corr_map))

        # ピーク検出（生の相関値を使用）
        # 閾値は相関マップの統計に基づいて設定
        # 平均値 + 2*標準偏差 を閾値として使用
        threshold = fft_map_mean + 2 * fft_map_std if fft_map_std > 0 else fft_map_mean

        fft_peaks = []
        peak_positions = _find_peaks(fft_corr_map)
        for x, y in peak_positions:
            score = float(fft_corr_map[y, x])
            if score >= threshold:
                fft_peaks.append(((x, y), score))
        fft_peaks.sort(key=lambda x: x[1], reverse=True)
        fft_peaks = fft_peaks[:5]

        if len(fft_peaks) > 0:
            best_peak = fft_peaks[0]
            results["fft_cross_correlation"] = {
                "best_position": best_peak[0],
                "best_score": best_peak[1],
                "all_peaks": fft_peaks,
                "num_peaks": len(fft_peaks),
                "note": "FFTベースの通常の相関（生の相関値、正規化なし）",
                "map_stats": {
                    "min": fft_map_min,
                    "max": fft_map_max,
                    "mean": fft_map_mean,
                    "std": fft_map_std,
                    "threshold": threshold,
                },
            }
        else:
            results["fft_cross_correlation"] = {
                "error": "No peaks found",
                "all_peaks": [],
                "map_stats": {
                    "min": fft_map_min,
                    "max": fft_map_max,
                    "mean": fft_map_mean,
                    "std": fft_map_std,
                    "threshold": threshold,
                },
            }
    except Exception as e:
        logger.error(f"FFT Cross Correlation (peaks) failed: {e}")
        results["fft_cross_correlation"] = {"error": str(e)}

    # Phase Correlation（複数ピーク検出版、位相情報のみ）
    try:
        pc_corr_map = phase_correlation_full(image, template)

        # デバッグ情報: 相関マップの統計
        pc_map_min = float(np.min(pc_corr_map))
        pc_map_max = float(np.max(pc_corr_map))
        pc_map_mean = float(np.mean(pc_corr_map))
        pc_map_std = float(np.std(pc_corr_map))

        # ピーク検出（生の相関値を使用）
        threshold = pc_map_mean + 2 * pc_map_std if pc_map_std > 0 else pc_map_mean

        pc_peaks = []
        peak_positions = _find_peaks(pc_corr_map)
        for x, y in peak_positions:
            score = float(pc_corr_map[y, x])
            if score >= threshold:
                pc_peaks.append(((x, y), score))
        pc_peaks.sort(key=lambda x: x[1], reverse=True)
        pc_peaks = pc_peaks[:5]

        if len(pc_peaks) > 0:
            best_peak = pc_peaks[0]
            results["phase_correlation_peaks"] = {
                "best_position": best_peak[0],
                "best_score": best_peak[1],
                "all_peaks": pc_peaks,
                "num_peaks": len(pc_peaks),
                "note": "Phase Correlation（位相情報のみ、生の相関値、正規化なし）",
                "map_stats": {
                    "min": pc_map_min,
                    "max": pc_map_max,
                    "mean": pc_map_mean,
                    "std": pc_map_std,
                    "threshold": threshold,
                },
            }
        else:
            results["phase_correlation_peaks"] = {
                "error": "No peaks found",
                "all_peaks": [],
                "map_stats": {
                    "min": pc_map_min,
                    "max": pc_map_max,
                    "mean": pc_map_mean,
                    "std": pc_map_std,
                    "threshold": threshold,
                },
            }
    except Exception as e:
        logger.error(f"Phase Correlation (peaks) failed: {e}")
        results["phase_correlation_peaks"] = {"error": str(e)}

    # 期待位置との比較
    if expected_position is not None:
        tm_error = abs(tm_loc[0] - expected_position[0]) + abs(
            tm_loc[1] - expected_position[1]
        )
        results["template_matching"]["error"] = tm_error

        if "best_position" in results.get("fft_cross_correlation", {}):
            fft_pos = results["fft_cross_correlation"]["best_position"]
            fft_error = abs(fft_pos[0] - expected_position[0]) + abs(
                fft_pos[1] - expected_position[1]
            )
            results["fft_cross_correlation"]["error"] = fft_error

        if "best_position" in results.get("phase_correlation_peaks", {}):
            pc_pos = results["phase_correlation_peaks"]["best_position"]
            pc_error = abs(pc_pos[0] - expected_position[0]) + abs(
                pc_pos[1] - expected_position[1]
            )
            results["phase_correlation_peaks"]["error"] = pc_error

    return results


def analyze_differences(
    image: np.ndarray,
    template: np.ndarray,
    moving_object: Optional[np.ndarray] = None,
    moving_object_position: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    動体が存在する場合のTemplate MatchingとPhase Correlationの違いを分析する。

    Args:
        image: 基準画像（背景）
        template: テンプレート画像
        moving_object: 動体の画像（オプション）
        moving_object_position: 動体の位置（オプション）

    Returns:
        分析結果の辞書
    """
    results = {}

    # 基準画像でのTemplate Matching
    tm_loc, tm_score = template_matching(image, template)
    results["background_template_matching"] = {
        "position": tm_loc,
        "score": tm_score,
    }

    # 動体を追加した画像
    if moving_object is not None and moving_object_position is not None:
        image_with_object = image.copy()
        y, x = moving_object_position
        h_obj, w_obj = moving_object.shape[:2]
        image_with_object[y : y + h_obj, x : x + w_obj] = moving_object

        # 動体ありでのTemplate Matching
        tm_obj_loc, tm_obj_score = template_matching(image_with_object, template)
        results["with_object_template_matching"] = {
            "position": tm_obj_loc,
            "score": tm_obj_score,
            "position_shift": (
                tm_obj_loc[0] - tm_loc[0],
                tm_obj_loc[1] - tm_loc[1],
            ),
        }

        # Phase Correlation（背景のみ）
        try:
            pc_shift, pc_response = phase_correlation(image, image_with_object)
            results["phase_correlation"] = {
                "shift": pc_shift,
                "response": pc_response,
                "note": "背景の移動を検出（動体はノイズとして扱われる）",
            }
        except Exception as e:
            results["phase_correlation"] = {"error": str(e)}

    return results


def create_test_case(
    image_size: Tuple[int, int] = (400, 600),
    template_size: Tuple[int, int] = (50, 50),
    template_position: Tuple[int, int] = (200, 300),
    noise_level: float = 0.0,
    brightness_change: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    テストケースを作成する。

    Args:
        image_size: 画像サイズ (height, width)
        template_size: テンプレートサイズ (height, width)
        template_position: テンプレートの位置 (y, x)
        noise_level: ノイズレベル（0-1）
        brightness_change: 明度変化（-1から1）

    Returns:
        (image, template, expected_position)
    """
    # 背景画像を作成（ノイズ付き）
    image = np.random.rand(*image_size).astype(np.float32) * 255
    if noise_level > 0:
        noise = np.random.randn(*image_size).astype(np.float32) * noise_level * 255
        image = np.clip(image + noise, 0, 255)

    # テンプレートを作成
    template = np.random.rand(*template_size).astype(np.float32) * 255

    # テンプレートを画像に配置
    y, x = template_position
    h_tmpl, w_tmpl = template_size
    image[y : y + h_tmpl, x : x + w_tmpl] = template

    # 明度変化を適用
    if brightness_change != 0:
        image = np.clip(image * (1 + brightness_change), 0, 255)

    expected_position = (x, y)  # (x, y)形式

    return image.astype(np.uint8), template.astype(np.uint8), expected_position


def compare_correlation_maps(image: np.ndarray, template: np.ndarray) -> dict:
    """
    Template MatchingとPhase Correlationの相関マップを直接比較する。

    Args:
        image: 探索対象画像
        template: テンプレート画像

    Returns:
        比較結果の辞書（相関マップ、統計情報など）
    """
    results = {}

    # Template Matchingの相関マップ
    tm_map = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    results["template_matching_map"] = {
        "shape": tm_map.shape,
        "min": float(np.min(tm_map)),
        "max": float(np.max(tm_map)),
        "mean": float(np.mean(tm_map)),
        "std": float(np.std(tm_map)),
    }

    # Phase Correlationの相関マップ
    try:
        pc_map = phase_correlation_full(image, template)
        results["phase_correlation_map"] = {
            "shape": pc_map.shape,
            "min": float(np.min(pc_map)),
            "max": float(np.max(pc_map)),
            "mean": float(np.mean(pc_map)),
            "std": float(np.std(pc_map)),
        }

        # 相関マップの類似度を計算（相関係数）
        if tm_map.shape == pc_map.shape:
            # 正規化してから相関係数を計算
            tm_norm = (tm_map - np.mean(tm_map)) / (np.std(tm_map) + 1e-10)
            pc_norm = (pc_map - np.mean(pc_map)) / (np.std(pc_map) + 1e-10)
            correlation = float(np.mean(tm_norm * pc_norm))
            results["map_correlation"] = correlation
        else:
            results["map_correlation"] = None
            results["shape_mismatch"] = f"TM: {tm_map.shape}, PC: {pc_map.shape}"

    except Exception as e:
        results["phase_correlation_map"] = {"error": str(e)}

    return results


def visualize_correlation_maps(
    image: np.ndarray, template: np.ndarray, expected_pos: Tuple[int, int]
):
    """
    各手法の相関マップ（score_map）を可視化し、スコアに合わせて着色する
    """
    # Template Matching
    tm_map = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    # FFTベースの相関（生の相関値）
    fft_map = fft_cross_correlation_full(image, template, use_phase_only=False)

    # Phase Correlation
    pc_map = phase_correlation_full(image, template)

    # 各マップを0-255の範囲に正規化（可視化用）
    def normalize_for_display(corr_map):
        """相関マップを0-255の範囲に正規化（可視化用）"""
        map_min = np.min(corr_map)
        map_max = np.max(corr_map)
        if map_max > map_min:
            normalized = ((corr_map - map_min) / (map_max - map_min) * 255).astype(
                np.uint8
            )
        else:
            normalized = np.zeros_like(corr_map, dtype=np.uint8)
        return normalized

    # カラーマップで着色（JET: 青→緑→黄→赤）
    tm_colored = cv2.applyColorMap(normalize_for_display(tm_map), cv2.COLORMAP_JET)
    fft_colored = cv2.applyColorMap(normalize_for_display(fft_map), cv2.COLORMAP_JET)
    pc_colored = cv2.applyColorMap(normalize_for_display(pc_map), cv2.COLORMAP_JET)

    # 期待位置にマーカーを描画
    exp_x, exp_y = expected_pos
    marker_color = (255, 255, 255)  # 白
    marker_thickness = 2

    # Template Matchingマップにマーカー
    if 0 <= exp_x < tm_map.shape[1] and 0 <= exp_y < tm_map.shape[0]:
        cv2.circle(tm_colored, (exp_x, exp_y), 5, marker_color, marker_thickness)
        cv2.putText(
            tm_colored,
            "Expected",
            (exp_x + 10, exp_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            marker_color,
            1,
        )

    # FFT相関マップにマーカー
    if 0 <= exp_x < fft_map.shape[1] and 0 <= exp_y < fft_map.shape[0]:
        cv2.circle(fft_colored, (exp_x, exp_y), 5, marker_color, marker_thickness)
        cv2.putText(
            fft_colored,
            "Expected",
            (exp_x + 10, exp_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            marker_color,
            1,
        )

    # Phase Correlationマップにマーカー
    if 0 <= exp_x < pc_map.shape[1] and 0 <= exp_y < pc_map.shape[0]:
        cv2.circle(pc_colored, (exp_x, exp_y), 5, marker_color, marker_thickness)
        cv2.putText(
            pc_colored,
            "Expected",
            (exp_x + 10, exp_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            marker_color,
            1,
        )

    # 最大値の位置にマーカーを描画
    _, tm_max_val, _, tm_max_loc = cv2.minMaxLoc(tm_map)
    fft_max_idx = np.unravel_index(np.argmax(fft_map), fft_map.shape)
    fft_max_loc = (fft_max_idx[1], fft_max_idx[0])
    pc_max_idx = np.unravel_index(np.argmax(pc_map), pc_map.shape)
    pc_max_loc = (pc_max_idx[1], pc_max_idx[0])

    max_marker_color = (0, 255, 0)  # 緑
    cv2.circle(tm_colored, tm_max_loc, 5, max_marker_color, marker_thickness)
    cv2.putText(
        tm_colored,
        "Max",
        (tm_max_loc[0] + 10, tm_max_loc[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        max_marker_color,
        1,
    )

    cv2.circle(fft_colored, fft_max_loc, 5, max_marker_color, marker_thickness)
    cv2.putText(
        fft_colored,
        "Max",
        (fft_max_loc[0] + 10, fft_max_loc[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        max_marker_color,
        1,
    )

    cv2.circle(pc_colored, pc_max_loc, 5, max_marker_color, marker_thickness)
    cv2.putText(
        pc_colored,
        "Max",
        (pc_max_loc[0] + 10, pc_max_loc[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        max_marker_color,
        1,
    )

    # 表示（サイズを調整して表示）
    scale = 2.0  # 拡大倍率
    tm_display = cv2.resize(
        tm_colored, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    fft_display = cv2.resize(
        fft_colored, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    pc_display = cv2.resize(
        pc_colored, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow("Template Matching (TM_CCOEFF_NORMED)", tm_display)
    cv2.imshow("FFT Cross Correlation", fft_display)
    cv2.imshow("Phase Correlation", pc_display)

    print("\n相関マップを表示しました。")
    print("  - Template Matching: 正規化相関（TM_CCOEFF_NORMED）")
    print("  - FFT Cross Correlation: 生の相関値（FFTベース）")
    print("  - Phase Correlation: 位相情報のみ（FFTベース）")
    print("  - 白いマーカー: 期待位置")
    print("  - 緑のマーカー: 最大値の位置")
    print("  - カラーマップ: JET（青=低スコア、赤=高スコア）")
    print("\nキーを押すと次の処理に進みます...")


def debug_fft_correlation(
    image: np.ndarray, template: np.ndarray, expected_pos: Tuple[int, int]
):
    """
    FFTベースの相関とTemplate Matchingの結果を詳細に比較するデバッグ関数
    """
    print("\n=== FFT相関のデバッグ ===")

    # Template Matching
    tm_map = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, tm_max_val, _, tm_max_loc = cv2.minMaxLoc(tm_map)
    print(f"Template Matching: position={tm_max_loc}, score={tm_max_val:.6f}")
    print(f"期待位置: {expected_pos}")
    print(
        f"Template Matching誤差: x={abs(tm_max_loc[0] - expected_pos[0])}, y={abs(tm_max_loc[1] - expected_pos[1])}"
    )

    # FFTベースの相関（生の相関値）
    fft_map = fft_cross_correlation_full(image, template, use_phase_only=False)
    fft_max_idx = np.unravel_index(np.argmax(fft_map), fft_map.shape)
    fft_max_val = fft_map[fft_max_idx]
    fft_max_loc = (fft_max_idx[1], fft_max_idx[0])  # (x, y)形式
    print(f"\nFFT相関（生の値）: position={fft_max_loc}, score={fft_max_val:.6f}")
    print(
        f"FFT相関誤差: x={abs(fft_max_loc[0] - expected_pos[0])}, y={abs(fft_max_loc[1] - expected_pos[1])}"
    )
    print(
        f"FFT相関マップ統計: min={np.min(fft_map):.6f}, max={np.max(fft_map):.6f}, mean={np.mean(fft_map):.6f}, std={np.std(fft_map):.6f}"
    )

    # 期待位置での相関値を確認
    exp_y, exp_x = expected_pos[1], expected_pos[0]  # (x, y) -> (y, x)
    if 0 <= exp_y < fft_map.shape[0] and 0 <= exp_x < fft_map.shape[1]:
        exp_val = fft_map[exp_y, exp_x]
        print(f"期待位置でのFFT相関値: {exp_val:.6f}")
        print(f"最大値との差: {fft_max_val - exp_val:.6f}")

    # 相関マップの相関係数を計算
    if tm_map.shape == fft_map.shape:
        # 正規化してから相関係数を計算
        tm_norm = (tm_map - np.mean(tm_map)) / (np.std(tm_map) + 1e-10)
        fft_norm = (fft_map - np.mean(fft_map)) / (np.std(fft_map) + 1e-10)
        correlation = float(np.mean(tm_norm * fft_norm))
        print(f"相関マップ間の相関係数: {correlation:.6f}")

    # 期待位置周辺の相関値を表示
    print(f"\n期待位置周辺のFFT相関値:")
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y, x = exp_y + dy, exp_x + dx
            if 0 <= y < fft_map.shape[0] and 0 <= x < fft_map.shape[1]:
                val = fft_map[y, x]
                marker = " <-- 期待位置" if dx == 0 and dy == 0 else ""
                print(f"  ({x:3d}, {y:3d}): {val:12.6f}{marker}")

    print(f"\n最大値周辺のFFT相関値:")
    max_y, max_x = fft_max_idx
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y, x = max_y + dy, max_x + dx
            if 0 <= y < fft_map.shape[0] and 0 <= x < fft_map.shape[1]:
                val = fft_map[y, x]
                marker = " <-- 最大値" if dx == 0 and dy == 0 else ""
                print(f"  ({x:3d}, {y:3d}): {val:12.6f}{marker}")


if __name__ == "__main__":
    # テストケース1: 基本的な比較
    print("=== テストケース1: 基本的な比較 ===")
    image, template, expected_pos = create_test_case()

    # 相関マップを可視化
    visualize_correlation_maps(image, template, expected_pos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # デバッグ情報を表示
    debug_fft_correlation(image, template, expected_pos)
    # cv2.imshow("template", template)
    results = compare_methods(image, template, expected_pos)
    # image上に、それぞれの方法で検出した位置を異なる色の四角で描きこむ。
    # imageは白黒の場合は事前にカラーにしておく必要があるね。
    # color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # for method, result in results.items():
    #     if "position" in result:
    #         x, y = result["position"]
    #     else:
    #         x, y = result["best_position"]
    #     cv2.rectangle(
    #         color_image,
    #         (x, y),
    #         (x + template.shape[1], y + template.shape[0]),
    #         (
    #             (0, 0, 255)
    #             if method == "template_matching_ccorr"
    #             else ((0, 255, 0) if method == "fft_cross_correlation" else (255, 0, 0))
    #         ),
    #         2,
    #     )
    # cv2.imshow("image", color_image)
    # cv2.waitKey(0)
    print(f"期待位置: {expected_pos}")
    print(f"Template Matching: {results['template_matching']}")
    print(f"FFT Cross Correlation (peaks): {results.get('fft_cross_correlation', {})}")
    print(f"Phase Correlation (peaks): {results.get('phase_correlation_peaks', {})}")
    print()

    # 相関マップの比較
    print("=== 相関マップの比較 ===")
    map_comparison = compare_correlation_maps(image, template)
    print(f"Template Matching Map: {map_comparison['template_matching_map']}")
    print(f"Phase Correlation Map: {map_comparison.get('phase_correlation_map', {})}")
    if "map_correlation" in map_comparison:
        print(f"相関マップ間の相関係数: {map_comparison['map_correlation']:.4f}")
    print()

    # テストケース2: 背景が手ぶれで移動 + 前景の動体が独立に移動
    print("=== テストケース2: 背景の手ぶれ + 前景の動体検出（前景の情報なし） ===")

    # 背景画像1（ノイズ）
    background1 = np.random.rand(400, 600).astype(np.float32) * 255
    background1 = background1.astype(np.uint8)

    # 背景画像2（背景1を数ピクセルずらしたもの = 手ぶれ）
    background_shift_x = 5  # 背景の手ぶれ量（x方向）
    background_shift_y = 3  # 背景の手ぶれ量（y方向）
    background2 = np.zeros_like(background1)
    background2[background_shift_y:, background_shift_x:] = background1[
        : background1.shape[0] - background_shift_y,
        : background1.shape[1] - background_shift_x,
    ]

    # 前景の動体（ノイズでできた正方形）- 検出時には使用しない
    object_size = 40
    foreground_object = (
        np.random.rand(object_size, object_size).astype(np.float32) * 255
    )
    foreground_object = foreground_object.astype(np.uint8)

    # 前景の移動量（50px以下）
    foreground_shift_x = 30  # 前景の移動量（x方向）
    foreground_shift_y = 20  # 前景の移動量（y方向）

    # 画像1に前景を埋め込む
    obj1_y, obj1_x = 150, 200  # 前景の初期位置（検証用のみ）
    image1 = background1.copy()
    image1[obj1_y : obj1_y + object_size, obj1_x : obj1_x + object_size] = (
        foreground_object
    )

    # 画像2に前景を埋め込む（異なる位置）
    obj2_y = obj1_y + foreground_shift_y
    obj2_x = obj1_x + foreground_shift_x
    image2 = background2.copy()
    image2[obj2_y : obj2_y + object_size, obj2_x : obj2_x + object_size] = (
        foreground_object
    )

    # 期待される前景の移動量（検証用のみ）
    expected_foreground_shift = (foreground_shift_x, foreground_shift_y)

    print(f"背景の手ぶれ: ({background_shift_x}, {background_shift_y})")
    print(f"前景の移動量（検証用）: {expected_foreground_shift}")
    print(f"前景の初期位置（検証用）: ({obj1_x}, {obj1_y})")
    print(f"前景の移動後位置（検証用）: ({obj2_x}, {obj2_y})")
    print()

    # 画像1をパディング（前景の移動量50px + 前景の大きさobject_size分）
    max_shift = 50
    padding_size = max_shift + object_size
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # 画像1をパディング（上下左右にpadding_size分）
    image1_padded = np.zeros(
        (h1 + 2 * padding_size, w1 + 2 * padding_size), dtype=image1.dtype
    )
    image1_padded[
        padding_size : padding_size + h1, padding_size : padding_size + w1
    ] = image1

    print(f"--- Template Matching（画像2を画像1内で検索） ---")
    print(f"画像1のサイズ: {image1.shape}")
    print(f"画像2のサイズ: {image2.shape}")
    print(
        f"パディングサイズ: {padding_size}px (最大移動量{max_shift}px + 前景の大きさ{object_size}px)"
    )
    print(f"パディング後の画像1のサイズ: {image1_padded.shape}")
    print()

    # Phase Correlationで背景の移動量を検出（前景検出の基準として使用）
    print("--- Phase Correlation (背景の移動量を検出) ---")
    try:
        pc_shift, pc_response = phase_correlation(image1, image2)
        detected_background_shift = (pc_shift[0], pc_shift[1])
        print(f"検出された背景の移動量: ({pc_shift[0]:.2f}, {pc_shift[1]:.2f})")
        print(f"期待される背景の移動量: ({background_shift_x}, {background_shift_y})")
        print(f"相関値: {pc_response:.4f}")
    except Exception as e:
        print(f"Phase Correlation failed: {e}")
        detected_background_shift = (background_shift_x, background_shift_y)
    print()

    # 画像2をテンプレートとして、パディングされた画像1内でTemplate Matchingを実行
    tm_result = cv2.matchTemplate(image1_padded, image2, cv2.TM_CCOEFF_NORMED)

    # 複数のピークを検出
    peak_positions = _find_peaks(tm_result)
    peaks = []
    for x, y in peak_positions:
        score = float(tm_result[y, x])
        # パディングを考慮して位置を調整
        adjusted_x = x - padding_size
        adjusted_y = y - padding_size
        peaks.append(((adjusted_x, adjusted_y), (x, y), score))

    # スコアでソート
    peaks.sort(key=lambda x: x[2], reverse=True)

    print(f"--- Template Matching（複数ピーク検出） ---")
    print(f"検出されたピーク数: {len(peaks)}")
    print("上位10個のピーク:")
    for i, ((adj_x, adj_y), (pad_x, pad_y), score) in enumerate(peaks[:10]):
        # 背景の移動量との差を計算
        bg_diff = abs(adj_x - detected_background_shift[0]) + abs(
            adj_y - detected_background_shift[1]
        )
        peak_type = "背景" if bg_diff < 5 else "前景候補"
        print(
            f"  {i+1}. 位置（画像1座標系）: ({adj_x}, {adj_y}), "
            f"スコア: {score:.4f}, 背景との差: {bg_diff:.2f}px [{peak_type}]"
        )
    print()

    # 前景の移動量を推定（背景の移動量と異なるピークを探す）
    print("--- 前景の移動量推定 ---")
    foreground_peaks = []
    for (adj_x, adj_y), _, score in peaks:
        bg_diff = abs(adj_x - detected_background_shift[0]) + abs(
            adj_y - detected_background_shift[1]
        )
        if bg_diff > 5:  # 背景の移動量と5px以上の差がある
            foreground_peaks.append(((adj_x, adj_y), score, bg_diff))

    if len(foreground_peaks) > 0:
        # スコアでソート
        foreground_peaks.sort(key=lambda x: x[1], reverse=True)
        best_foreground = foreground_peaks[0]
        detected_x, detected_y = best_foreground[0]
        detected_foreground_shift = (detected_x - obj1_x, detected_y - obj1_y)

        print(f"最良の前景候補:")
        print(f"  位置（画像1座標系）: ({detected_x}, {detected_y})")
        print(f"  スコア: {best_foreground[1]:.4f}")
        print(f"  背景との差: {best_foreground[2]:.2f}px")
        print(f"  検出された前景の移動量: {detected_foreground_shift}")
        print(f"  期待される前景の移動量: {expected_foreground_shift}")
        error = abs(detected_foreground_shift[0] - expected_foreground_shift[0]) + abs(
            detected_foreground_shift[1] - expected_foreground_shift[1]
        )
        print(f"  誤差: {error:.2f} px")
    else:
        print("前景候補が見つかりませんでした")
        detected_foreground_shift = None
        detected_x, detected_y = None, None
    print()

    # 可視化
    print("--- 可視化 ---")
    # 画像1と画像2を表示
    cv2.imshow("Image 1 (Background + Foreground)", image1)
    cv2.imshow("Image 2 (Shifted Background + Moved Foreground)", image2)

    # 相関マップを可視化
    tm_colored = cv2.applyColorMap(
        (
            (tm_result - np.min(tm_result))
            / (np.max(tm_result) - np.min(tm_result) + 1e-10)
            * 255
        ).astype(np.uint8),
        cv2.COLORMAP_JET,
    )

    # 複数のピークにマーカーを描画
    # 背景のピーク（青）- 最大スコアのピーク（通常は背景）
    if len(peaks) > 0:
        bg_peak = peaks[0]
        bg_pad_x, bg_pad_y = bg_peak[1]
        cv2.circle(tm_colored, (bg_pad_x, bg_pad_y), 5, (255, 0, 0), 2)  # 青
        cv2.putText(
            tm_colored,
            "Background",
            (bg_pad_x + 10, bg_pad_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
        )

    # 前景候補のピーク（緑）
    if len(foreground_peaks) > 0:
        for i, ((adj_x, adj_y), score, bg_diff) in enumerate(foreground_peaks[:3]):
            # パディング座標系に変換
            pad_x = adj_x + padding_size
            pad_y = adj_y + padding_size
            if 0 <= pad_x < tm_result.shape[1] and 0 <= pad_y < tm_result.shape[0]:
                cv2.circle(tm_colored, (pad_x, pad_y), 5, (0, 255, 0), 2)  # 緑
                cv2.putText(
                    tm_colored,
                    f"Foreground{i+1}",
                    (pad_x + 10, pad_y + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )

    cv2.imshow("Template Matching Result (Image2 in Padded Image1)", tm_colored)

    # 結果を画像上に描画
    result_image = image2.copy()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)

    # 背景の位置を描画（青）
    if len(peaks) > 0:
        bg_adj_x, bg_adj_y = peaks[0][0]
        if 0 <= bg_adj_x < w1 and 0 <= bg_adj_y < h1:
            cv2.rectangle(
                result_image,
                (bg_adj_x, bg_adj_y),
                (bg_adj_x + w2, bg_adj_y + h2),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                result_image,
                f"Background: ({detected_background_shift[0]:.1f}, {detected_background_shift[1]:.1f})",
                (bg_adj_x, bg_adj_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )

    # 前景候補の位置を描画（緑）
    if (
        detected_foreground_shift is not None
        and detected_x is not None
        and detected_y is not None
    ):
        if 0 <= detected_x < w1 and 0 <= detected_y < h1:
            cv2.rectangle(
                result_image,
                (detected_x, detected_y),
                (detected_x + w2, detected_y + h2),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                result_image,
                f"Foreground: {detected_foreground_shift}",
                (detected_x, detected_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

    # 期待される前景の位置を描画（検証用、赤）
    cv2.rectangle(
        result_image,
        (obj1_x, obj1_y),
        (obj1_x + object_size, obj1_y + object_size),
        (0, 0, 255),
        2,
    )
    cv2.putText(
        result_image,
        f"Expected: {expected_foreground_shift}",
        (obj1_x, obj1_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 255),
        1,
    )
    cv2.imshow("Detection Result", result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
