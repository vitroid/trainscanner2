#!/usr/bin/env python3
"""
ts2dumpファイルを可視化するスクリプト

【可視化内容】
- z軸: frame_index（時刻）
- x軸: delta_x
- y軸: delta_y
- 各pathを三次元で描画
- マーカーは円で、半径はmatch_scoreに比例

【使用方法】
python ts2undump.py <ts2dump_file>
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import argparse


def load_ts2dump(filename):
    """ts2dumpファイルを読み込む"""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_paths(ts2dump_data, min_score=0.0, max_marker_size=100, min_marker_size=5):
    """
    ts2dumpデータを三次元プロットで可視化
    
    Args:
        ts2dump_data: ts2dumpファイルから読み込んだデータ
        min_score: 表示する最小のmatch_score（フィルタリング用）
        max_marker_size: 最大マーカーサイズ
        min_marker_size: 最小マーカーサイズ
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # カラーマップを生成（pathごとに異なる色）
    num_paths = len(ts2dump_data)
    colors = plt.cm.tab20(np.linspace(0, 1, num_paths))

    # すべてのmatch_scoreを収集してスケーリング範囲を決定
    all_scores = []
    for path_id, path_data in ts2dump_data.items():
        for item in path_data.get("history", []):
            all_scores.append(item.get("match_score", 0.0))
    
    if not all_scores:
        print("警告: データが見つかりませんでした")
        return
    
    min_score_val = min(all_scores)
    max_score_val = max(all_scores)
    score_range = max_score_val - min_score_val if max_score_val > min_score_val else 1.0

    # 各pathを描画
    for idx, (path_id, path_data) in enumerate(ts2dump_data.items()):
        history = path_data.get("history", [])
        if not history:
            continue

        # データを抽出
        frame_indices = []
        delta_xs = []
        delta_ys = []
        match_scores = []

        for item in history:
            frame_idx = item.get("frame_index", 0)
            delta_x = item.get("delta_x", 0.0)
            delta_y = item.get("delta_y", 0.0)
            match_score = item.get("match_score", 0.0)

            # 最小スコアでフィルタリング
            if match_score < min_score:
                continue

            frame_indices.append(frame_idx)
            delta_xs.append(delta_x)
            delta_ys.append(delta_y)
            match_scores.append(match_score)

        if not frame_indices:
            continue

        # マーカーサイズをmatch_scoreに比例して計算
        marker_sizes = [
            min_marker_size
            + (max_marker_size - min_marker_size)
            * ((score - min_score_val) / score_range)
            for score in match_scores
        ]

        # 色を選択
        color = colors[idx % len(colors)]

        # パスの線を描画
        ax.plot(
            delta_xs,
            delta_ys,
            frame_indices,
            color=color,
            alpha=0.3,
            linewidth=1,
            label=f"Path {path_id}",
        )

        # マーカーを描画（match_scoreに比例したサイズ）
        ax.scatter(
            delta_xs,
            delta_ys,
            frame_indices,
            s=marker_sizes,
            c=[color],
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )

    # 軸ラベルを設定
    ax.set_xlabel("delta_x", fontsize=12)
    ax.set_ylabel("delta_y", fontsize=12)
    ax.set_zlabel("frame_index (時刻)", fontsize=12)
    ax.set_title("Path Visualization (マーカーサイズ = match_score)", fontsize=14)

    # 凡例を表示（pathが多い場合は非表示推奨）
    if num_paths <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # グリッドを表示
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="ts2dumpファイルを三次元プロットで可視化"
    )
    parser.add_argument(
        "ts2dump_file", type=str, help="ts2dumpファイルのパス"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="表示する最小のmatch_score（デフォルト: 0.0）",
    )
    parser.add_argument(
        "--max-marker-size",
        type=float,
        default=100,
        help="最大マーカーサイズ（デフォルト: 100）",
    )
    parser.add_argument(
        "--min-marker-size",
        type=float,
        default=5,
        help="最小マーカーサイズ（デフォルト: 5）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力ファイル名（指定しない場合は表示のみ）",
    )

    args = parser.parse_args()

    # ts2dumpファイルを読み込む
    try:
        ts2dump_data = load_ts2dump(args.ts2dump_file)
        print(f"読み込んだpath数: {len(ts2dump_data)}")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {args.ts2dump_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"エラー: JSONの解析に失敗しました: {e}")
        sys.exit(1)

    # プロット
    fig, ax = plot_paths(
        ts2dump_data,
        min_score=args.min_score,
        max_marker_size=args.max_marker_size,
        min_marker_size=args.min_marker_size,
    )

    # 出力または表示
    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"画像を保存しました: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()








