"""
複数の画像を1つのウィンドウ内に並べて表示するマルチビューウィンドウ

複数のPathを1つのウィンドウ内にタイル状に配置して表示し、
各Pathの進捗を同時に確認できるようにする。
"""

import sys
import time
import os
import json
import cv2
from logging import getLogger
from typing import Dict, Optional, Tuple

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QLabel,
)
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtCore import Qt, QTimer

from trainscanner2.imagestrips import ImageStrips
from trainscanner2.pathview import PathViewWidget, cv2_to_qpixmap


class MultiViewWindow(QMainWindow):
    """
    複数のPathを1つのウィンドウ内に並べて表示するメインウィンドウ
    """

    def __init__(
        self, video_base: str = None, show_gaps: bool = False, show_buttons: bool = True
    ):
        super().__init__()
        self.video_base = video_base
        self.show_gaps = show_gaps
        self.show_buttons = show_buttons

        self.path_widgets: Dict[int, PathViewWidget] = {}
        self.renderers: Dict[int, object] = {}  # Render_oneインスタンス
        self.active_path_ids = set()
        self.logger = getLogger(__name__)

        # ウィンドウタイトルをビデオファイル名のbasenameに設定
        if video_base:
            video_basename = os.path.basename(video_base)
            self.setWindowTitle(f"Train Scanner - {video_basename}")
        else:
            self.setWindowTitle("Train Scanner - Multi View")
        # スクリーンサイズに合わせて初期サイズを調整
        screen_geo = QApplication.primaryScreen().availableGeometry()
        # 横幅は最大1200、高さは画面高さの80%程度（最大800）に抑える
        default_width = min(1200, int(screen_geo.width() * 0.9))
        default_height = min(800, int(screen_geo.height() * 0.8))
        self.resize(default_width, default_height)
        # 最小サイズも画面に合わせて緩和
        self.setMinimumSize(min(800, screen_geo.width()), min(400, screen_geo.height()))

        # 1. Window: ユーザーの操作で大きさが変わる
        # (QMainWindowが自動的に処理)

        # 2. ウィンドウ全体を覆う枠 (幅=ウィンドウ幅、縦=ウィンドウ高さ、コンテンツが収まらない場合は縦スクロール)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)  # マージンなし

        # 3. 縦にstackしたパネル用のコンテナ
        self.panels_container = QWidget()
        self.panels_layout = QVBoxLayout(self.panels_container)
        self.panels_layout.setSpacing(10)  # パネル間のスペース
        self.panels_layout.setContentsMargins(10, 10, 10, 10)  # パネル周りのマージン
        self.panels_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # スクロールエリアで縦スクロールを実現
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.panels_container)
        self.scroll_area.setWidgetResizable(
            True
        )  # ウィジェットサイズを保持（コンテンツサイズに合わせる）
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )  # 横スクロールは不要
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )  # 縦スクロールは必要に応じて

        # スクロールエリアをメインレイアウトに追加（ウィンドウサイズに固定）
        main_layout.addWidget(self.scroll_area)

        # プレビュー表示用ラベル（右下にフローティング）
        self.preview_label = QLabel(self)
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setFixedSize(240, 135)  # 16:9
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setText("Waiting for video...")
        self.preview_label.setStyleSheet(
            """
            #previewLabel {
                background-color: rgba(0, 0, 0, 150);
                border: 2px solid #555;
                border-radius: 5px;
                color: white;
                font-size: 10px;
            }
        """
        )
        self.preview_label.hide()

        # キーボードショートカット
        close_shortcut = QShortcut(QKeySequence.StandardKey.Close, self)
        close_shortcut.activated.connect(self.close)

        # 更新タイマー
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_all_paths)
        self.update_timer.start(1000)  # 1秒ごとに更新

    def add_path(self, path_id: int, render_one: object):
        """新しいPathを追加"""
        if path_id in self.path_widgets:
            return

        # render_oneがNoneの場合は追加しない
        if render_one is None:
            self.logger.info(f"Skipping path {path_id} - render_one is None")
            return

        # canvasが初期化されていない場合は追加しない
        if not hasattr(render_one, "canvas") or render_one.canvas is None:
            self.logger.info(f"Skipping path {path_id} - canvas not initialized")
            self.logger.info(f"render_one type: {type(render_one)}")
            self.logger.info(f"hasattr canvas: {hasattr(render_one, 'canvas')}")
            if hasattr(render_one, "canvas"):
                self.logger.info(f"canvas value: {render_one.canvas}")
            return

        # PathViewWidgetを作成
        path_widget = PathViewWidget(
            path_id=path_id, video_base=self.video_base, show_gaps=self.show_gaps
        )

        # ボタンの表示/非表示
        if not self.show_buttons:
            path_widget.save_button.setVisible(False)
            path_widget.save_hires_button.setVisible(False)

        path_widget.set_active(True)
        self.path_widgets[path_id] = path_widget
        self.renderers[path_id] = render_one
        self.active_path_ids.add(path_id)

        # 縦レイアウトに追加
        self._add_to_vertical(path_widget, path_id)

        # 品質順で並べ替え
        self._sort_panels_by_score()

        self.logger.info(f"Successfully added path {path_id} to MultiViewWindow")
        self.logger.info(f"Total paths in window: {len(self.path_widgets)}")

    def set_active_paths(self, active_path_ids):
        """MotionDetector.pathsに含まれるPath ID一覧を設定"""
        active_ids = set(active_path_ids)
        self.active_path_ids = active_ids

        for path_id, path_widget in self.path_widgets.items():
            path_widget.set_active(path_id in active_ids)

    def mark_path_inactive(self, path_id: int):
        """特定のPathを非アクティブにマーク"""
        if path_id not in self.path_widgets:
            return

        self.active_path_ids.discard(path_id)
        self.path_widgets[path_id].set_active(False)

    def _add_to_vertical(self, path_widget: PathViewWidget, path_id: int):
        """縦レイアウトにウィジェットを追加"""
        # デバッグ情報を出力
        self.logger.info(f"Adding Path {path_id} to vertical layout")

        # 3. 縦にstackしたパネルに追加
        self.panels_layout.addWidget(path_widget)

        # ウィジェットを表示
        path_widget.setVisible(True)

        self.logger.info(
            f"Path {path_id} widget added to vertical layout and made visible"
        )

    def remove_path(self, path_id: int, reason: str = "不明"):
        """
        Pathを削除

        Args:
            path_id: 削除するPathのID
            reason: 削除理由（ログ表示用）
        """
        if path_id not in self.path_widgets:
            return

        path_widget = self.path_widgets[path_id]

        # 削除前に実行中のスレッドがあれば停止を待つ（クラッシュ防止）
        if (
            hasattr(path_widget, "save_worker")
            and path_widget.save_worker
            and path_widget.save_worker.isRunning()
        ):
            path_widget.save_worker.wait()
        if (
            hasattr(path_widget, "hires_worker")
            and path_widget.hires_worker
            and path_widget.hires_worker.isRunning()
        ):
            path_widget.hires_worker.wait()

        # 削除前に品質情報を取得
        score = None
        if path_id in self.renderers and hasattr(self.renderers[path_id], "score"):
            score = self.renderers[path_id].score

        # 3. 縦にstackしたパネルから削除
        self.panels_layout.removeWidget(path_widget)

        # ウィジェットを削除
        path_widget.deleteLater()
        del self.path_widgets[path_id]
        del self.renderers[path_id]
        self.active_path_ids.discard(path_id)

        # 完成した画像を削除する場合はログに残す
        score_info = f" (score: {score:.3f})" if score is not None else ""
        self.logger.info(
            f"Removed path {path_id} from MultiViewWindow: {reason}{score_info}"
        )

    def _sort_panels_by_score(self):
        """パネルを品質が高い順に並べ替える"""
        if not self.path_widgets:
            return

        # 品質でソート（高い順）
        sorted_widgets = sorted(
            self.path_widgets.values(),
            key=lambda widget: (
                widget.render_one.score
                if widget.render_one and hasattr(widget.render_one, "score")
                else 0.0
            ),
            reverse=True,
        )

        # レイアウトから全てのウィジェットを削除
        for widget in self.path_widgets.values():
            self.panels_layout.removeWidget(widget)

        # ソートされた順序で再配置
        for widget in sorted_widgets:
            self.panels_layout.addWidget(widget)

    def update_all_paths(self):
        """全てのPathを更新（更新が終わったパネルはスキップ）"""
        updated_count = 0
        for path_id, path_widget in self.path_widgets.items():
            render_one = self.renderers.get(path_id)
            if render_one is None:
                continue
            if not path_widget.is_active:
                continue

            path_widget.update_image(render_one)
            updated_count += 1

        # 更新があった場合のみ品質順で並べ替え
        if updated_count > 0:
            self._sort_panels_by_score()

    def has_paths(self) -> bool:
        """Pathが存在するかチェック"""
        return len(self.path_widgets) > 0

    def resizeEvent(self, event):
        """ウィンドウサイズが変更されたとき"""
        super().resizeEvent(event)
        # プレビューラベルの位置を調整
        if hasattr(self, "preview_label"):
            margin = 20
            self.preview_label.move(
                self.width() - self.preview_label.width() - margin,
                self.height() - self.preview_label.height() - margin,
            )

    def set_video_base(self, video_base: str):
        """ビデオのベース名を設定し、タイトルを更新する"""
        self.video_base = video_base
        if video_base:
            video_basename = os.path.basename(video_base)
            self.setWindowTitle(f"Train Scanner - {video_basename}")
        else:
            self.setWindowTitle("Train Scanner - Multi View")

    def update_preview(self, cv_img):
        """プレビュー画像を更新"""
        if cv_img is None:
            return

        # 小さくリサイズ（アスペクト比維持）
        h, w = cv_img.shape[:2]
        target_w = self.preview_label.width()
        target_h = self.preview_label.height()

        scale = min(target_w / w, target_h / h)
        small_img = cv2.resize(cv_img, (0, 0), fx=scale, fy=scale)

        pixmap = cv2_to_qpixmap(small_img)
        if pixmap:
            self.preview_label.setPixmap(pixmap)
            if self.preview_label.isHidden():
                self.preview_label.show()

    def clear_all_paths(self):
        """すべてのPath（パネル）を削除して初期状態に戻す"""
        self.logger.info("Clearing all paths from MultiViewWindow")

        # タイマーを一時停止
        self.update_timer.stop()

        # すべてのウィジェットを安全に削除（スレッド待機を含む）
        for path_id in list(self.path_widgets.keys()):
            self.remove_path(path_id, reason="一括クリア")

        # データのクリア（念のため）
        self.renderers.clear()
        self.active_path_ids.clear()

        # ウィンドウタイトルをリセット
        self.setWindowTitle("Train Scanner - Multi View")

        # タイマーを再開
        self.update_timer.start(1000)

    def closeEvent(self, event):
        """ウィンドウが閉じられるとき"""
        self.update_timer.stop()

        # 実行中のすべての保存ワーカーの終了を待つ（クラッシュ防止）
        active_workers = []
        for widget in self.path_widgets.values():
            if (
                hasattr(widget, "save_worker")
                and widget.save_worker
                and widget.save_worker.isRunning()
            ):
                active_workers.append(widget.save_worker)
            if (
                hasattr(widget, "hires_worker")
                and widget.hires_worker
                and widget.hires_worker.isRunning()
            ):
                active_workers.append(widget.hires_worker)

        if active_workers:
            self.logger.info(
                f"Waiting for {len(active_workers)} background workers to finish..."
            )
            for worker in active_workers:
                worker.wait()  # 終了までブロック

        event.accept()


class MultiViewManager:
    """
    マルチビューウィンドウを管理するマネージャー
    """

    def __init__(
        self, video_base: str = None, show_gaps: bool = False, show_buttons: bool = True
    ):
        self.video_base = video_base
        self.show_gaps = show_gaps
        self.show_buttons = show_buttons
        self.window = None
        self.app = None  # QApplicationインスタンスを保持
        self.logger = getLogger(__name__)

        # ウィンドウを即座に作成
        self.create_window()

    def create_window(self) -> MultiViewWindow:
        """マルチビューウィンドウを作成"""
        if self.window is None:
            # QApplicationが存在しない場合は作成
            if QApplication.instance() is None:
                self.app = QApplication(sys.argv)
                # QApplicationを完全に初期化するため、イベントを処理
                self.app.processEvents()
            else:
                self.app = QApplication.instance()

            try:
                self.window = MultiViewWindow(
                    video_base=self.video_base,
                    show_gaps=self.show_gaps,
                    show_buttons=self.show_buttons,
                )
                self.window.show()
            except Exception as e:
                import traceback

                traceback.print_exc()
                raise
        return self.window

    def clear_all_paths(self):
        """すべてのPathをクリア"""
        if self.window is not None:
            self.window.clear_all_paths()

    def update_preview(self, cv_img):
        """プレビューを更新"""
        if self.window is not None:
            self.window.update_preview(cv_img)

    def set_video_base(self, video_base: str):
        """ビデオのベース名を設定"""
        if self.window is not None:
            self.window.set_video_base(video_base)

    def add_path(self, path_id: int, render_one: object):
        """Pathを追加"""
        if self.window is None:
            self.create_window()
        self.window.add_path(path_id, render_one)

    def remove_path(self, path_id: int, reason: str = "不明"):
        """Pathを削除"""
        if self.window is not None:
            self.window.remove_path(path_id, reason=reason)

    def set_active_paths(self, active_path_ids):
        """MotionDetectorでアクティブなPath IDをMultiViewに伝える"""
        if self.window is not None:
            self.window.set_active_paths(active_path_ids)

    def mark_path_inactive(self, path_id: int):
        """特定のPathを非アクティブとしてマーキング"""
        if self.window is not None:
            self.window.mark_path_inactive(path_id)

    def has_paths(self) -> bool:
        """Pathが存在するかチェック"""
        return self.window is not None and self.window.has_paths()

    def wait_for_close(self):
        """ウィンドウが閉じられるまで待機"""
        if self.window is not None:
            while self.window.isVisible():
                QApplication.processEvents()
                time.sleep(0.1)
