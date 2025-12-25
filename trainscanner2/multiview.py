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

try:
    from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QScrollArea,
        QLabel,
        QPushButton,
        QMessageBox,
        QProgressBar,
        QProgressDialog,
        QSplitter,
        QTabWidget,
    )
    from PyQt6.QtGui import QImage, QPixmap, QShortcut, QKeySequence
    from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

    PYQT6_AVAILABLE = True
except ImportError as e:
    PYQT6_AVAILABLE = False
except Exception as e:
    PYQT6_AVAILABLE = False

from trainscanner2.imagestrips import ImageStrips


def create_styled_message_box(parent, icon, title, text):
    """
    スタイルが適用されたQMessageBoxを作成する
    
    アクティブパネルのスタイルがダイアログに影響しないように、
    明示的にスタイルを設定する。
    """
    msg_box = QMessageBox(parent)
    msg_box.setIcon(icon)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    
    # ダイアログのスタイルを設定（白背景、青ボタン）
    msg_box.setStyleSheet("""
        QMessageBox {
            background-color: #ffffff;
        }
        QMessageBox QPushButton {
            font-size: 11px;
            padding: 6px 12px;
            border: 2px solid #3498db;
            border-radius: 6px;
            background-color: #3498db;
            color: #ffffff;
            font-weight: bold;
            min-height: 24px;
            min-width: 80px;
        }
        QMessageBox QPushButton:hover {
            background-color: #2980b9;
            border-color: #2980b9;
        }
        QMessageBox QPushButton:pressed {
            background-color: #21618c;
            border-color: #21618c;
        }
        QMessageBox QLabel {
            color: #333333;
        }
    """)
    
    return msg_box


class ProgressButton(QPushButton):
    """プログレス表示機能付きボタン"""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.progress = 0
        self.is_processing = False
        self.is_completed = False
        # 初期スタイルを設定
        self.update_style()

    def set_progress(self, progress):
        """進捗を設定（0-100）"""
        self.progress = max(0, min(100, progress))
        self.update_style()

    def set_processing(self, is_processing, text=None):
        """処理状態を設定"""
        self.is_processing = is_processing
        if text:
            self.setText(text)
        self.update_style()

    def set_completed(self, is_completed, text=None):
        """完了状態を設定"""
        self.is_completed = is_completed
        if text:
            self.setText(text)
        self.update_style()

    def update_style(self):
        """スタイルを更新"""
        if self.is_processing:
            # プログレスバー風のスタイル（左から右へバーが伸びる、境界明確）
            progress_percent = self.progress / 100.0
            button_name = self.objectName() if self.objectName() else ""
            if button_name:
                style = f"""
                QPushButton#{button_name} {{
                    font-size: 11px;
                    padding: 6px 12px;
                    border: 2px solid #27ae60;
                    border-radius: 6px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #27ae60, stop:{progress_percent - 0.001} #27ae60, 
                        stop:{progress_percent} #ecf0f1, stop:1 #ecf0f1);
                    color: #ffffff;
                    font-weight: bold;
                    min-height: 24px;
                }}
                QPushButton#{button_name}:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #229954, stop:{progress_percent - 0.001} #229954, 
                        stop:{progress_percent} #d5dbdb, stop:1 #d5dbdb);
                    border-color: #229954;
                }}
                """
            else:
                style = f"""
                QPushButton {{
                    font-size: 11px;
                    padding: 6px 12px;
                    border: 2px solid #27ae60;
                    border-radius: 6px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #27ae60, stop:{progress_percent - 0.001} #27ae60, 
                        stop:{progress_percent} #ecf0f1, stop:1 #ecf0f1);
                    color: #ffffff;
                    font-weight: bold;
                    min-height: 24px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #229954, stop:{progress_percent - 0.001} #229954, 
                        stop:{progress_percent} #d5dbdb, stop:1 #d5dbdb);
                    border-color: #229954;
                }}
                """
            self.setStyleSheet(style)
        elif self.is_completed:
            # 完了後のスタイル（グレー背景、無効化）
            button_name = self.objectName() if self.objectName() else ""
            if button_name:
                style = f"""
                QPushButton#{button_name} {{
                    font-size: 11px;
                    padding: 6px 12px;
                    border: 2px solid #95a5a6;
                    border-radius: 6px;
                    background-color: #95a5a6;
                    color: #ffffff;
                    font-weight: bold;
                    min-height: 24px;
                }}
                QPushButton#{button_name}:hover {{
                    background-color: #7f8c8d;
                    border-color: #7f8c8d;
                }}
                QPushButton#{button_name}:disabled {{
                    background-color: #bdc3c7;
                    border-color: #95a5a6;
                    color: #ffffff;
                }}
                """
            else:
                style = """
                QPushButton {
                    font-size: 11px;
                    padding: 6px 12px;
                    border: 2px solid #95a5a6;
                    border-radius: 6px;
                    background-color: #95a5a6;
                    color: #ffffff;
                    font-weight: bold;
                    min-height: 24px;
                }
                QPushButton:hover {
                    background-color: #7f8c8d;
                    border-color: #7f8c8d;
                }
                QPushButton:disabled {
                    background-color: #bdc3c7;
                    border-color: #95a5a6;
                    color: #ffffff;
                }
                """
            self.setStyleSheet(style)
        else:
            # 通常のスタイル（ボタンらしい見た目）
            button_name = self.objectName() if self.objectName() else ""
            if button_name:
                # objectNameが設定されている場合は、より具体的なセレクタを使用
                style = f"""
                QPushButton#{button_name} {{
                    font-size: 11px;
                    padding: 6px 12px;
                    border: 2px solid #3498db;
                    border-radius: 6px;
                    background-color: #3498db;
                    color: #ffffff;
                    font-weight: bold;
                    min-height: 24px;
                }}
                QPushButton#{button_name}:hover {{
                    background-color: #2980b9;
                    border-color: #2980b9;
                }}
                QPushButton#{button_name}:pressed {{
                    background-color: #21618c;
                    border-color: #21618c;
                }}
                QPushButton#{button_name}:disabled {{
                    background-color: #bdc3c7;
                    border-color: #95a5a6;
                    color: #7f8c8d;
                }}
                """
            else:
                # objectNameが設定されていない場合は通常のセレクタ
                style = """
                QPushButton {
                    font-size: 11px;
                    padding: 6px 12px;
                    border: 2px solid #3498db;
                    border-radius: 6px;
                    background-color: #3498db;
                    color: #ffffff;
                    font-weight: bold;
                    min-height: 24px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                    border-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #21618c;
                    border-color: #21618c;
                }
                QPushButton:disabled {
                    background-color: #bdc3c7;
                    border-color: #95a5a6;
                    color: #7f8c8d;
                }
                """
            self.setStyleSheet(style)


class SaveWorker(QThread):
    """並列処理で画像保存を行うワーカークラス"""

    finished = pyqtSignal(str, bool)  # ファイルパス, 成功フラグ
    progress = pyqtSignal(int)  # 進捗（0-100）

    def __init__(self, render_one, base_path, is_hires=False):
        super().__init__()
        self.render_one = render_one
        self.base_path = base_path
        self.is_hires = is_hires

    def run(self):
        try:
            self.progress.emit(10)

            if self.is_hires:
                # 高精細保存
                hires_base_path = f"{self.base_path}_hires"

                # まず通常保存を実行
                self.render_one.save(base_path=self.base_path)
                self.progress.emit(30)

                # .tspos2ファイルのパス
                tspos2_path = f"{self.base_path}.tspos2"

                if not os.path.exists(tspos2_path):
                    raise FileNotFoundError(
                        f".tspos2ファイルが見つかりません: {tspos2_path}"
                    )

                # 高精細処理を実行
                from trainscanner2.stitch import stitch

                def update_progress(current, total):
                    percentage = int((current / total) * 60) + 30  # 30-90%の範囲
                    self.progress.emit(percentage)

                # stitch関数を呼び出し
                render = stitch(
                    tspos2file=tspos2_path,
                    verbose=False,
                    progress_callback=update_progress,
                )

                # 高解像度画像を保存
                render.save(base_path=hires_base_path)
                self.progress.emit(100)
                self.finished.emit(f"{hires_base_path}.png", True)
            else:
                # 通常保存
                self.render_one.save(base_path=self.base_path)
                self.progress.emit(100)
                self.finished.emit(f"{self.base_path}.png", True)

        except Exception as e:
            self.finished.emit(str(e), False)


class HiresWorker(QThread):
    """高精細画像生成専用のワーカークラス（並列処理用）"""

    finished = pyqtSignal(str, bool)  # ファイルパス, 成功フラグ
    progress = pyqtSignal(int)  # 進捗（0-100）
    status_update = pyqtSignal(str)  # ステータス更新

    def __init__(self, render_one, base_path, path_id):
        super().__init__()
        self.render_one = render_one
        self.base_path = base_path
        self.path_id = path_id

    def run(self):
        try:
            self.status_update.emit("通常画像を保存中...")
            self.progress.emit(10)

            # まず通常保存を実行
            self.render_one.save(base_path=self.base_path)
            self.progress.emit(20)

            # .tspos2ファイルのパス
            tspos2_path = f"{self.base_path}.tspos2"

            if not os.path.exists(tspos2_path):
                raise FileNotFoundError(
                    f".tspos2ファイルが見つかりません: {tspos2_path}"
                )

            self.status_update.emit("高精細処理を開始中...")
            self.progress.emit(25)

            # 高精細処理を実行
            from trainscanner2.stitch import stitch

            def update_progress(current, total):
                if total > 0:
                    # 25-90%の範囲で進捗を更新
                    percentage = int((current / total) * 65) + 25
                    self.progress.emit(percentage)
                    self.status_update.emit(f"フレーム {current}/{total} を処理中...")

            # stitch関数を呼び出し
            render = stitch(
                tspos2file=tspos2_path,
                verbose=False,
                progress_callback=update_progress,
            )

            self.status_update.emit("高解像度画像を保存中...")
            self.progress.emit(90)

            # 高解像度画像を保存
            hires_base_path = f"{self.base_path}_hires"
            render.save(base_path=hires_base_path)

            self.progress.emit(100)
            self.status_update.emit("完了")
            self.finished.emit(f"{hires_base_path}.png", True)

        except Exception as e:
            self.status_update.emit("エラー")
            self.finished.emit(str(e), False)


def cv2_to_qpixmap(cv_img):
    """OpenCV画像をQPixmapに変換"""
    if cv_img is None:
        return None

    # カラー画像の場合
    if len(cv_img.shape) == 3:
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            cv_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        ).rgbSwapped()
    else:
        # グレースケール画像の場合
        height, width = cv_img.shape
        bytes_per_line = width
        q_image = QImage(
            cv_img.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8
        )

    return QPixmap.fromImage(q_image)


class PathViewWidget(QWidget):
    """
    1つのPathを表示するウィジェット
    """

    def __init__(self, path_id: int, video_base: str = None, show_gaps: bool = False):
        if QApplication.instance() is None:
            raise RuntimeError("QApplication must be created before creating QWidget")
        super().__init__()
        self.path_id = path_id
        self.video_base = video_base
        self.show_gaps = show_gaps

        # 3. 縦にstackしたパネル (幅=コンテナ幅いっぱい、高さ=画像の高さ)
        # 灰色の枠で囲まれたパネル（背景は赤）
        self.active_panel_style = """
            QWidget {
                border: 1px solid #999999;
                border-radius: 8px;
                background-color: #ffcccc;
                margin: 5px;
                padding: 5px;
            }
            QWidget:hover {
                border: 1px solid #999999;
                background-color: #ffaaaa;
            }
        """
        self.inactive_panel_style = """
            QWidget {
                border: 1px solid #999999;
                border-radius: 8px;
                background-color: #f7f7f7;
                margin: 5px;
                padding: 5px;
            }
            QWidget:hover {
                border: 1px solid #999999;
                background-color: #f0f0f0;
            }
        """
        self.setStyleSheet(self.active_panel_style)

        # 4. 左:情報枠(固定幅) + 右:画像(横スクロール)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # 4a. 左側の情報枠 (固定幅、狭くていい)
        info_widget = QWidget()
        info_widget.setFixedWidth(150)  # 固定幅150px
        info_widget.setStyleSheet("background-color: white;")  # 常に白背景
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(5, 5, 5, 5)
        info_layout.setSpacing(5)

        # Path番号
        self.base_title = f"Path {path_id}"
        self.title_label = QLabel(self.base_title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_active_style = "font-weight: bold; font-size: 14px; color: #2c3e50; padding: 3px; border: none;"
        self.title_inactive_style = "font-weight: bold; font-size: 14px; color: #7f8c8d; padding: 3px; border: none;"
        self.title_label.setStyleSheet(self.title_active_style)
        info_layout.addWidget(self.title_label)

        # 品質表示
        self.score_label = QLabel("Score: 0.000")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet(
            "font-size: 11px; color: #666; padding: 2px; border: none;"
        )
        info_layout.addWidget(self.score_label)

        # 最初のframe番号表示
        self.frame_label = QLabel("Frame: -")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setStyleSheet(
            "font-size: 10px; color: #888; padding: 2px; border: none;"
        )
        info_layout.addWidget(self.frame_label)

        # ボタン
        self.save_button = ProgressButton("保存")
        self.save_button.setObjectName("saveButton")  # objectNameを設定してスタイルを確実に適用
        self.save_button.clicked.connect(self.save_image)
        info_layout.addWidget(self.save_button)

        self.save_hires_button = ProgressButton("高精細保存")
        self.save_hires_button.setObjectName("saveHiresButton")  # objectNameを設定してスタイルを確実に適用
        self.save_hires_button.clicked.connect(self.save_hires_image)
        info_layout.addWidget(self.save_hires_button)

        main_layout.addWidget(info_widget)

        # 4b. 右側の画像 (横スクロール対応、残りの幅を全て使用)
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.image_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.image_scroll_area.setWidgetResizable(False)  # 画像サイズを保持

        # 画像表示ウィジェット
        try:
            from trainscanner2.widget import ImageStripsWidget

            self.image_widget = ImageStripsWidget()
        except Exception as e:
            self.image_widget = QLabel("画像表示エリア")
            self.image_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_widget.setStyleSheet(
                "border: 1px solid gray; background-color: #f0f0f0;"
            )

        self.image_scroll_area.setWidget(self.image_widget)
        main_layout.addWidget(
            self.image_scroll_area, 1
        )  # stretch factor = 1 で残りの幅を全て使用

        # 横スクロールの自動追従制御
        self._auto_scroll = True
        self._scrollbar_updating = False
        self._horizontal_scrollbar = self.image_scroll_area.horizontalScrollBar()
        if self._horizontal_scrollbar is not None:
            self._horizontal_scrollbar.valueChanged.connect(
                self._on_horizontal_scroll_changed
            )

        # 共通スタイル設定（QPushButtonは除外して、ボタン個別のスタイルを適用）
        self.common_style = """
            QLabel {
                color: #333;
                font-family: Arial, sans-serif;
                border: none;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """
        self._current_panel_style = self.active_panel_style
        self._apply_panel_style()

        # データ
        self.render_one = None
        self.current_image = None
        self.last_update_time = 0

        # 保存ワーカーの管理
        self.save_worker = None
        self.hires_worker = None

        # MotionDetector.pathsに含まれるかどうか
        self.is_active = True

    def _apply_panel_style(self):
        """パネルのスタイルを再適用"""
        combined_style = self._current_panel_style + "\n" + self.common_style
        self.setStyleSheet(combined_style)

    def set_active(self, active: bool):
        """MotionDetector.pathsに含まれているかどうかを更新"""
        self.is_active = active
        if active:
            self.title_label.setText(self.base_title)
            self.title_label.setStyleSheet(self.title_active_style)
            self._current_panel_style = self.active_panel_style
        else:
            self.title_label.setText(f"{self.base_title} (停止)")
            self.title_label.setStyleSheet(self.title_inactive_style)
            self._current_panel_style = self.inactive_panel_style

        self._apply_panel_style()

    def update_image(self, render_one, force: bool = False):
        """画像を更新"""
        if render_one is None:
            return

        self.render_one = render_one

        # 更新頻度制限（1秒に1回）
        current_time = time.time()
        if not force and current_time - self.last_update_time < 1.0:
            return

        self.last_update_time = current_time

        # 画像を取得（ImageStripsWidgetまたはQLabelでの表示）
        try:
            if hasattr(render_one, "canvas") and render_one.canvas is not None:
                # ImageStripsWidgetを使用している場合
                if hasattr(self.image_widget, "set_imagestrips"):
                    self.image_widget.set_imagestrips(render_one.canvas)
                    self.current_image = render_one.canvas
                else:
                    # QLabelを使用している場合（フォールバック）
                    if hasattr(render_one.canvas, "get_image"):
                        img = render_one.canvas.get_image()
                        if img is not None:
                            pixmap = cv2_to_qpixmap(img)
                            if pixmap:
                                # 画像を適切なサイズにスケール
                                scaled_pixmap = pixmap.scaled(
                                    400,
                                    300,
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation,
                                )
                                self.image_widget.setPixmap(scaled_pixmap)
                                self.image_widget.setText("")  # テキストをクリア
                                self.current_image = img
                            else:
                                self.image_widget.setText("画像読み込みエラー")
                        else:
                            self.image_widget.setText("画像なし")
                    else:
                        self.image_widget.setText("サポートされていない画像形式")
            else:
                if hasattr(self.image_widget, "set_imagestrips"):
                    # ImageStripsWidgetの場合は何もしない（空の状態を維持）
                    pass
                else:
                    self.image_widget.setText("キャンバスなし")
        except Exception as e:
            # 画像更新エラーを無視（ログに記録しない）
            if hasattr(self.image_widget, "set_imagestrips"):
                # ImageStripsWidgetの場合はエラーを無視
                pass
            else:
                self.image_widget.setText(f"更新エラー: {str(e)[:50]}")

        # 品質表示を更新
        if hasattr(render_one, "score"):
            score = render_one.score
            if score > 0:
                self.score_label.setText(f"Score: {score:.3f}")

                # 品質に応じて色を変更
                if score > 0.7:
                    self.score_label.setStyleSheet(
                        "font-size: 12px; color: #4CAF50; font-weight: bold;"
                    )
                elif score > 0.4:
                    self.score_label.setStyleSheet(
                        "font-size: 12px; color: #FF9800; font-weight: bold;"
                    )
                else:
                    self.score_label.setStyleSheet(
                        "font-size: 12px; color: #F44336; font-weight: bold;"
                    )
            else:
                self.score_label.setText("Score: 0.000")
                self.score_label.setStyleSheet("font-size: 12px; color: #666;")

        # 横スクロールを右端に自動追従
        if self._horizontal_scrollbar is not None:
            maximum = self._horizontal_scrollbar.maximum()
            if self._auto_scroll and maximum > 0:
                self._scrollbar_updating = True
                self._horizontal_scrollbar.setValue(maximum)
                self._scrollbar_updating = False
            elif maximum == 0:
                # 横スクロールが不要な場合は次回に備えて自動追従を有効化
                self._auto_scroll = True

        # フレーム番号を更新（stitch中は現在のフレーム、完了時は最終フレーム）
        if hasattr(render_one, "pathitem_history") and render_one.pathitem_history:
            # 最新のフレーム番号を取得
            latest_pathitem = render_one.pathitem_history[-1]
            if hasattr(latest_pathitem, "frame_index"):
                frame_index = latest_pathitem.frame_index
                # stitchが終わった場合は「最終フレーム」、進行中は「現在フレーム」と表示
                if hasattr(render_one, "alive") and not render_one.alive:
                    self.frame_label.setText(f"最終フレーム: {frame_index}")
                else:
                    self.frame_label.setText(f"現在フレーム: {frame_index}")
            else:
                self.frame_label.setText("Frame: -")
        else:
            self.frame_label.setText("Frame: -")

        # 画像ウィジェットのサイズを調整（canvasが存在する場合は常に実行）
        if hasattr(render_one, "canvas") and render_one.canvas is not None:
            self._adjust_image_widget_size()

        # ボタンの状態を更新
        self._update_button_states()

    def _is_rendering_complete(self) -> bool:
        """作画が完了しているかチェック"""
        if not self.render_one:
            return False

        # alive=False の場合は作画完了（long missedに至った）
        if hasattr(self.render_one, "alive") and not self.render_one.alive:
            return True

        # canvasが初期化されていない場合は作画未完了
        if not hasattr(self.render_one, "canvas") or not self.render_one.canvas:
            return False

        # 品質が設定されていない場合は作画未完了
        if not hasattr(self.render_one, "score") or self.render_one.score is None:
            return False

        # 品質が0の場合は作画未完了
        if self.render_one.score <= 0:
            return False

        # 画像が存在する場合は作画完了とみなす
        try:
            if hasattr(self.render_one.canvas, "get_image"):
                combined_image = self.render_one.canvas.get_image()
                if combined_image is not None:
                    # 画像サイズが妥当かチェック
                    height, width = combined_image.shape[:2]
                    return height > 0 and width > 0
            elif (
                hasattr(self.render_one.canvas, "images")
                and self.render_one.canvas.images
            ):
                # 画像リストが存在し、空でない場合
                return len(self.render_one.canvas.images) > 0
        except Exception:
            pass

        return False

    def _update_button_states(self):
        """ボタンの有効/無効状態を更新"""
        is_complete = self._is_rendering_complete()

        # 作画完了時のみボタンを有効化
        self.save_button.setEnabled(is_complete)
        self.save_hires_button.setEnabled(is_complete)

    def _on_horizontal_scroll_changed(self, value: int):
        """ユーザーの横スクロール操作を検知して自動追従を制御"""
        if self._horizontal_scrollbar is None:
            return

        if self._scrollbar_updating:
            return

        maximum = self._horizontal_scrollbar.maximum()
        if maximum == 0:
            self._auto_scroll = True
            return

        if value < maximum:
            # 右端未満ならユーザー操作と判断し自動追従を一時停止
            self._auto_scroll = False
        else:
            # 右端まで戻ったら自動追従を再開
            self._auto_scroll = True

    def _adjust_image_widget_size(self):
        """画像ウィジェットのサイズを調整"""
        if not hasattr(self, "image_widget") or not self.image_widget:
            return

        # 画像の実際のサイズを取得
        image_width = 600  # デフォルト幅
        image_height = 200  # デフォルト高さ

        if hasattr(self.render_one, "canvas") and self.render_one.canvas:
            try:
                # get_imageメソッドで実際の画像を取得
                if hasattr(self.render_one.canvas, "get_image"):
                    combined_image = self.render_one.canvas.get_image()
                    if combined_image is not None and hasattr(combined_image, "shape"):
                        image_height, image_width = combined_image.shape[
                            :2
                        ]  # shape[0] = 高さ, shape[1] = 幅
                        # デバッグメッセージを削除
                elif (
                    hasattr(self.render_one.canvas, "images")
                    and self.render_one.canvas.images
                ):
                    # 最初の画像のサイズを取得
                    first_image = self.render_one.canvas.images[0]
                    if hasattr(first_image, "shape"):
                        image_height, image_width = first_image.shape[:2]
                        # デバッグメッセージを削除
            except Exception as e:
                # エラーメッセージを削除（サイズ取得失敗は無視）
                pass

        # 画像ウィジェットのサイズを設定
        self.image_widget.setMinimumSize(image_width, image_height)
        self.image_widget.resize(image_width, image_height)
        # デバッグメッセージを削除

    def save_image(self):
        """画像を並列処理で保存"""
        if self.render_one is None:
            QMessageBox.warning(self, "警告", "保存する画像がありません")
            return

        # 既に保存処理中の場合は無視
        if self.save_worker and self.save_worker.isRunning():
            return

        try:
            # ベースパスを決定
            if self.video_base:
                base_path = f"{self.video_base}_{self.path_id}"
            else:
                base_path = f"train_scan_{self.path_id}"

            # 保存ワーカーを作成・開始
            self.save_worker = SaveWorker(self.render_one, base_path, is_hires=False)
            self.save_worker.finished.connect(self._on_save_finished)
            self.save_worker.progress.connect(self._on_save_progress)
            self.save_worker.start()

            # ボタンを処理中状態に設定
            self.save_button.set_processing(True, "保存中...")
            self.save_button.setEnabled(False)

        except Exception as e:
            msg_box = create_styled_message_box(
                self, QMessageBox.Icon.Critical, "エラー", f"保存処理の開始に失敗しました:\n{e}"
            )
            msg_box.exec()

    def save_hires_image(self):
        """高精細画像を並列処理で保存"""
        if self.render_one is None:
            msg_box = create_styled_message_box(
                self, QMessageBox.Icon.Warning, "警告", "保存する画像がありません"
            )
            msg_box.exec()
            return

        # 既に保存処理中の場合は無視
        if self.hires_worker and self.hires_worker.isRunning():
            return

        try:
            # ベースパスを決定
            if self.video_base:
                base_path = f"{self.video_base}_{self.path_id}"
            else:
                base_path = f"train_scan_{self.path_id}"

            # 高精細ワーカーを作成・開始
            self.hires_worker = HiresWorker(self.render_one, base_path, self.path_id)
            self.hires_worker.finished.connect(self._on_hires_finished)
            self.hires_worker.progress.connect(self._on_hires_progress)
            self.hires_worker.status_update.connect(self._on_hires_status_update)
            self.hires_worker.start()

            # ボタンを処理中状態に設定
            self.save_hires_button.set_processing(True, "高精細保存中...")
            self.save_hires_button.setEnabled(False)

            # 通常保存ボタンも処理中状態に設定（高精細保存では通常保存も実行される）
            self.save_button.set_processing(True, "通常保存中...")
            self.save_button.setEnabled(False)

        except Exception as e:
            msg_box = create_styled_message_box(
                self, QMessageBox.Icon.Critical, "エラー", f"高精細保存処理の開始に失敗しました:\n{e}"
            )
            msg_box.exec()

    def _on_save_finished(self, result, success):
        """通常保存完了時の処理"""
        if not success:
            msg_box = create_styled_message_box(
                self, QMessageBox.Icon.Critical, "エラー", f"画像の保存に失敗しました:\n{result}"
            )
            msg_box.exec()
            # エラー時は通常状態に戻す
            self.save_button.set_processing(False, "保存")
        else:
            # 成功時は完了状態に設定
            self.save_button.set_completed(True, "保存完了")
            self.save_button.setEnabled(False)  # 完了後は無効化
        self._update_button_states()

        # ワーカーをクリーンアップ
        if self.save_worker:
            self.save_worker.deleteLater()
            self.save_worker = None

    def _on_hires_finished(self, result, success):
        """高精細保存完了時の処理"""
        if not success:
            msg_box = create_styled_message_box(
                self, QMessageBox.Icon.Critical, "エラー", f"高精細画像の保存に失敗しました:\n{result}"
            )
            msg_box.exec()
            # エラー時は通常状態に戻す
            self.save_hires_button.set_processing(False, "高精細保存")
            self.save_button.set_processing(False, "保存")
        else:
            # 成功時は完了状態に設定
            self.save_hires_button.set_completed(True, "高精細保存完了")
            self.save_button.set_completed(True, "保存完了")
            # 完了後は無効化
            self.save_hires_button.setEnabled(False)
            self.save_button.setEnabled(False)
        self._update_button_states()

        # ワーカーをクリーンアップ
        if self.hires_worker:
            self.hires_worker.deleteLater()
            self.hires_worker = None

    def _on_save_progress(self, value):
        """通常保存の進捗更新"""
        self.save_button.set_progress(value)

    def _on_hires_progress(self, value):
        """高精細保存の進捗更新"""
        self.save_hires_button.set_progress(value)

        # 通常保存の進捗も更新（高精細保存では通常保存も実行される）
        if value <= 20:  # 通常保存の段階
            self.save_button.set_progress(value * 5)  # 0-100%に変換
        else:
            self.save_button.set_progress(100)  # 通常保存完了

    def _on_hires_status_update(self, status):
        """高精細保存のステータス更新"""
        # ボタンのテキストを更新
        self.save_hires_button.set_processing(True, f"高精細保存中... ({status})")


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
        self.setMinimumSize(1200, 800)

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

        # パネルコンテナの初期サイズを設定（幅はウィンドウ幅いっぱい）
        self.panels_container.setMinimumSize(800, 0)  # 最小幅800px

        # スクロールエリアで縦スクロールを実現
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.panels_container)
        self.scroll_area.setWidgetResizable(
            False
        )  # ウィジェットサイズを保持（コンテンツサイズに合わせる）
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )  # 横スクロールは不要
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )  # 縦スクロールは必要に応じて

        # スクロールエリアをメインレイアウトに追加（ウィンドウサイズに固定）
        main_layout.addWidget(self.scroll_area)

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

        # パネルコンテナのサイズを調整
        self._adjust_panels_container_size()

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

        # パネルコンテナのサイズを調整
        self._adjust_panels_container_size()

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

        # パネルコンテナのサイズを再調整
        self._adjust_panels_container_size()

    def _adjust_panels_container_size(self):
        """パネルコンテナのサイズを調整（コンテンツに合わせて）"""
        if not self.path_widgets:
            # パネルがない場合は最小サイズ
            self.panels_container.setMinimumSize(0, 0)
            return

        # 各パネルの高さを合計
        total_height = 20  # マージン
        for path_widget in self.path_widgets.values():
            if path_widget.isVisible():
                # パネルの実際の高さを取得
                widget_height = path_widget.sizeHint().height()
                if widget_height <= 0:
                    widget_height = 200  # デフォルト高さ
                total_height += widget_height + 10  # パネル高さ + スペース

        # パネルコンテナのサイズを設定（幅はウィンドウ幅いっぱい）
        # スクロールエリアの幅を取得
        scroll_area_width = self.scroll_area.width()
        if scroll_area_width <= 0:
            scroll_area_width = 800  # デフォルト幅

        self.panels_container.setMinimumSize(scroll_area_width, total_height)
        self.panels_container.resize(scroll_area_width, total_height)

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
        # パネルコンテナのサイズを調整
        self._adjust_panels_container_size()

    def closeEvent(self, event):
        """ウィンドウが閉じられるとき"""
        self.update_timer.stop()
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
