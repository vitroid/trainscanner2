import os
import time
import cv2
from logging import getLogger
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QScrollArea,
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal


def cv2_to_qpixmap(cv_img):
    """OpenCVの画像(BGR)をQPixmapに変換する"""
    if cv_img is None:
        return None
    try:
        # カラー画像の場合
        if len(cv_img.shape) == 3:
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            # BGRからRGBに変換し、確実にコピーを作成する
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(
                rgb_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            ).copy()  # ここでコピーを取るのがクラッシュ防止に重要
            return QPixmap.fromImage(q_img)
        else:
            # グレースケール画像の場合
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(
                cv_img.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_Grayscale8,
            ).copy()
            return QPixmap.fromImage(q_img)
    except Exception:
        return None


def create_styled_message_box(parent, icon, title, text):
    """
    スタイルが適用されたQMessageBoxを作成する
    """
    msg_box = QMessageBox(parent)
    msg_box.setIcon(icon)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)

    # ダイアログのスタイルを設定（白背景、青ボタン）
    msg_box.setStyleSheet(
        """
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
    """
    )
    return msg_box


class ProgressButton(QPushButton):
    """プログレス表示機能付きボタン"""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.progress = 0
        self.is_processing = False
        self.is_completed = False
        self.update_style()

    def set_progress(self, progress):
        self.progress = max(0, min(100, progress))
        self.update_style()

    def set_processing(self, is_processing, text=None):
        self.is_processing = is_processing
        if text:
            self.setText(text)
        self.update_style()

    def set_completed(self, is_completed, text=None):
        self.is_completed = is_completed
        if text:
            self.setText(text)
        self.update_style()

    def update_style(self):
        if self.is_processing:
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
            button_name = self.objectName() if self.objectName() else ""
            if button_name:
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
    task_finished = pyqtSignal(str, bool)
    progress = pyqtSignal(int)

    def __init__(self, render_one, base_path, is_hires=False, parent=None):
        super().__init__(parent)
        self.render_one = render_one
        self.base_path = base_path
        self.is_hires = is_hires

    def run(self):
        try:
            self.progress.emit(10)
            if self.is_hires:
                hires_base_path = f"{self.base_path}_hires"
                self.render_one.save(base_path=self.base_path)
                self.progress.emit(30)
                tspos2_path = f"{self.base_path}.tspos2"
                if not os.path.exists(tspos2_path):
                    raise FileNotFoundError(
                        f".tspos2ファイルが見つかりません: {tspos2_path}"
                    )
                from trainscanner2.stitch import stitch

                def update_progress(current, total):
                    percentage = int((current / total) * 60) + 30
                    self.progress.emit(percentage)

                self.hires_render = stitch(
                    tspos2file=tspos2_path,
                    verbose=False,
                    progress_callback=update_progress,
                )
                self.hires_render.save(base_path=hires_base_path)
                self.progress.emit(100)
                self.msleep(50)
                self.hires_render = None
                self.msleep(50)
                self.task_finished.emit(f"{hires_base_path}.png", True)
            else:
                self.render_one.save(base_path=self.base_path)
                self.progress.emit(100)
                self.msleep(50)
                self.task_finished.emit(f"{self.base_path}.png", True)
        except Exception as e:
            self.hires_render = None
            self.task_finished.emit(str(e), False)


class HiresWorker(QThread):
    task_finished = pyqtSignal(str, bool)
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)

    def __init__(self, render_one, base_path, path_id, parent=None):
        super().__init__(parent)
        self.render_one = render_one
        self.base_path = base_path
        self.path_id = path_id

    def run(self):
        try:
            self.status_update.emit("通常画像を保存中...")
            self.progress.emit(10)
            self.render_one.save(base_path=self.base_path)
            self.progress.emit(20)
            tspos2_path = f"{self.base_path}.tspos2"
            if not os.path.exists(tspos2_path):
                raise FileNotFoundError(
                    f".tspos2ファイルが見つかりません: {tspos2_path}"
                )
            self.status_update.emit("高精細処理を開始中...")
            self.progress.emit(25)
            from trainscanner2.stitch import stitch

            def update_progress(current, total):
                if total > 0:
                    percentage = int((current / total) * 65) + 25
                    self.progress.emit(percentage)
                    self.status_update.emit(f"フレーム {current}/{total} を処理中...")

            self.hires_render = stitch(
                tspos2file=tspos2_path, verbose=False, progress_callback=update_progress
            )
            self.status_update.emit("高解像度画像を保存中...")
            self.progress.emit(90)
            hires_base_path = f"{self.base_path}_hires"
            self.hires_render.save(base_path=hires_base_path)
            self.progress.emit(100)
            self.status_update.emit("完了")
            self.msleep(100)
            self.msleep(50)
            self.hires_render = None
            self.msleep(50)
            self.task_finished.emit(f"{hires_base_path}.png", True)
        except Exception as e:
            self.hires_render = None
            self.task_finished.emit(str(e), False)


class PathViewWidget(QWidget):
    def __init__(self, path_id: int, video_base: str = None, show_gaps: bool = False):
        if QApplication.instance() is None:
            raise RuntimeError("QApplication must be created before creating QWidget")
        super().__init__()
        self.path_id = path_id
        self.video_base = video_base
        self.show_gaps = show_gaps

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

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        info_widget = QWidget()
        info_widget.setFixedWidth(150)
        info_widget.setStyleSheet("background-color: white;")
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(5, 5, 5, 5)
        info_layout.setSpacing(5)

        self.base_title = f"Path {path_id}"
        self.title_label = QLabel(self.base_title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_active_style = "font-weight: bold; font-size: 14px; color: #2c3e50; padding: 3px; border: none;"
        self.title_inactive_style = "font-weight: bold; font-size: 14px; color: #7f8c8d; padding: 3px; border: none;"
        self.title_label.setStyleSheet(self.title_active_style)
        info_layout.addWidget(self.title_label)

        self.score_label = QLabel("Score: 0.000")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet(
            "font-size: 11px; color: #666; padding: 2px; border: none;"
        )
        info_layout.addWidget(self.score_label)

        self.frame_label = QLabel("Frame: -")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setStyleSheet(
            "font-size: 10px; color: #888; padding: 2px; border: none;"
        )
        info_layout.addWidget(self.frame_label)

        self.save_button = ProgressButton("保存")
        self.save_button.setObjectName("saveButton")
        self.save_button.clicked.connect(self.save_image)
        info_layout.addWidget(self.save_button)

        self.save_hires_button = ProgressButton("高精細保存")
        self.save_hires_button.setObjectName("saveHiresButton")
        self.save_hires_button.clicked.connect(self.save_hires_image)
        info_layout.addWidget(self.save_hires_button)

        main_layout.addWidget(info_widget)

        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.image_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.image_scroll_area.setWidgetResizable(False)

        try:
            from trainscanner2.widget import ImageStripsWidget

            self.image_widget = ImageStripsWidget()
        except Exception:
            self.image_widget = QLabel("画像表示エリア")
            self.image_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_widget.setStyleSheet(
                "border: 1px solid gray; background-color: #f0f0f0;"
            )

        self.image_scroll_area.setWidget(self.image_widget)
        main_layout.addWidget(self.image_scroll_area, 1)

        self._auto_scroll = True
        self._scrollbar_updating = False
        self._horizontal_scrollbar = self.image_scroll_area.horizontalScrollBar()
        if self._horizontal_scrollbar is not None:
            self._horizontal_scrollbar.valueChanged.connect(
                self._on_horizontal_scroll_changed
            )

        self.common_style = """
            QLabel { color: #333; font-family: Arial, sans-serif; border: none; }
            QProgressBar { border: 1px solid #ccc; border-radius: 4px; text-align: center; background-color: #f0f0f0; }
            QProgressBar::chunk { background-color: #4CAF50; border-radius: 3px; }
        """
        self._current_panel_style = self.active_panel_style
        self._apply_panel_style()

        self.render_one = None
        self.current_image = None
        self.last_update_time = 0
        self.save_worker = None
        self.hires_worker = None
        self.is_active = True

    def _apply_panel_style(self):
        combined_style = self._current_panel_style + "\n" + self.common_style
        self.setStyleSheet(combined_style)

    def set_active(self, active: bool):
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
        if render_one is None:
            return
        self.render_one = render_one
        current_time = time.time()
        if not force and current_time - self.last_update_time < 1.0:
            return
        self.last_update_time = current_time

        try:
            if hasattr(render_one, "canvas") and render_one.canvas is not None:
                if hasattr(self.image_widget, "set_imagestrips"):
                    self.image_widget.set_imagestrips(render_one.canvas)
                    self.current_image = render_one.canvas
                else:
                    if hasattr(render_one.canvas, "get_image"):
                        img = render_one.canvas.get_image()
                        if img is not None:
                            pixmap = cv2_to_qpixmap(img)
                            if pixmap:
                                scaled_pixmap = pixmap.scaled(
                                    400,
                                    300,
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation,
                                )
                                self.image_widget.setPixmap(scaled_pixmap)
                                self.image_widget.setText("")
                                self.current_image = img
                            else:
                                self.image_widget.setText("画像読み込みエラー")
                        else:
                            self.image_widget.setText("画像なし")
                    else:
                        self.image_widget.setText("サポートされていない画像形式")
        except Exception as e:
            if not hasattr(self.image_widget, "set_imagestrips"):
                self.image_widget.setText(f"更新エラー: {str(e)[:50]}")

        if hasattr(render_one, "score"):
            score = render_one.score
            if score > 0:
                self.score_label.setText(f"Score: {score:.3f}")
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

        if self._horizontal_scrollbar is not None:
            maximum = self._horizontal_scrollbar.maximum()
            if self._auto_scroll and maximum > 0:
                self._scrollbar_updating = True
                self._horizontal_scrollbar.setValue(maximum)
                self._scrollbar_updating = False
            elif maximum == 0:
                self._auto_scroll = True

        if hasattr(render_one, "pathitem_history") and render_one.pathitem_history:
            latest_pathitem = render_one.pathitem_history[-1]
            if hasattr(latest_pathitem, "frame_index"):
                frame_index = latest_pathitem.frame_index
                if hasattr(render_one, "alive") and not render_one.alive:
                    self.frame_label.setText(f"最終フレーム: {frame_index}")
                else:
                    self.frame_label.setText(f"現在フレーム: {frame_index}")
            else:
                self.frame_label.setText("Frame: -")
        else:
            self.frame_label.setText("Frame: -")

        if hasattr(render_one, "canvas") and render_one.canvas is not None:
            self._adjust_image_widget_size()
        self._update_button_states()

    def _is_rendering_complete(self) -> bool:
        if not self.render_one:
            return False
        if hasattr(self.render_one, "alive") and not self.render_one.alive:
            return True
        if not hasattr(self.render_one, "canvas") or not self.render_one.canvas:
            return False
        if not hasattr(self.render_one, "score") or self.render_one.score is None:
            return False
        if self.render_one.score <= 0:
            return False
        try:
            if hasattr(self.render_one.canvas, "get_image"):
                combined_image = self.render_one.canvas.get_image()
                if combined_image is not None:
                    height, width = combined_image.shape[:2]
                    return height > 0 and width > 0
            elif (
                hasattr(self.render_one.canvas, "images")
                and self.render_one.canvas.images
            ):
                return len(self.render_one.canvas.images) > 0
        except Exception:
            pass
        return False

    def _update_button_states(self):
        is_complete = self._is_rendering_complete()
        self.save_button.setEnabled(is_complete)
        self.save_hires_button.setEnabled(is_complete)

    def _on_horizontal_scroll_changed(self, value: int):
        if self._horizontal_scrollbar is None:
            return
        if self._scrollbar_updating:
            return
        maximum = self._horizontal_scrollbar.maximum()
        if maximum == 0:
            self._auto_scroll = True
            return
        if value < maximum:
            self._auto_scroll = False
        else:
            self._auto_scroll = True

    def _adjust_image_widget_size(self):
        if not hasattr(self, "image_widget") or not self.image_widget:
            return
        image_width = 600
        image_height = 200
        if hasattr(self.render_one, "canvas") and self.render_one.canvas:
            try:
                if hasattr(self.render_one.canvas, "get_image"):
                    combined_image = self.render_one.canvas.get_image()
                    if combined_image is not None and hasattr(combined_image, "shape"):
                        image_height, image_width = combined_image.shape[:2]
                elif (
                    hasattr(self.render_one.canvas, "images")
                    and self.render_one.canvas.images
                ):
                    first_image = self.render_one.canvas.images[0]
                    if hasattr(first_image, "shape"):
                        image_height, image_width = first_image.shape[:2]
            except Exception:
                pass
        self.image_widget.setMinimumSize(image_width, image_height)
        self.image_widget.resize(image_width, image_height)

    def save_image(self):
        if self.render_one is None:
            QMessageBox.warning(self, "警告", "保存する画像がありません")
            return
        if self.save_worker and self.save_worker.isRunning():
            return
        try:
            if self.video_base:
                base_path = f"{self.video_base}_{self.path_id}"
            else:
                base_path = f"train_scan_{self.path_id}"
            self.save_worker = SaveWorker(
                self.render_one, base_path, is_hires=False, parent=self
            )
            self.save_worker.task_finished.connect(self._on_save_finished)
            self.save_worker.progress.connect(self._on_save_progress)
            self.save_worker.finished.connect(self.save_worker.deleteLater)
            self.save_worker.start()
            self.save_button.set_processing(True, "保存中...")
            self.save_button.setEnabled(False)
        except Exception as e:
            msg_box = create_styled_message_box(
                self,
                QMessageBox.Icon.Critical,
                "エラー",
                f"保存処理の開始に失敗しました:\n{e}",
            )
            msg_box.exec()

    def save_hires_image(self):
        if self.render_one is None:
            msg_box = create_styled_message_box(
                self, QMessageBox.Icon.Warning, "警告", "保存する画像がありません"
            )
            msg_box.exec()
            return
        if self.hires_worker and self.hires_worker.isRunning():
            return
        try:
            if self.video_base:
                base_path = f"{self.video_base}_{self.path_id}"
            else:
                base_path = f"train_scan_{self.path_id}"
            self.hires_worker = HiresWorker(
                self.render_one, base_path, self.path_id, parent=self
            )
            self.hires_worker.task_finished.connect(self._on_hires_finished)
            self.hires_worker.progress.connect(self._on_hires_progress)
            self.hires_worker.status_update.connect(self._on_hires_status_update)
            self.hires_worker.finished.connect(self.hires_worker.deleteLater)
            self.hires_worker.start()
            self.save_hires_button.set_processing(True, "高精細保存中...")
            self.save_hires_button.setEnabled(False)
            self.save_button.set_processing(True, "通常保存中...")
            self.save_button.setEnabled(False)
        except Exception as e:
            msg_box = create_styled_message_box(
                self,
                QMessageBox.Icon.Critical,
                "エラー",
                f"高精細保存処理の開始に失敗しました:\n{e}",
            )
            msg_box.exec()

    def _on_save_finished(self, result, success):
        if self.save_worker:
            try:
                self.save_worker.task_finished.disconnect(self._on_save_finished)
                self.save_worker.progress.disconnect(self._on_save_progress)
            except Exception:
                pass
        if not success:
            msg_box = create_styled_message_box(
                self,
                QMessageBox.Icon.Critical,
                "エラー",
                f"画像の保存に失敗しました:\n{result}",
            )
            msg_box.exec()
            self.save_button.set_processing(False, "保存")
        else:
            self.save_button.set_completed(True, "保存完了")
            self.save_button.setEnabled(False)
        self._update_button_states()
        if self.save_worker:
            self.save_worker = None

    def _on_hires_finished(self, result, success):
        if self.hires_worker:
            try:
                self.hires_worker.task_finished.disconnect(self._on_hires_finished)
                self.hires_worker.progress.disconnect(self._on_hires_progress)
                self.hires_worker.status_update.disconnect(self._on_hires_status_update)
            except Exception:
                pass
        if not success:
            msg_box = create_styled_message_box(
                self,
                QMessageBox.Icon.Critical,
                "エラー",
                f"高精細画像の保存に失敗しました:\n{result}",
            )
            msg_box.exec()
            self.save_hires_button.set_processing(False, "高精細保存")
            self.save_button.set_processing(False, "保存")
        else:
            self.save_hires_button.set_completed(True, "高精細保存完了")
            self.save_button.set_completed(True, "保存完了")
            self.save_hires_button.setEnabled(False)
            self.save_button.setEnabled(False)
        self._update_button_states()
        if self.hires_worker:
            self.hires_worker = None

    def _on_save_progress(self, value):
        self.save_button.set_progress(value)

    def _on_hires_progress(self, value):
        self.save_hires_button.set_progress(value)
        if value <= 20:
            self.save_button.set_progress(value * 5)
        else:
            self.save_button.set_progress(100)

    def _on_hires_status_update(self, status):
        self.save_hires_button.set_processing(True, f"高精細保存中... ({status})")
