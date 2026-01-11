"""
PyQt6ベースのウィンドウ管理

ImageWindowとWindowManagerクラスを提供し、複数のトレーンスキャン画像を
効率的に表示・管理する。
"""

import sys
import time
import os
import json
import cv2
from logging import getLogger

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QScrollArea,
    QLabel,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
    QScrollBar,
)
from PyQt6.QtGui import QImage, QPixmap, QShortcut, QKeySequence
from PyQt6.QtCore import Qt, QTimer

from trainscanner2.widget import ImageStripsWidget
from trainscanner2.imagestrips import ImageStrips


def cv2_to_qpixmap(cv_img):
    """OpenCVの画像(BGR)をQPixmapに変換する"""
    if cv_img is None:
        return None
    try:
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        # BGRからRGBに変換し、確実にコピーを作成する
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        q_img = QImage(
            rgb_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        ).copy()  # ここでコピーを取るのがクラッシュ防止に重要
        return QPixmap.fromImage(q_img)
    except Exception:
        return None


class ImageWindow(QMainWindow):
    """
    PyQt6を使ったスクロール可能な画像表示ウィンドウ

    【目的】
    - OpenCVのcv2.imshowの代替として、より高機能なGUIを提供
    - 画像が大きくなってもウィンドウサイズは固定（スクロールバーで表示）
    - 保存・閉じるボタンを提供
    - Command-W（Ctrl-W）でウィンドウを閉じられる

    【macOSでの表示問題の解決】
    - 更新頻度を1秒に1回に制限（last_update_time, pending_image）
    - repaint() + processEvents() で強制的に再描画
    - これにより、影だけでなく実際の内容が表示されるようになった
    """

    logger = getLogger(__name__)

    def __init__(
        self,
        window_id: int,
        video_base: str = None,
        close_callback=None,  # ウィンドウが閉じられたときにRenderに通知するコールバック
        render_one=None,  # Render_oneインスタンス（履歴保存用）
        show_gaps=False,  # デバッグ用: 短冊間に1px隙間を表示
        show_buttons=True,  # 保存・閉じるボタンを表示するか
        parent=None,
    ):
        super().__init__(parent)
        self.window_id = window_id
        self.video_base = video_base or "train_scan"
        self.close_callback = close_callback
        self.render_one = render_one  # stitching履歴にアクセスするため
        self.is_preview = window_id == -1  # プレビューウィンドウかどうか

        # ウィンドウタイトルをビデオファイル名のbasenameに設定
        if video_base:
            video_basename = os.path.basename(video_base)
            self.setWindowTitle(f"TrainScanner - {video_basename} (ID: {window_id})")
        else:
            self.setWindowTitle(f"TrainScanner - ID: {window_id}")

        # 画像管理
        self.current_image = None  # 現在表示中の画像
        self.last_update_time = 0  # 最後の更新時刻（更新頻度制限用）
        self.pending_image = None  # 保留中の画像（1秒以内の更新はここに保存）
        self.initial_size_set = (
            False  # プレビューウィンドウで初期サイズを設定したかどうか
        )

        # 最大ウィンドウサイズを設定（画面サイズの80%程度）
        self.setMaximumSize(1920, 1080)
        if not self.is_preview:
            # プレビューウィンドウでない場合のみ固定サイズで初期化
            self.resize(800, 600)

        # メインウィジェットとレイアウト
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # ImageStrips用の表示ウィジェット（仮想スクロール）
        self.imagestrips_widget = ImageStripsWidget(show_gaps=show_gaps)

        # カスタム横スクロールバー
        self.h_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.h_scrollbar.valueChanged.connect(self._on_scroll)
        self.h_scrollbar.setVisible(False)  # 初期は非表示

        # 従来のシンプル表示用（後方互換性）
        # ImageStripsではなく通常の画像を表示する場合に使用
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setVisible(False)  # ImageStrips使用時は非表示

        # レイアウトに追加
        # ImageStrips用（デフォルト表示）
        main_layout.addWidget(self.imagestrips_widget)
        main_layout.addWidget(self.h_scrollbar)
        # 従来の表示用（非表示）
        main_layout.addWidget(self.scroll_area)

        # ボタンを配置するレイアウト（show_buttons=Trueの場合のみ）
        if show_buttons:
            button_layout = QHBoxLayout()

            # 保存ボタン
            self.save_button = QPushButton("保存")
            self.save_button.clicked.connect(self.save_image)
            button_layout.addWidget(self.save_button)

            # 高精細保存ボタン
            self.save_hires_button = QPushButton("高精細保存")
            self.save_hires_button.clicked.connect(self.save_hires_image)
            button_layout.addWidget(self.save_hires_button)

            # 閉じるボタンは廃止されました

            # ボタンをレイアウトに追加
            main_layout.addLayout(button_layout)

        self.setCentralWidget(main_widget)

        # ウィンドウを閉じるショートカットも廃止されました

    def _on_scroll(self, value):
        """スクロールバーの値が変更されたときに呼ばれる"""
        self.logger.debug(f"Scroll position changed to: {value}")
        self.imagestrips_widget.set_scroll_position(value)

    def update_image(self, cv_img, force=False):
        """
        画像を更新する（ImageStrips対応版）

        【動作】
        - cv_imgではなく、render_one.canvasのImageStripsを直接使う
        - 従来のcv_img表示モードも維持（後方互換性）
        """
        # タイトルに現在のスコアを表示
        if self.render_one:
            score = self.render_one.score
            if self.video_base and self.video_base != "train_scan":
                video_basename = os.path.basename(self.video_base)
                self.setWindowTitle(
                    f"TrainScanner - {video_basename} (ID: {self.window_id}) [Score: {score:.3f}]"
                )
            else:
                self.setWindowTitle(
                    f"TrainScanner - ID: {self.window_id} [Score: {score:.3f}]"
                )

        # ImageStripsモードの場合
        if (
            self.render_one
            and hasattr(self.render_one, "canvas")
            and isinstance(self.render_one.canvas, ImageStrips)
        ):
            return self.update_imagestrips(force=force)

        # 従来モード（cv_imgを表示）
        return self._update_image_legacy(cv_img, force=force)

    def update_imagestrips(self, force=False):
        """
        ImageStripsモードで画像を更新（更新頻度制限あり）

        【動作】
        - render_one.canvasのImageStripsを使って表示
        - 可視範囲だけをレンダリング（効率的）
        - 大きな画像全体を結合する必要がない
        """
        # 初回は必ず表示
        is_first_update = self.last_update_time == 0

        current_time = time.time()
        # 1秒に1回の更新に制限（force=True または初回は例外）
        if (
            not force
            and not is_first_update
            and (current_time - self.last_update_time) < 1.0
        ):
            return  # ImageStripsモードではpending不要（canvasに既に追加されている）

        self.last_update_time = current_time

        # ImageStripsウィジェットを表示モードに
        self.imagestrips_widget.setVisible(True)
        self.h_scrollbar.setVisible(True)
        self.scroll_area.setVisible(False)

        # ImageStripsを設定して表示
        self.imagestrips_widget.set_imagestrips(self.render_one.canvas)

        # スクロールバーの範囲を設定
        total_width = self.imagestrips_widget.total_width
        visible_width = self.imagestrips_widget.width()
        if total_width > visible_width:
            self.h_scrollbar.setMaximum(total_width - visible_width)
            self.h_scrollbar.setPageStep(visible_width)
        else:
            self.h_scrollbar.setMaximum(0)

        # 強制的に再描画
        self.imagestrips_widget.repaint()
        QApplication.processEvents()

    def _update_image_legacy(self, cv_img, force=False):
        """
        従来モードで画像を更新（後方互換性）

        【動作】
        - cv_imgを直接表示（ImageStripsを使わない場合）
        - 従来のQLabelとQScrollAreaを使用
        """
        if cv_img is None:
            return

        # 初回は必ず表示
        is_first_update = self.last_update_time == 0

        current_time = time.time()
        if (
            not force
            and not is_first_update
            and (current_time - self.last_update_time) < 1.0
        ):
            self.pending_image = cv_img.copy()
            return

        self.current_image = cv_img.copy()
        self.pending_image = None
        self.last_update_time = current_time

        # 従来表示モードに
        self.imagestrips_widget.setVisible(False)
        self.h_scrollbar.setVisible(False)
        self.scroll_area.setVisible(True)

        pixmap = cv2_to_qpixmap(cv_img)
        if pixmap:
            self.image_label.setPixmap(pixmap)
            self.image_label.adjustSize()
            self.image_label.repaint()

            # プレビューウィンドウの場合、画像サイズに合わせてウィンドウサイズを設定
            if self.is_preview and not self.initial_size_set:
                img_height, img_width = cv_img.shape[:2]
                # 最大サイズを超えないように調整
                max_width, max_height = 1920, 1080
                if img_width > max_width or img_height > max_height:
                    scale = min(max_width / img_width, max_height / img_height)
                    window_width = int(img_width * scale)
                    window_height = int(img_height * scale)
                else:
                    window_width = img_width
                    window_height = img_height
                self.resize(window_width, window_height)
                self.initial_size_set = True

            QApplication.processEvents()
            QTimer.singleShot(10, self._scroll_to_left)

    def _scroll_to_left(self):
        """スクロールバーを左端に移動（画像の最新部分を表示）"""
        h_scrollbar = self.scroll_area.horizontalScrollBar()
        h_scrollbar.setValue(h_scrollbar.minimum())

    def flush_pending(self):
        """
        保留中の画像を強制的に更新

        【目的】1秒ごとのタイマーから呼ばれて、保留中の画像を表示
        【効果】更新頻度を抑えつつ、最新の画像も表示できる
        """
        # ImageStripsモードでは常に更新（canvasに既にデータがある）
        if (
            self.render_one
            and hasattr(self.render_one, "canvas")
            and isinstance(self.render_one.canvas, ImageStrips)
        ):
            self.update_imagestrips(force=True)
        # 従来モード
        elif self.pending_image is not None:
            self._update_image_legacy(self.pending_image, force=True)

    def set_finished(self, score: float):
        """処理が完了したことを表示する"""
        if self.video_base and self.video_base != "train_scan":
            video_basename = os.path.basename(self.video_base)
            self.setWindowTitle(
                f"TrainScanner - {video_basename} (ID: {self.window_id}) [Done - Score: {score:.3f}]"
            )
        else:
            self.setWindowTitle(
                f"TrainScanner - ID: {self.window_id} [Done - Score: {score:.3f}]"
            )

    def save_image(self):
        """
        画像とstitching履歴を保存する（ダイアログなしで自動保存）

        【仕様】
        - 画像ファイル名: {動画名}_{ウィンドウID}.png
        - 履歴ファイル名: {動画名}_{ウィンドウID}.tspos2 (JSON形式)
        - ImageStripsの場合は全体を結合して保存
        - ダイアログは表示せず、ワンクリックで保存完了

        【.tspos2ファイル形式例】
        {
          "id": 28,
          "video_path": "/path/to/video.mp4",
          "train_position": 1234.5,
          "score": 0.856,
          "scaling_factor": 0.5,
          "history": [
            {
              "frame_index": 100,
              "match_score": 0.85,
              "delta_x": -5.2,
              "delta_y": 0.1,
              "train_position": 123.4,
              "abs_pos_x": 10.5,
              "abs_pos_y": -2.3
            },
            ...
          ]
        }

        【各フィールドの意味】
        - video_path: 元動画ファイルのパス（高解像度再スキャン時に使用）
        - scaling_factor: 低解像度でのスキャン係数（例: 0.5 = 元の50%サイズ）
          実際の変位 = delta * (1 / scaling_factor)
        - abs_pos_x, abs_pos_y: カメラの手ぶれによる背景の移動量
          高解像度再スキャン時に、この分だけフレーム全体をシフトする

        【高解像度再スキャンの手順】
        1. .tspos2ファイルを読み込む
        2. video_pathから元動画を開く
        3. historyの各フレームを高解像度で処理
        4. abs_posで手ぶれ補正、deltaで列車の動きを追跡
        5. scaling_factorで座標を変換
        """
        # 画像を取得
        image_to_save = None

        # ImageStripsモードの場合
        if (
            self.render_one
            and hasattr(self.render_one, "canvas")
            and isinstance(self.render_one.canvas, ImageStrips)
        ):
            image_to_save = self.render_one.canvas.get_image()
        # 従来モード
        else:
            image_to_save = (
                self.pending_image
                if self.pending_image is not None
                else self.current_image
            )

        if image_to_save is None:
            QMessageBox.warning(self, "警告", "保存する画像がありません")
            return

        # 動画のベース名 + ID でファイル名を作成
        base_path = f"{self.video_base}_{self.window_id}"
        image_path = f"{base_path}.png"
        history_path = f"{base_path}.tspos2"

        try:
            # 画像を保存
            cv2.imwrite(image_path, image_to_save)

            # stitching履歴を保存（Render_oneがあれば）
            if self.render_one:
                history_data = self.render_one.export_history()
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(history_data, f, indent=2, ensure_ascii=False)

            # 保存成功のメッセージは表示しない（ワンクリック操作を維持）
            # 保存が成功してもウィンドウは閉じないように変更されました
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"保存に失敗しました:\n{e}")

    def save_hires_image(self):
        """
        高精細画像を保存する（stitch関数を直接呼び出して高解像度で再処理）

        【仕様】
        1. 低解像度画像を保存（.tspos2ファイルを作成）
        2. stitch関数を直接呼び出して高解像度で再処理
        3. 高解像度画像を{動画名}_{ウィンドウID}_hires.pngとして保存
        4. GUI上にプログレスバーを表示
        5. 完了後にウィンドウを閉じる

        【処理の流れ】
        1. 低解像度画像と.tspos2ファイルを保存
        2. stitch関数を直接呼び出し
        3. プログレスバーで進捗を表示
        4. 高解像度画像の生成を待つ
        5. ウィンドウを閉じる
        """
        import os
        from pathlib import Path

        # まず低解像度画像を保存（.tspos2ファイルを作成）
        # 高精細保存の場合はウィンドウを閉じないようにフラグを設定
        self._skip_close_after_save = True
        try:
            self.save_image()
        except Exception as e:
            QMessageBox.critical(
                self, "エラー", f"低解像度画像の保存に失敗しました:\n{e}"
            )
            return
        finally:
            # フラグをクリア
            if hasattr(self, "_skip_close_after_save"):
                delattr(self, "_skip_close_after_save")

        # .tspos2ファイルのパスを取得
        base_path = f"{self.video_base}_{self.window_id}"
        tspos2_path = f"{base_path}.tspos2"

        if not os.path.exists(tspos2_path):
            QMessageBox.critical(
                self, "エラー", f".tspos2ファイルが見つかりません:\n{tspos2_path}"
            )
            return

        # プログレスバー付きのカスタムダイアログを作成
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar

        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("高精細保存中...")
        progress_dialog.setModal(True)
        progress_dialog.setFixedSize(400, 150)

        # レイアウトを作成
        layout = QVBoxLayout()

        # 説明ラベル
        label = QLabel("高解像度での再処理を実行中です...")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        # プログレスバー
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setFormat("準備中...")
        # プログレスバーのスタイルを設定
        progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #ccc;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """
        )
        layout.addWidget(progress_bar)

        # レイアウトをダイアログに設定
        progress_dialog.setLayout(layout)
        progress_dialog.show()

        try:
            # stitch関数を直接呼び出し
            from trainscanner2.stitch import stitch

            # プログレスコールバック関数
            def update_progress(current, total):
                percentage = int((current / total) * 100)
                progress_bar.setValue(percentage)
                progress_bar.setFormat(
                    f"フレーム処理中... {current}/{total} ({percentage}%)"
                )
                # ラベルのテキストも更新
                label.setText(f"フレーム {current}/{total} を処理中... ({percentage}%)")

                # GUIの更新を強制
                QApplication.processEvents()

            self.logger.debug(f"Executing high-resolution stitching: {tspos2_path}")

            # stitch関数を呼び出し（verbose=Falseでコンソール出力を抑制）
            render = stitch(
                tspos2file=tspos2_path, verbose=False, progress_callback=update_progress
            )

            # 高解像度画像を保存
            hires_base_path = f"{base_path}_hires"
            progress_bar.setValue(100)
            progress_bar.setFormat("画像を保存中...")
            label.setText("高解像度画像を保存中...")
            QApplication.processEvents()

            render.save(base_path=hires_base_path)

            progress_dialog.close()

            # 処理完了後も、ウィンドウは閉じないように変更されました

        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(
                self, "エラー", f"高精細処理の実行に失敗しました:\n{e}"
            )

    def closeEvent(self, event):
        """
        ウィンドウが閉じられるときに呼ばれる（PyQt6の標準イベント）

        【目的】ウィンドウが閉じられたことをRenderに通知し、Pathを削除
        【呼ばれるタイミング】
        - ユーザーが「閉じる」ボタンをクリック
        - Command-W（Ctrl-W）を押す
        - ウィンドウの×ボタンをクリック
        """
        if self.close_callback:
            self.close_callback(self.window_id)
        event.accept()


class WindowManager:
    """
    PyQt6のウィンドウを管理するクラス

    【目的】
    - 複数のImageWindowを一元管理
    - メインスレッドでイベントループを回す（PyQt6の要件）
    - 定期的に保留中の画像を更新

    【重要】PyQt6では、GUIの更新はメインスレッドでしか行えない
    そのため、タイマーで定期的にprocessEventsを呼ぶ必要がある
    """

    logger = getLogger(__name__)

    def __init__(
        self,
        video_base: str = None,
        renderer_callback=None,
        show_gaps=False,
        show_buttons=True,  # ウィンドウに保存・閉じるボタンを表示するか
    ):
        try:
            # QApplicationのインスタンスを取得または作成（必須）
            self.app = QApplication.instance()
            if self.app is None:
                self.app = QApplication(sys.argv)

            self.windows = {}  # window_id -> ImageWindow
            self.video_base = video_base or "train_scan"
            self.show_gaps = show_gaps  # デバッグ用: 短冊間に隙間を表示
            self.show_buttons = show_buttons  # 保存・閉じるボタンを表示するか
            self.renderer_callback = (
                renderer_callback  # ウィンドウが閉じられたときにRenderに通知
            )

            # イベントループを定期的に処理するタイマー
            # 【重要】これがないとウィンドウが応答しなくなる
            self.event_timer = QTimer()
            self.event_timer.timeout.connect(self.process_events)
            self.event_timer.start(5)  # 5msごとにイベント処理（頻繁に回す）

            # 保留中の画像を更新するタイマー
            # 【目的】1秒以内に複数回更新があった場合、最新の画像を表示
            self.flush_timer = QTimer()
            self.flush_timer.timeout.connect(self.flush_all_pending)
            self.flush_timer.start(1000)  # 1秒ごと

            self.logger.debug("WindowManager initialized with PyQt6")
        except Exception as e:
            self.logger.error(f"Failed to initialize WindowManager: {e}")
            raise

    def process_events(self):
        """
        Qtのイベントループを処理

        【重要】PyQt6のGUIを動作させるために必要
        これを呼ばないと、ウィンドウが応答しなくなる
        """
        self.app.processEvents()

    def flush_all_pending(self):
        """
        すべてのウィンドウの保留中の画像を更新

        【目的】更新頻度制限（1秒に1回）により保留された画像を表示
        【効果】毎フレーム更新しなくても、最新の画像が定期的に表示される
        """
        # 辞書のコピーを作成してからイテレート（競合状態を回避）
        windows_copy = list(self.windows.values())
        for window in windows_copy:
            window.flush_pending()

    def _on_window_closed(self, window_id: int):
        """
        ウィンドウが閉じられたときのコールバック

        【呼ばれるタイミング】ImageWindow.closeEventから呼ばれる
        【動作】
        1. windowsから削除（WindowManagerの管理から外す）
        2. Renderに通知してPathを削除（メモリ解放、以降の処理をスキップ）
        """
        if window_id in self.windows:
            self.logger.debug(f"Window {window_id} closed")
            del self.windows[window_id]
            # Renderにも通知してPathを削除
            if self.renderer_callback:
                self.renderer_callback(window_id)

    def create_window(self, window_id: int, render_one=None):
        """
        新しいウィンドウを作成

        Args:
            window_id: ウィンドウID
            render_one: Render_oneインスタンス（履歴保存用）
        """
        if window_id not in self.windows:
            window = ImageWindow(
                window_id,
                video_base=self.video_base,
                close_callback=self._on_window_closed,
                render_one=render_one,
                show_gaps=self.show_gaps,  # デバッグ用
                show_buttons=self.show_buttons,  # ボタン表示設定
            )
            window.show()
            self.windows[window_id] = window
            # 即座にprocessEventsを呼んで表示を確実にする
            self.app.processEvents()
        return self.windows[window_id]

    def update_window(self, window_id: int, cv_img):
        """ウィンドウの画像を更新"""
        if window_id in self.windows:
            self.windows[window_id].update_image(cv_img)

    def update_preview_window(self, cv_img):
        """
        プレビューウィンドウ（現在のフレーム表示用）を更新

        Args:
            cv_img: 表示するOpenCV画像
        """
        PREVIEW_WINDOW_ID = -1  # プレビューウィンドウ用の特別なID
        if PREVIEW_WINDOW_ID not in self.windows:
            # プレビューウィンドウを作成（render_oneなし、ボタンなし）
            window = ImageWindow(
                PREVIEW_WINDOW_ID,
                video_base=self.video_base,
                close_callback=None,  # プレビューウィンドウは閉じても問題ない
                render_one=None,  # render_oneは不要
                show_gaps=False,
                show_buttons=False,  # ボタンは表示しない
            )
            window.setWindowTitle("TrainScanner - プレビュー")
            window.show()
            self.windows[PREVIEW_WINDOW_ID] = window
            # 即座にprocessEventsを呼んで表示を確実にする
            self.app.processEvents()
        if PREVIEW_WINDOW_ID in self.windows:
            self.windows[PREVIEW_WINDOW_ID].update_image(cv_img)

    def set_window_finished(self, window_id: int, score: float):
        """ウィンドウに処理完了を表示"""
        if window_id in self.windows:
            # 処理完了時は保留中の画像を強制更新
            self.windows[window_id].flush_pending()
            self.windows[window_id].set_finished(score)

    def close_window(self, window_id: int):
        """ウィンドウを閉じる（プログラムから）"""
        if window_id in self.windows:
            self.windows[window_id].close()
            # closeEventで自動削除されるので、ここでは削除しない

    def close_all(self):
        """すべてのウィンドウを閉じる"""
        for window in list(self.windows.values()):
            window.close()
        # closeEventで自動削除されるので、clearは不要

    def has_windows(self):
        """ウィンドウが1つでも開いているかチェック"""
        return len(self.windows) > 0

    def wait_for_close(self):
        """すべてのウィンドウが閉じられるまで待機"""
        self.logger.debug("Waiting for all windows to be closed...")
        while self.has_windows():
            self.app.processEvents()
            time.sleep(0.1)
        self.logger.debug("All windows closed.")
