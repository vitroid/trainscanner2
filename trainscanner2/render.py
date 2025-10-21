import numpy as np
from tiledimage.simpleimage import SimpleImage
from trainscanner.image import linear_alpha
from trainscanner2.imagestrips import ImageStrips
import cv2
from logging import getLogger
from trainscanner2 import FIFO, PathItem
import sys
import time
import os
import json
from pyperbox import Rect

# PyQt6のインポートを試みる（インストールされていない場合はNoneに）
try:
    from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QScrollArea,
        QLabel,
        QWidget,
        QVBoxLayout,
    )
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtCore import Qt, QTimer

    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    QApplication = QMainWindow = QScrollArea = QLabel = None
    QWidget = QVBoxLayout = QImage = QPixmap = Qt = QTimer = None


def cv2_to_qpixmap(cv_img):
    """OpenCVの画像(BGR)をQPixmapに変換する"""
    if not PYQT6_AVAILABLE or cv_img is None:
        return None
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    # BGRからRGBに変換
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    q_img = QImage(
        rgb_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
    )
    return QPixmap.fromImage(q_img)


if PYQT6_AVAILABLE:
    from PyQt6.QtWidgets import (
        QPushButton,
        QHBoxLayout,
        QFileDialog,
        QMessageBox,
        QScrollBar,
    )
    from PyQt6.QtGui import QShortcut, QKeySequence
    from trainscanner2.widget import ImageStripsWidget

    # ImageStripsWidgetはwidget.pyに移動しました

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
            self.setWindowTitle(f"Train Scanner - ID: {window_id}")

            # 画像管理
            self.current_image = None  # 現在表示中の画像
            self.last_update_time = 0  # 最後の更新時刻（更新頻度制限用）
            self.pending_image = None  # 保留中の画像（1秒以内の更新はここに保存）

            # 最大ウィンドウサイズを設定（画面サイズの80%程度）
            self.setMaximumSize(1920, 1080)
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

                # 閉じるボタン
                self.close_button = QPushButton("閉じる")
                self.close_button.clicked.connect(self.close)
                button_layout.addWidget(self.close_button)

                # ボタンをレイアウトに追加
                main_layout.addLayout(button_layout)

            self.setCentralWidget(main_widget)

            # macOS標準のCommand-W（Windows/LinuxではCtrl-W）でウィンドウを閉じる
            close_shortcut = QShortcut(QKeySequence.StandardKey.Close, self)
            close_shortcut.activated.connect(self.close)

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

        def set_finished(self, quality: float):
            """処理が完了したことを表示する"""
            self.setWindowTitle(
                f"Train Scanner - ID: {self.window_id} [処理完了 - Quality: {quality:.3f}]"
            )

        def save_image(self):
            """
            画像とstitching履歴を保存する（ダイアログなしで自動保存）

            【仕様】
            - 画像ファイル名: {動画名}_{ウィンドウID}.jpg
            - 履歴ファイル名: {動画名}_{ウィンドウID}.tspos2 (JSON形式)
            - ImageStripsの場合は全体を結合して保存
            - ダイアログは表示せず、ワンクリックで保存完了

            【.tspos2ファイル形式例】
            {
              "id": 28,
              "video_path": "/path/to/video.mp4",
              "train_position": 1234.5,
              "quality": 0.856,
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
            image_path = f"{base_path}.jpg"
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
                # 保存が成功したらウィンドウを閉じる
                self.close()
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"保存に失敗しました:\n{e}")

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

else:
    ImageWindow = None


def rotated_placement(canvas, frame, sine, cosine, train_position, first=False):
    h, w = frame.shape[:2]
    rh = int(abs(h * cosine) + abs(w * sine))
    rw = int(abs(h * sine) + abs(w * cosine))
    halfw, halfh = w / 2, h / 2
    R = np.matrix(
        (
            (cosine, sine, -cosine * halfw - sine * halfh + rw / 2),
            (-sine, cosine, sine * halfw - cosine * halfh + rh / 2),
        )
    )
    alpha = linear_alpha(img_width=rw, mixing_width=20, slit_pos=0, head_right=False)
    rotated = cv2.warpAffine(frame, R, (rw, rh))
    # cv2.imshow("rotated", rotated)
    # cv2.waitKey(0)
    # 画像中心をそろえる
    # if first:
    canvas.put_image(
        lefttop=(int(train_position) - rw // 2, -rh // 2),
        image=rotated,
    )


class Render_one:
    """
    1つのPathの描画を担当する。
    """

    logger = getLogger(__name__)

    def __init__(
        self,
        id: int,
        num_leading_frames: int,
        window_manager=None,
        scaling_factor: float = 1.0,
        video_path: str = None,
        cache: bool = False,
    ):
        self.num_leading_frames = num_leading_frames
        self.leading_frames = FIFO(num_leading_frames)
        self.history = []  # PathItemのリスト
        self.abs_positions = []  # 各フレームのabsolute_position（手ぶれ補正）
        self.train_positions = []  # 各フレームでのtrain_position（再計算不要に）
        self.id = id
        self.canvas = ImageStrips(cache=cache)
        self.first = False
        self.train_position = 0
        self.alive = True
        self.window_manager = window_manager
        self.window = None
        self.scaling_factor = scaling_factor  # 低解像度→高解像度への変換係数
        self.video_path = video_path  # 動画ファイルパス（高解像度再スキャン用）

    def done(self):
        self.alive = False
        # PyQt6ウィンドウは閉じない（ユーザーが手動で閉じるボタンを押すまで残す）
        if self.window_manager and self.window:
            # ウィンドウのタイトルを「処理完了」に更新
            self.window_manager.set_window_finished(self.id, self.quality)
        else:
            # OpenCVウィンドウのみ自動で閉じる
            cv2.destroyWindow(f"{self.id}")
            cv2.waitKey(1)

    def _render_one(
        self,
        frame: np.ndarray,
        h: PathItem,
    ):
        delta = h.xy
        frame_index, value = h.value
        self.logger.debug(f"{id=} {frame_index=} {delta=} ")
        dx, dy = delta
        dd = (dx**2 + dy**2) ** 0.5
        if dd != 0:
            self.train_position += dd
            # 各フレームでの位置を記録（export_history()で使用）
            self.train_positions.append(self.train_position)
            cosine = dx / dd
            sine = dy / dd
            rotated_placement(
                self.canvas, frame, sine, cosine, self.train_position, self.first
            )
            self.first = False
        else:
            # dd=0の場合も位置を記録（前のフレームと同じ位置）
            self.train_positions.append(self.train_position)

    def put(
        self,
        frame: np.ndarray,
        pathitem: PathItem,
        quality_threshold=0.0,
        absolute_position=None,
    ):
        if not self.alive:
            return
        self.history.append(pathitem)
        self.abs_positions.append(
            absolute_position if absolute_position is not None else (0, 0)
        )
        self.leading_frames.append(frame)
        if len(self.history) > self.num_leading_frames:
            if 0 < self.quality < quality_threshold:  # or abs(self.train_position) < 3:
                # close the window
                self.logger.info(
                    f"Close {self.id=} {self.quality=} {quality_threshold=} {self.train_position=}"
                )
                # self.logger.info(f"{0 < self.quality < quality_threshold}")
                # self.logger.info(f"{self.train_position}")
                # self.logger.info(self.history)
                self.done()
                return

            self._render_one(frame, pathitem)

            # ImageStripsモードの場合、canvas.get_image()は重い（全体結合）
            # PyQt6では、update_windowにNoneを渡してcanvasを直接参照させる
            use_imagestrips = isinstance(self.canvas, ImageStrips)
            if use_imagestrips:
                # ImageStripsモード: ウィンドウに通知だけ（画像は渡さない）
                if self.window_manager:
                    try:
                        if self.window is None:
                            self.window = self.window_manager.create_window(
                                self.id, render_one=self
                            )
                        self.window_manager.update_window(self.id, None)  # Noneを渡す
                    except Exception as e:
                        self.logger.error(f"PyQt6 window error: {e}")
                # OpenCVは非対応（ImageStripsの全体画像が必要）
            else:
                # 従来モード: 画像を取得して渡す
                img = self.canvas.get_image()
                if img is not None:
                    cv2.putText(
                        img,
                        f"Quality: {self.quality:.3f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        img,
                        f"ID: {self.id}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    if self.window_manager:
                        try:
                            if self.window is None:
                                self.window = self.window_manager.create_window(
                                    self.id, render_one=self
                                )
                            self.window_manager.update_window(self.id, img)
                        except Exception as e:
                            self.logger.error(
                                f"PyQt6 window error: {e}, falling back to OpenCV"
                            )
                            cv2.imshow(f"{self.id}", img)
                            cv2.waitKey(1)
                    else:
                        # PyQt6が使用されていない場合はOpenCVを使用
                        cv2.imshow(f"{self.id}", img)
                        cv2.waitKey(1)
        elif len(self.history) == 20:
            for f, pi in zip(self.leading_frames.queue, self.history):
                self._render_one(f, pi)

    @property
    def quality(self):
        # 最初の20フレームで判別する。
        if len(self.history) > 20:
            return np.mean([h.value[1] for h in self.history])
        return 0.0

    def export_history(self):
        """
        stitching履歴をエクスポート（.tspos2ファイル保存用）

        【目的】
        - レンダリングとデータ保存の責務を分離
        - 画像とは別にstitching情報を保存できるようにする
        - 高解像度での再レンダリングに必要な情報を提供

        【.tspos2ファイルの用途】
        - 低解像度でスキャンした後、高解像度で再スキャンする際に使用
        - 既知のPath情報を使って、高精度なstitchingを実行

        Returns:
            dict: stitching履歴データ（JSON形式）
                - id: PathのID
                - video_path: 元動画ファイルのパス（高解像度再スキャン用）
                - train_position: 最終的なキャンバス上の位置
                - quality: 平均品質スコア
                - scaling_factor: 低解像度→高解像度への変換係数
                  （実際の変位 = delta * (1/scaling_factor)）
                - history: フレームごとの詳細情報のリスト
                    * frame_index: ビデオのフレーム番号
                    * match_score: マッチングスコア
                    * delta_x, delta_y: 直前のコマからの変位ベクトル（スケール済み）
                    * train_position: そのフレームでのキャンバス位置
                    * abs_pos_x, abs_pos_y: 手ぶれによる背景移動量
        """
        # 【重要】train_positionは再計算せず、_render_one()で記録した値を使う
        # これにより、コードの重複を避け、計算ミスを防ぐ
        history_data = []

        for idx, h in enumerate(self.history):
            frame_index, match_score = h.value
            delta_x, delta_y = h.xy

            # absolute_positionを取得（保存されていれば）
            abs_pos = (
                self.abs_positions[idx] if idx < len(self.abs_positions) else (0, 0)
            )

            # train_positionを取得（_render_one()で記録済み）
            train_pos = (
                self.train_positions[idx] if idx < len(self.train_positions) else 0.0
            )

            history_data.append(
                {
                    "frame_index": int(frame_index),
                    "match_score": float(match_score),
                    "delta_x": float(delta_x),
                    "delta_y": float(delta_y),
                    "train_position": float(train_pos),
                    "abs_pos_x": float(abs_pos[0]),
                    "abs_pos_y": float(abs_pos[1]),
                }
            )

        return {
            "id": self.id,
            "video_path": self.video_path,
            "train_position": float(self.train_position),
            "quality": float(self.quality),
            "scaling_factor": float(self.scaling_factor),
            "history": history_data,
        }

    def save(self, base_path=None):
        """
        画像とstitching履歴を保存する（メモリ効率的）

        【目的】
        - stitch.pyなど、WindowManagerを使わない環境でも保存できるようにする
        - ImageStrips.save_to_file()を使ってメモリ効率的に保存

        【保存されるファイル】
        - {base_path}.jpg: stitchされた画像
        - {base_path}.tspos2: stitching履歴（JSON形式）

        Args:
            base_path: ファイルのベースパス（拡張子なし）
                      Noneの場合、video_path + "_" + id を使用
        """
        # ベースパスを決定
        if base_path is None:
            if self.video_path:
                video_base = os.path.splitext(self.video_path)[0]
                base_path = f"{video_base}_{self.id}"
            else:
                base_path = f"train_scan_{self.id}"

        image_path = f"{base_path}.jpg"
        history_path = f"{base_path}.tspos2"

        # 画像を保存（メモリ効率的）
        if isinstance(self.canvas, ImageStrips):
            self.logger.info(f"Saving image to {image_path} (memory-efficient mode)...")
            self.canvas.save_to_file(image_path)
        else:
            # 従来のSimpleImage形式
            img = self.canvas.get_image()
            if img is not None:
                cv2.imwrite(image_path, img)
                self.logger.info(f"Saved image to {image_path}")

        # stitching履歴を保存
        history_data = self.export_history()
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved history to {history_path}")

        return image_path, history_path


if PYQT6_AVAILABLE:

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

                self.logger.info("WindowManager initialized with PyQt6")
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
            for window in self.windows.values():
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
                self.logger.info(f"Window {window_id} closed")
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

        def set_window_finished(self, window_id: int, quality: float):
            """ウィンドウに処理完了を表示"""
            if window_id in self.windows:
                # 処理完了時は保留中の画像を強制更新
                self.windows[window_id].flush_pending()
                self.windows[window_id].set_finished(quality)

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
            self.logger.info("Waiting for all windows to be closed...")
            while self.has_windows():
                self.app.processEvents()
                time.sleep(0.1)
            self.logger.info("All windows closed.")

else:
    WindowManager = None


class Render:
    """
    極値とフレームをうけとり、個別のレンダラーに差配する。
    生きているrendererの最高品質を調査し、低品質rendererの打ち切り指示をする

    【Path管理の仕組み】
    - self.renderers = {id: Render_one}  # 全てのPathを管理
    - Render_one.alive フラグで状態を管理:
      * alive=True:  処理中（新しいフレームデータを受け入れる）
      * alive=False: 処理完了（ウィンドウは残るが、データは受け入れない）

    【Pathの削除タイミング】
    1. 品質が閾値以下になったとき (done())
    2. ユーザーがウィンドウを閉じたとき (remove_renderer())
    3. 処理完了後に低品質Pathを一括削除 (close_low_quality_windows())
    """

    logger = getLogger(__name__)

    def __init__(
        self,
        use_pyqt=True,
        video_path: str = None,
        scaling_factor: float = 1.0,
        show_gaps=False,
        show_buttons=True,  # ウィンドウに保存・閉じるボタンを表示するか
    ):
        self.renderers = {}  # {id: Render_one} 全Path（active/finished両方）
        self.max_quality = 0.0
        self.window_manager = None
        self.scaling_factor = scaling_factor  # 低解像度→高解像度への変換係数
        self.video_path = video_path  # 動画ファイルパス

        if use_pyqt:
            # 動画ファイルパスからベース名を取得
            video_base = None
            if video_path:
                # 拡張子を除いたベース名を取得
                video_base = os.path.splitext(video_path)[0]

            if not PYQT6_AVAILABLE:
                self.logger.warning(
                    "PyQt6 is not installed, using OpenCV windows instead"
                )
            elif WindowManager is None:
                self.logger.warning(
                    "WindowManager is not available, using OpenCV windows"
                )
            else:
                try:
                    # ウィンドウが閉じられたときにPathも削除するコールバックを渡す
                    self.window_manager = WindowManager(
                        video_base=video_base,
                        renderer_callback=self.remove_renderer,
                        show_gaps=show_gaps,  # デバッグ用
                        show_buttons=show_buttons,  # ボタン表示設定
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to initialize PyQt6, using OpenCV windows: {e}"
                    )
                    self.window_manager = None

    def put(
        self,
        id: int,
        frame: np.ndarray,
        historyitem: PathItem,
        absolute_position=None,
    ):
        # 既に削除されたPathはスキップ（効率化）
        if id in self.renderers:
            r = self.renderers[id]
        else:
            # 新しいPathを作成
            r = Render_one(
                id,
                num_leading_frames=20,
                window_manager=self.window_manager,
                scaling_factor=self.scaling_factor,
                video_path=self.video_path,
            )
            self.renderers[id] = r

        r.put(
            frame,
            historyitem,
            quality_threshold=self.max_quality * 0.75,
            absolute_position=absolute_position,
        )
        q = r.quality
        # 最高品質が更新されたら、低品質ウィンドウをチェックして閉じる
        if self.max_quality < q:
            self.max_quality = q
            # 閾値が上がったので、低品質ウィンドウを閉じる
            self._check_and_close_low_quality_windows()

    def done(self, id):
        """
        Pathの処理を終了（品質が閾値以下の場合）

        【呼ばれるタイミング】Render_one.put()で品質が閾値以下になったとき
        【動作】
        1. ウィンドウを閉じる（または閉じない）
        2. Pathを削除（メモリ解放、以降の処理をスキップ）
        """
        if id in self.renderers:
            r = self.renderers[id]
            r.done()
            # Pathを削除（メモリ解放、以降の処理をスキップ）
            del self.renderers[id]
            self.logger.info(f"Removed renderer {id}")

    def remove_renderer(self, id):
        """
        レンダラーを削除（ユーザーが手動でウィンドウを閉じた場合など）

        【呼ばれるタイミング】
        - WindowManager._on_window_closedから呼ばれる
        - ユーザーがウィンドウを閉じたとき

        【目的】ウィンドウが閉じられたPathは以降処理不要なので削除
        """
        if id in self.renderers:
            del self.renderers[id]
            self.logger.info(f"Removed renderer {id}")

    def get_active_paths(self):
        """
        処理中のPath（alive=True）のみを返す

        【用途】
        - まだフレームデータを受け入れているPathを確認
        - 統計情報の取得

        Returns:
            dict: {id: Render_one} 処理中のPathの辞書
        """
        return {id: r for id, r in self.renderers.items() if r.alive}

    def get_finished_paths(self):
        """
        処理が完了したPath（alive=False）のみを返す

        【用途】
        - ウィンドウは残っているが処理は終了したPathを確認
        - PyQt6ウィンドウで保存待ちのPathを取得

        Returns:
            dict: {id: Render_one} 処理完了したPathの辞書
        """
        return {id: r for id, r in self.renderers.items() if not r.alive}

    def get_path_stats(self):
        """
        Path管理の統計情報を返す

        Returns:
            dict: 統計情報
                - total: 総Path数
                - active: 処理中のPath数
                - finished: 処理完了Path数
                - max_quality: 最高品質
        """
        active = self.get_active_paths()
        finished = self.get_finished_paths()
        return {
            "total": len(self.renderers),
            "active": len(active),
            "finished": len(finished),
            "max_quality": self.max_quality,
        }

    def _check_and_close_low_quality_windows(self, quality_ratio: float = 0.5):
        """
        低品質ウィンドウをチェックして閉じる（処理中に随時実行）

        【目的】
        - 最高品質が更新されるたびに、閾値以下のウィンドウを閉じる
        - ウィンドウが無限に増え続けないようにする

        【呼ばれるタイミング】
        - Render.put()でmax_qualityが更新されたとき

        Args:
            quality_ratio: 最高品質に対する比率（デフォルト: 0.5 = 50%）
        """
        if not self.renderers or self.max_quality == 0:
            return

        threshold = self.max_quality * quality_ratio
        to_close = []

        # 閉じるウィンドウをリストアップ
        for id, renderer in self.renderers.items():
            # 品質が計算されていて（quality > 0）、かつ閾値以下
            if renderer.quality > 0 and renderer.quality < threshold:
                to_close.append(id)

        # ウィンドウを閉じてPathを削除
        for id in to_close:
            # closeEventで既に削除されている可能性があるのでチェック
            if id not in self.renderers:
                continue

            renderer = self.renderers[id]
            self.logger.info(
                f"Auto-closing window {id}: quality={renderer.quality:.3f} < threshold={threshold:.3f}"
            )
            if self.window_manager:
                # PyQt6の場合: close()を呼ぶとcloseEventが発火して自動的に削除される
                self.window_manager.close_window(id)
                # closeEventでremove_renderer()が呼ばれるので、ここでは削除しない
            else:
                # OpenCVの場合: 手動でウィンドウを閉じて削除
                cv2.destroyWindow(f"{id}")
                cv2.waitKey(1)
                # Pathを削除
                if id in self.renderers:
                    del self.renderers[id]

        if to_close:
            self.logger.info(
                f"Closed {len(to_close)} low-quality windows during processing"
            )

    def close_low_quality_windows(self, quality_ratio: float = 0.5):
        """
        処理完了後に品質が閾値以下のウィンドウを自動で閉じる（互換性のため）

        【注意】
        - このメソッドは処理完了後に明示的に呼ぶ用
        - 実際には、処理中も随時_check_and_close_low_quality_windows()が呼ばれている
        - このメソッドは最終確認として呼ばれる

        Args:
            quality_ratio: 最高品質に対する比率（デフォルト: 0.5 = 50%）
        """
        self._check_and_close_low_quality_windows(quality_ratio)

    def close_all(self):
        """すべてのウィンドウを閉じる"""
        if self.window_manager:
            self.window_manager.close_all()

    def wait_for_windows_close(self):
        """PyQt6ウィンドウが全て閉じられるまで待機（PyQt6使用時のみ）"""
        if self.window_manager:
            self.window_manager.wait_for_close()
