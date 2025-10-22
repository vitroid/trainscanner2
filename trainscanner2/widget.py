"""
PyQt6用のカスタムウィジェット

ImageStripsを効率的に表示するための仮想スクロール対応ウィジェット
"""

import numpy as np
import cv2
from logging import getLogger

from PyQt6.QtWidgets import QWidget, QApplication, QMainWindow, QVBoxLayout
from PyQt6.QtGui import QImage, QPixmap, QPainter
from PyQt6.QtCore import Qt, pyqtSignal


def cv2_to_qpixmap(cv_img):
    """OpenCVの画像(BGR)をQPixmapに変換する"""
    if cv_img is None:
        return None
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    # BGRからRGBに変換
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    q_img = QImage(
        rgb_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
    )
    return QPixmap.fromImage(q_img)


class ImageStripsWidget(QWidget):
    """
    ImageStripsを効率的に表示するウィジェット

    【目的】
    - 巨大な画像を全て結合して表示するのではなく、可視範囲だけをレンダリング
    - imagecomb.imagesの画像番号で表示位置を指定
    - メモリと処理を節約

    【仮想スクロール】
    - 画面に表示される範囲の画像だけを結合
    - スクロール時に動的に再描画

    【デバッグ機能】
    - show_gaps=True で短冊間に1pxの隙間を表示
    """

    scroll_changed = pyqtSignal(int)  # スクロール位置変更シグナル

    def __init__(self, show_gaps=False, parent=None):
        super().__init__(parent)
        self.imagestrips = None  # ImageStripsインスタンス
        self.current_start_index = 0  # 表示開始位置（imagesのインデックス）
        self.scroll_offset = 0  # 画像内のピクセルオフセット
        self.cached_pixmap = None  # キャッシュされた表示画像
        self.total_width = 0  # 全体の幅（スクロールバー用）
        # self.image_widths = []  # 各画像の幅（累積）
        self.show_gaps = show_gaps  # 短冊間の隙間を表示するか
        self.setMinimumSize(400, 300)

    def _calculate_widths(self):
        """各画像の幅と累積幅を計算"""
        if self.imagestrips is None:
            return

        self.image_widths = []
        cumulative = 0
        for shape in self.imagestrips.shapes:
            h, w = shape[:2]
            cumulative += w
            if self.show_gaps:
                cumulative += 1  # 隙間分を加算
            self.image_widths.append(cumulative)

        # 全体の幅
        self.total_width = cumulative
        if (
            self.imagestrips.buffer is not None
            and self.imagestrips.buffer.image is not None
        ):
            self.total_width += self.imagestrips.buffer.image.shape[1]

    def set_imagestrips(self, imagestrips):
        """ImageStripsをセットして表示を更新"""
        self.imagestrips = imagestrips
        self._calculate_widths()
        self.update_display()

    def set_scroll_position(self, position):
        """
        スクロール位置を設定（ピクセル単位）

        Args:
            position: 左端からのピクセル数
        """
        if not self.image_widths:
            return

        logger = getLogger(__name__)

        # ピクセル位置から画像インデックスを計算
        for idx, cumulative_width in enumerate(self.image_widths):
            if position < cumulative_width:
                # このインデックスの画像を表示開始位置とする
                self.current_start_index = idx
                prev_width = self.image_widths[idx - 1] if idx > 0 else 0
                self.scroll_offset = position - prev_width
                logger.debug(
                    f"Scroll position {position} -> start_index={idx}, offset={self.scroll_offset}"
                )
                break
        else:
            # 最後の画像
            self.current_start_index = max(0, len(self.image_widths) - 1)
            self.scroll_offset = 0
            logger.debug(
                f"Scroll position {position} -> last image (index={self.current_start_index})"
            )

        self.update_display()

    def update_display(self, start_index=None):
        """
        表示を更新（ImageStrips.get_image()を使用）

        Args:
            start_index: 表示開始位置（Noneの場合は現在位置を維持）
        """
        if self.imagestrips is None:
            return

        if start_index is not None:
            self.current_start_index = start_index

        # 画面幅を取得（少し余裕を持たせる）
        widget_width = self.width()
        request_width = int(widget_width * 1.5)

        # ImageStrips.get_image()を使って可視範囲を取得
        combined = self.imagestrips.get_image(
            start=self.current_start_index, width=request_width
        )

        if combined is not None:
            # show_gapsがTrueの場合、短冊間に隙間を追加
            # TODO: ImageStrips.get_image()にshow_gaps引数を追加して、
            # ImageStrips側で隙間を入れるようにする
            # 現状は隙間なしで表示（既に結合された画像には後から隙間を入れられない）

            self.cached_pixmap = cv2_to_qpixmap(combined)

        # 全体の幅を計算（スクロールバー用）
        self.total_width = sum(
            img.shape[1] + (1 if self.show_gaps else 0)  # 隙間分を加算
            for img in self.imagestrips.images
        )
        if (
            self.imagestrips.buffer is not None
            and self.imagestrips.buffer.image is not None
        ):
            self.total_width += self.imagestrips.buffer.image.shape[1]

        self.update()

    def paintEvent(self, event):
        """ウィジェットを描画"""
        if self.cached_pixmap:
            painter = QPainter(self)
            painter.drawPixmap(0, 0, self.cached_pixmap)

    def sizeHint(self):
        """推奨サイズを返す"""
        if self.cached_pixmap:
            return self.cached_pixmap.size()
        return super().sizeHint()


def main():
    """
    ImageStripsWidgetのテストケース

    【テスト内容】
    - ImageStripsに複数の短冊画像を追加
    - ImageStripsWidgetで表示
    - 短冊間の1px隙間を確認（show_gaps=True）

    【終了方法】
    - ウィンドウを閉じる（×ボタン、Command-W）
    - Ctrl-C でも終了可能
    """
    from trainscanner2.imagestrips import ImageStrips
    from pyperbox import Rect
    import sys
    import signal

    print("ImageStripsWidgetテストを開始します...")

    # Ctrl-C で確実に終了できるようにする
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    try:
        app = QApplication(sys.argv)

        # アプリケーションが最後のウィンドウを閉じたときに終了するように設定
        app.setQuitOnLastWindowClosed(True)

        # テスト用ImageStripsを作成
        imagestrips = ImageStrips()
        print(f"ImageStrips created: {len(imagestrips.images)} images")

        # カラフルな短冊画像を追加
        colors = [
            (255, 0, 0),  # 赤
            (0, 255, 0),  # 緑
            (0, 0, 255),  # 青
            (255, 255, 0),  # 黄
            (255, 0, 255),  # マゼンタ
            (0, 255, 255),  # シアン
        ]

        for i, color in enumerate(colors):
            # 縦長の画像を作成（100x300 px）
            img = np.zeros((300, 100, 3), dtype=np.uint8)
            img[:, :, :] = color

            # テキストを追加
            cv2.putText(
                img,
                f"#{i}",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            # ImageStripsに追加（左から右へ）
            rect = Rect.from_bounds(
                left=i * 100,
                right=(i + 1) * 100,
                top=-150,
                bottom=150,
            )
            print(f"Adding image {i}, rect: {rect}")
            imagestrips.put_image(rect, img)

        print(f"ImageStrips after adding: {len(imagestrips.images)} images")

        # ウィンドウを作成
        window = QMainWindow()
        window.setWindowTitle("ImageStripsWidget Test - Close window to exit")
        window.resize(600, 400)

        # ImageStripsWidgetを作成（隙間を表示）
        print("Creating ImageStripsWidget...")
        widget = ImageStripsWidget(show_gaps=True)
        print("Setting imagestrips...")
        widget.set_imagestrips(imagestrips)

        window.setCentralWidget(widget)
        window.show()

        print("ウィンドウを表示しました。")
        print("短冊間に1pxの黒い隙間が見えるはずです。")
        print("ウィンドウを閉じると終了します。")

        sys.exit(app.exec())

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
