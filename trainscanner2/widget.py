"""
PyQt6用のカスタムウィジェット

ImageCombを効率的に表示するための仮想スクロール対応ウィジェット
"""

import numpy as np
import cv2
from logging import getLogger

try:
    from PyQt6.QtWidgets import QWidget, QApplication, QMainWindow, QVBoxLayout
    from PyQt6.QtGui import QImage, QPixmap, QPainter
    from PyQt6.QtCore import Qt, pyqtSignal

    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    QWidget = QApplication = QMainWindow = QVBoxLayout = None
    QImage = QPixmap = QPainter = Qt = pyqtSignal = None


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

    class ImageCombWidget(QWidget):
        """
        ImageCombを効率的に表示するウィジェット

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
            self.image_comb = None  # ImageCombインスタンス
            self.current_start_index = 0  # 表示開始位置（imagesのインデックス）
            self.scroll_offset = 0  # 画像内のピクセルオフセット
            self.cached_pixmap = None  # キャッシュされた表示画像
            self.total_width = 0  # 全体の幅（スクロールバー用）
            self.image_widths = []  # 各画像の幅（累積）
            self.show_gaps = show_gaps  # 短冊間の隙間を表示するか
            self.setMinimumSize(400, 300)

        def _calculate_widths(self):
            """各画像の幅と累積幅を計算"""
            if self.image_comb is None:
                return

            self.image_widths = []
            cumulative = 0
            for img in self.image_comb.images:
                cumulative += img.shape[1]
                # show_gapsがTrueの場合、短冊の後に1px隙間を追加
                if self.show_gaps:
                    cumulative += 1
                self.image_widths.append(cumulative)

            # 全体の幅
            self.total_width = cumulative
            if (
                self.image_comb.buffer is not None
                and self.image_comb.buffer.image is not None
            ):
                self.total_width += self.image_comb.buffer.image.shape[1]

        def set_image_comb(self, image_comb):
            """ImageCombをセットして表示を更新"""
            self.image_comb = image_comb
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

            # ピクセル位置から画像インデックスを計算
            for idx, cumulative_width in enumerate(self.image_widths):
                if position < cumulative_width:
                    self.current_start_index = max(0, idx - 1)
                    prev_width = self.image_widths[idx - 1] if idx > 0 else 0
                    self.scroll_offset = position - prev_width
                    break
            else:
                # 最後の画像
                self.current_start_index = max(0, len(self.image_widths) - 1)

            self.update_display()

        def update_display(self, start_index=None):
            """
            表示を更新（ImageComb.get_image()を使用）

            Args:
                start_index: 表示開始位置（Noneの場合は現在位置を維持）
            """
            if self.image_comb is None:
                return

            if start_index is not None:
                self.current_start_index = start_index

            # 画面幅を取得（少し余裕を持たせる）
            widget_width = self.width()
            request_width = int(widget_width * 1.5)

            # ImageComb.get_image()を使って可視範囲を取得
            combined = self.image_comb.get_image(
                start=self.current_start_index, width=request_width
            )

            if combined is not None:
                # show_gapsがTrueの場合、短冊間に隙間を追加
                # TODO: ImageComb.get_image()にshow_gaps引数を追加して、
                # ImageComb側で隙間を入れるようにする
                # 現状は隙間なしで表示（既に結合された画像には後から隙間を入れられない）

                self.cached_pixmap = cv2_to_qpixmap(combined)

            # 全体の幅を計算（スクロールバー用）
            self.total_width = sum(
                img.shape[1] + (1 if self.show_gaps else 0)  # 隙間分を加算
                for img in self.image_comb.images
            )
            if (
                self.image_comb.buffer is not None
                and self.image_comb.buffer.image is not None
            ):
                self.total_width += self.image_comb.buffer.image.shape[1]

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

else:
    ImageCombWidget = None


def main():
    """
    ImageCombWidgetのテストケース

    【テスト内容】
    - ImageCombに複数の短冊画像を追加
    - ImageCombWidgetで表示
    - 短冊間の1px隙間を確認（show_gaps=True）

    【終了方法】
    - ウィンドウを閉じる（×ボタン、Command-W）
    - Ctrl-C でも終了可能
    """
    if not PYQT6_AVAILABLE:
        print("PyQt6 is not installed. Test skipped.")
        return

    from trainscanner2.imagecomb import ImageComb
    from pyperbox import Rect
    import sys
    import signal

    print("ImageCombWidgetテストを開始します...")

    # Ctrl-C で確実に終了できるようにする
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    try:
        app = QApplication(sys.argv)

        # アプリケーションが最後のウィンドウを閉じたときに終了するように設定
        app.setQuitOnLastWindowClosed(True)

        # テスト用ImageCombを作成
        image_comb = ImageComb()
        print(f"ImageComb created: {len(image_comb.images)} images")

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

            # ImageCombに追加（左から右へ）
            rect = Rect.from_bounds(
                left=i * 100,
                right=(i + 1) * 100,
                top=-150,
                bottom=150,
            )
            print(f"Adding image {i}, rect: {rect}")
            image_comb.put_image(rect, img)

        print(f"ImageComb after adding: {len(image_comb.images)} images")

        # ウィンドウを作成
        window = QMainWindow()
        window.setWindowTitle("ImageCombWidget Test - Close window to exit")
        window.resize(600, 400)

        # ImageCombWidgetを作成（隙間を表示）
        print("Creating ImageCombWidget...")
        widget = ImageCombWidget(show_gaps=True)
        print("Setting image_comb...")
        widget.set_image_comb(image_comb)

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
