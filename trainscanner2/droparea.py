import os
from logging import getLogger
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtGui import (
    QDragEnterEvent,
    QDragMoveEvent,
    QDragLeaveEvent,
    QDropEvent,
    QPainter,
    QPen,
    QColor,
    QKeySequence,
)
from PyQt6.QtCore import Qt, QTimer

class DropAreaWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.parent_window = parent
        # 最初から全画面を覆うように設定（親がQMainWindowの場合）
        if parent:
            self.resize(parent.size())
        # マウスイベントを透過させない（ドロップを確実に奪うため）
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        # キーイベントを受け取れるようにする
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 背景を少し暗く透過させる（ドラッグ中や初期状態で見やすくするため）
        painter.fillRect(self.rect(), QColor(255, 255, 255, 180))

        # 破線のペンを設定
        pen = QPen(QColor(100, 150, 255), 4, Qt.PenStyle.DashLine)
        painter.setPen(pen)

        # 枠を描画（余白を少し持たせる）
        margin = 20
        rect = self.rect().adjusted(margin, margin, -margin, -margin)
        painter.drawRect(rect)

        # テキストを描画
        painter.setPen(QColor(50, 100, 200))
        font = painter.font()
        font.setPointSize(24)
        font.setBold(True)
        painter.setFont(font)

        main_text = "動画ファイルをここにドロップ"
        sub_text = "または YouTube URL をペースト"

        # メインテキスト
        text_rect = painter.fontMetrics().boundingRect(main_text)
        text_x = (self.width() - text_rect.width()) // 2
        text_y = self.height() // 2 - 20
        painter.drawText(text_x, text_y, main_text)

        # サブテキスト
        font.setPointSize(14)
        font.setBold(False)
        painter.setFont(font)
        sub_text_rect = painter.fontMetrics().boundingRect(sub_text)
        sub_x = (self.width() - sub_text_rect.width()) // 2
        sub_y = text_y + 40
        painter.drawText(sub_x, sub_y, sub_text)

    def keyPressEvent(self, event):
        # ペースト操作 (Cmd+V / Ctrl+V) をハンドル
        if event.matches(QKeySequence.StandardKey.Paste):
            if self.handle_paste():
                return
        super().keyPressEvent(event)

    def handle_paste(self) -> bool:
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()

        target = None
        # 1. クリップボードにURL（ファイルパス含む）が含まれている場合
        if mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                file_path = urls[0].toLocalFile()
                url_str = urls[0].toString()
                if file_path and self._is_valid_video_file(file_path):
                    target = file_path
                elif self._is_youtube_url(url_str):
                    target = url_str

        # 2. テキストとしてパスやURLが貼り付けられた場合
        if not target and mime_data.hasText():
            text = mime_data.text().strip()
            # 引用符で囲まれている場合を考慮
            if (text.startswith('"') and text.endswith('"')) or (
                text.startswith("'") and text.endswith("'")
            ):
                text = text[1:-1]

            if self._is_youtube_url(text):
                target = text
            elif os.path.exists(text) and self._is_valid_video_file(text):
                target = text

        if target:
            if self.parent_window:
                self.parent_window.processing_started = True
                self.hide()
                # 100msの遅延を入れてOSのクリーンアップ時間を確保してから開始指示
                QTimer.singleShot(
                    100, lambda: self.parent_window.start_processing(target)
                )
            return True
        return False

    def _is_valid_video_file(self, file_path: str) -> bool:
        video_extensions = [
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".mts",
            ".m2ts",
            ".ts",
            ".m4v",
            ".webm",
        ]
        return any(file_path.lower().endswith(ext) for ext in video_extensions)

    def _is_youtube_url(self, url: str) -> bool:
        # YouTubeやその他のサポートされているURLか簡易チェック
        # file:// で始まるものはローカルファイルとして扱うため除外
        if url.startswith("file://"):
            return False
        return "youtube.com/" in url or "youtu.be/" in url or url.startswith("http")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                local_file = urls[0].toLocalFile()
                url_str = urls[0].toString()
                if (
                    local_file and self._is_valid_video_file(local_file)
                ) or self._is_youtube_url(url_str):
                    # ドラッグが入ってきたら自分を表示する（オーバーレイ効果）
                    self.show()
                    self.raise_()
                    event.setDropAction(Qt.DropAction.CopyAction)
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent):
        event.acceptProposedAction()

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        # 処理中の場合は、ドラッグが外れたら隠す
        if self.parent_window and getattr(
            self.parent_window, "processing_started", False
        ):
            self.hide()
        event.accept()

    def dropEvent(self, event: QDropEvent):
        logger = getLogger(__name__)
        # 処理中でもドロップを受け付けるように変更（中断フラグで制御するためチェックを削除）

        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                url_str = urls[0].toString()

                target = None
                if file_path:
                    # ローカルパスが取得できれば優先
                    if self._is_valid_video_file(file_path):
                        target = file_path
                elif url_str.startswith("file://"):
                    # toLocalFile()が空でもfile://で始まる場合はデコードを試みる
                    from urllib.parse import unquote

                    decoded_path = url_str[7:]  # skip file://
                    if decoded_path and self._is_valid_video_file(decoded_path):
                        target = unquote(decoded_path)
                elif self._is_youtube_url(url_str):
                    target = url_str

                if target:
                    event.setDropAction(Qt.DropAction.CopyAction)
                    event.acceptProposedAction()
                    # ドロップされたら隠す
                    self.hide()
                    if self.parent_window:
                        # 処理開始フラグを立てる
                        self.parent_window.processing_started = True
                        # 100msの遅延を入れてOSのクリーンアップ時間を確保してから開始指示
                        QTimer.singleShot(
                            100, lambda: self.parent_window.start_processing(target)
                        )
                    return
        event.ignore()
        # 不正なファイルなどの場合も隠す
        if self.parent_window and getattr(
            self.parent_window, "processing_started", False
        ):
            self.hide()

