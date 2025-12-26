import sys
import os
import json
import numpy as np
import tempfile
import yt_dlp
from logging import getLogger, INFO, DEBUG, WARNING, basicConfig
from tqdm import tqdm

try:
    from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
    from PyQt6.QtGui import (
        QDragEnterEvent,
        QDragMoveEvent,
        QDragLeaveEvent,
        QDropEvent,
        QPainter,
        QPen,
        QColor,
    )
    from PyQt6.QtCore import Qt, QTimer, QMimeData

    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False

from trainscanner2.video import video_loader_factory
from trainscanner2.analyze import analyze_iter
from trainscanner2.detect import MotionDetector
from trainscanner2.render import Render


def process_video(videofile: str, multiview_manager=None, wait=False, video_base=None):
    """
    動画ファイルを処理する関数
    """
    logger = getLogger(__name__)
    logger.info(f"process_video starting for: {videofile}")

    # 保存時のベース名（パスを含む）を決定
    if video_base is None:
        video_base = os.path.splitext(videofile)[0]
        # URLドロップなどで一時フォルダにある場合は、現在のディレクトリに保存するようにする
        if "temp" in videofile or videofile.startswith(tempfile.gettempdir()):
            video_base = os.path.basename(video_base)

    vl = video_loader_factory(videofile)
    total_frames = vl.total_frames()
    logger.info(f"Video loaded. Total frames: {total_frames}")
    frame = vl.next()
    scale = (300 * 300 / (frame.shape[0] * frame.shape[1])) ** 0.5
    if scale > 1.0:
        scale = 1.0

    # 既存のMultiViewManagerがある場合は、それを使用
    if multiview_manager is not None:
        multiview_manager.clear_all_paths()
        multiview_manager.set_video_base(video_base)
        renderer = Render(
            video_path=videofile,
            video_base=video_base,
            scaling_factor=scale,
            use_pyqt=False,
            use_multiview=False,
        )
        renderer.multiview_manager = multiview_manager
        renderer.window_manager = None
    else:
        renderer = Render(
            video_path=videofile,
            video_base=video_base,
            scaling_factor=scale,
            use_multiview=True,
        )

    def iterator():
        for frame_index, absolute_position, matchscore, scaled_frame in analyze_iter(
            vl, scaling_ratio=scale
        ):
            yield frame_index, absolute_position, matchscore, scaled_frame

    logger.info("Starting video processing...")
    motiondetector = MotionDetector()

    for frame_index, absolute_position, matchscore, frame in iterator():
        # 中断チェック：新しいファイルがドロップされたら即座に終了する
        if (
            multiview_manager
            and hasattr(multiview_manager.window, "interrupted")
            and multiview_manager.window.interrupted
        ):
            logger.info("Processing interrupted by new drop.")
            return

        if multiview_manager:
            multiview_manager.update_preview(frame)

        paths, dropped_paths, active_path_ids = motiondetector._detect(
            matchscore, frame_index=frame_index
        )

        for id, path in paths.items():
            renderer.put(
                id, frame, path.history[-1], absolute_position=absolute_position
            )

        if hasattr(renderer, "multiview_manager") and renderer.multiview_manager:
            renderer.multiview_manager.set_active_paths(active_path_ids)
            renderer.multiview_manager.app.processEvents()

        for path_id in dropped_paths:
            renderer.mark_inactive(id=path_id)

    all_detected_paths = dict(motiondetector.paths)
    for path_id, history in motiondetector.done():
        renderer.done(id=path_id)

    # JSON保存処理などは省略せず維持
    logger.info("Saving path data...")
    try:
        all_paths_data = {}
        for path_id, render_one in renderer.renderers.items():
            if render_one is not None:
                path_data = render_one.export_history()
                all_paths_data[str(path_id)] = path_data

        if all_paths_data:
            dump_file = os.path.splitext(videofile)[0] + ".ts2dump"
            with open(dump_file, "w", encoding="utf-8") as f:
                json.dump(all_paths_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save paths data: {e}")

    if wait:
        logger.info("Waiting for windows to close...")
        renderer.wait_for_windows_close()

    logger.info("process_video finished.")


def download_from_url(url: str, progress_callback=None) -> tuple[str, str]:
    """
    YouTubeなどのURLから動画を一時ファイルにダウンロードする

    Returns:
        tuple[str, str]: (ダウンロードされたファイルのパス, 動画のタイトル)
    """
    logger = getLogger(__name__)
    logger.info(f"Downloading from URL: {url}")

    # 一時ファイルのパスを作成
    temp_dir = tempfile.gettempdir()
    # yt-dlpに一時ファイル名を使わせるためのテンプレート
    # 拡張子はyt-dlpが自動で付与するので、ベース名だけ指定
    outtmpl = os.path.join(temp_dir, "ts2_download_%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
    }

    if progress_callback:
        ydl_opts["progress_hooks"] = [progress_callback]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # 実際に保存されたファイルパスを取得
        downloaded_file = ydl.prepare_filename(info)
        # info['ext'] が実際と異なる場合があるため（マージ後など）、
        # 実際にファイルが存在するか、あるいはマージ後の名前を確認する
        if not os.path.exists(downloaded_file):
            # webmなどがmp4にマージされた場合、拡張子がmp4に変わっている可能性がある
            base, _ = os.path.splitext(downloaded_file)
            for ext in [".mp4", ".mkv", ".webm"]:
                if os.path.exists(base + ext):
                    downloaded_file = base + ext
                    break

        # 動画のタイトルを取得して、ファイル名として安全な形式に変換
        title = info.get("title", "youtube_video")
        safe_title = "".join(
            [c for c in title if c.isalnum() or c in (" ", ".", "_", "-")]
        ).strip()

        logger.info(f"Download complete: {downloaded_file} (Title: {safe_title})")
        return downloaded_file, safe_title


if PYQT6_AVAILABLE:

    class DropAreaWidget(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setAcceptDrops(True)
            self.parent_window = None
            # 最初から全画面を覆うように設定（親がQMainWindowの場合）
            if parent:
                self.resize(parent.size())
            # マウスイベントを透過させない（ドロップを確実に奪うため）
            self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

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
            text = "動画ファイルをここにドロップ"
            text_rect = painter.fontMetrics().boundingRect(text)
            text_x = (self.width() - text_rect.width()) // 2
            text_y = self.height() // 2
            painter.drawText(text_x, text_y, text)

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


def main():
    if "-d" in sys.argv or "--debug" in sys.argv:
        basicConfig(level=DEBUG)
    elif "-v" in sys.argv or "--verbose" in sys.argv:
        basicConfig(level=INFO)
    else:
        basicConfig(level=WARNING)

    if not PYQT6_AVAILABLE:
        sys.exit(1)

    app = QApplication.instance() or QApplication(sys.argv)

    from trainscanner2.multiview import MultiViewManager

    manager = MultiViewManager(video_base=None, show_gaps=False, show_buttons=True)

    if manager.window:
        # DropAreaWidgetをオーバーレイとして作成
        drop_area = DropAreaWidget(manager.window)
        drop_area.parent_window = manager.window

        # ウィンドウのリサイズに合わせてDropAreaもリサイズする仕掛け
        def resize_drop_area(event):
            drop_area.resize(manager.window.size())
            if hasattr(manager.window, "_original_resizeEvent"):
                manager.window._original_resizeEvent(event)

        manager.window._original_resizeEvent = manager.window.resizeEvent
        manager.window.resizeEvent = resize_drop_area

        # 初期表示
        drop_area.show()
        drop_area.raise_()

        # 処理中フラグ
        manager.window.processing = False
        # 中断フラグ
        manager.window.interrupted = False
        manager.window.processing_started = False

        # ウィンドウ自体でもドラッグを受け取れるようにし、DropAreaを表示する
        manager.window.setAcceptDrops(True)

        def window_dragEnterEvent(event):
            # 処理中であってもドラッグを受け入れる
            urls = event.mimeData().urls()
            if urls:
                local_file = urls[0].toLocalFile()
                url_str = urls[0].toString()
                if (
                    local_file and drop_area._is_valid_video_file(local_file)
                ) or drop_area._is_youtube_url(url_str):
                    drop_area.show()
                    drop_area.raise_()
                    event.acceptProposedAction()

        manager.window.dragEnterEvent = window_dragEnterEvent

        def start_processing(videofile: str):
            logger = getLogger(__name__)

            # もし既に処理中なら、中断フラグを立てて、少し待ってから再試行する
            if manager.window.processing:
                logger.info(
                    "Already processing. Signaling interruption and rescheduling..."
                )
                manager.window.interrupted = True
                # 現在のループが終了して processing = False になるまで待って再試行
                QTimer.singleShot(200, lambda: start_processing(videofile))
                return

            # フラグを初期化して処理開始
            manager.window.interrupted = False
            manager.window.processing = True

            # ドロップエリアを隠す（処理中はパネルを見せる）
            drop_area.hide()

            # URLの場合はダウンロードする
            actual_video_file = videofile
            video_base = None
            if videofile.startswith("http"):
                try:
                    # ダウンロード進捗を表示するための簡易フック
                    def download_hook(d):
                        if d["status"] == "downloading":
                            p = d.get("_percent_str", "0%")
                            logger.info(f"Downloading YouTube video: {p}")
                            # UIのスレッドを止めないように
                            QApplication.processEvents()

                    actual_video_file, video_base = download_from_url(
                        videofile, progress_callback=download_hook
                    )
                except Exception as e:
                    logger.error(f"Failed to download video from URL: {e}")
                    manager.window.processing = False
                    return

            try:
                process_video(
                    actual_video_file,
                    multiview_manager=manager,
                    wait=False,
                    video_base=video_base,
                )
            finally:
                # 処理が終わったら（または中断されたら）フラグを下ろす
                manager.window.processing = False
                logger.info("Ready for next drop.")

        manager.window.start_processing = start_processing

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
