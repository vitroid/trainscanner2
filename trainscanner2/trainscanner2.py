import sys
import os
import json
import numpy as np
from logging import getLogger, INFO, DEBUG, WARNING, basicConfig
from tqdm import tqdm

try:
    from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
    from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDragLeaveEvent, QDropEvent, QPainter, QPen, QColor
    from PyQt6.QtCore import Qt, QTimer, QMimeData
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False

from trainscanner2.video import video_loader_factory
from trainscanner2.analyze import analyze_iter
from trainscanner2.detect import MotionDetector
from trainscanner2.render import Render

def process_video(videofile: str, multiview_manager=None, wait=False):
    """
    動画ファイルを処理する関数
    """
    logger = getLogger(__name__)
    logger.info(f"process_video starting for: {videofile}")

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
        video_base = os.path.splitext(videofile)[0]
        multiview_manager.set_video_base(video_base)
        renderer = Render(video_path=videofile, scaling_factor=scale, use_pyqt=False, use_multiview=False)
        renderer.multiview_manager = multiview_manager
        renderer.window_manager = None
    else:
        renderer = Render(video_path=videofile, scaling_factor=scale, use_multiview=True)

    def iterator():
        for frame_index, absolute_position, matchscore, scaled_frame in analyze_iter(vl, scaling_ratio=scale):
            yield frame_index, absolute_position, matchscore, scaled_frame

    logger.info("Starting video processing...")
    motiondetector = MotionDetector()

    for frame_index, absolute_position, matchscore, frame in iterator():
        # 中断チェック：新しいファイルがドロップされたら即座に終了する
        if multiview_manager and hasattr(multiview_manager.window, "interrupted") and multiview_manager.window.interrupted:
            logger.info("Processing interrupted by new drop.")
            return

        if multiview_manager:
            multiview_manager.update_preview(frame)

        paths, dropped_paths, active_path_ids = motiondetector._detect(
            matchscore, frame_index=frame_index
        )

        for id, path in paths.items():
            renderer.put(id, frame, path.history[-1], absolute_position=absolute_position)

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
            video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.mts', '.m2ts', '.ts', '.m4v']
            return any(file_path.lower().endswith(ext) for ext in video_extensions)
        
        def dragEnterEvent(self, event: QDragEnterEvent):
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                if urls and self._is_valid_video_file(urls[0].toLocalFile()):
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
            if self.parent_window and getattr(self.parent_window, 'processing_started', False):
                self.hide()
            event.accept()
                
        def dropEvent(self, event: QDropEvent):
            logger = getLogger(__name__)
            # 処理中でもドロップを受け付けるように変更（中断フラグで制御するためチェックを削除）
            
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                if urls:
                    file_path = urls[0].toLocalFile()
                    if self._is_valid_video_file(file_path):
                        event.setDropAction(Qt.DropAction.CopyAction)
                        event.acceptProposedAction()
                        # ドロップされたら隠す
                        self.hide()
                        if self.parent_window:
                            # 処理開始フラグを立てる
                            self.parent_window.processing_started = True
                            # 100msの遅延を入れてOSのクリーンアップ時間を確保してから開始指示
                            QTimer.singleShot(100, lambda: self.parent_window.start_processing(file_path))
                        return
            event.ignore()
            # 不正なファイルなどの場合も隠す
            if self.parent_window and getattr(self.parent_window, 'processing_started', False):
                self.hide()

def main():
    if '-d' in sys.argv or '--debug' in sys.argv:
        basicConfig(level=DEBUG)
    elif '-v' in sys.argv or '--verbose' in sys.argv:
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
            if hasattr(manager.window, '_original_resizeEvent'):
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
            if urls and drop_area._is_valid_video_file(urls[0].toLocalFile()):
                drop_area.show()
                drop_area.raise_()
                event.acceptProposedAction()
        manager.window.dragEnterEvent = window_dragEnterEvent

        def start_processing(videofile: str):
            logger = getLogger(__name__)
            
            # もし既に処理中なら、中断フラグを立てて、少し待ってから再試行する
            if manager.window.processing:
                logger.info("Already processing. Signaling interruption and rescheduling...")
                manager.window.interrupted = True
                # 現在のループが終了して processing = False になるまで待って再試行
                QTimer.singleShot(200, lambda: start_processing(videofile))
                return

            # フラグを初期化して処理開始
            manager.window.interrupted = False
            manager.window.processing = True
            
            # ドロップエリアを隠す（処理中はパネルを見せる）
            drop_area.hide()
            
            try:
                process_video(videofile, multiview_manager=manager, wait=False)
            finally:
                # 処理が終わったら（または中断されたら）フラグを下ろす
                manager.window.processing = False
                logger.info("Ready for next drop.")
        
        manager.window.start_processing = start_processing
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
