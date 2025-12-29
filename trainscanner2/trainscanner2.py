import sys
import os
import json
import numpy as np
import tempfile
import yt_dlp
from logging import getLogger, INFO, DEBUG, WARNING, basicConfig
from tqdm import tqdm

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtCore import QTimer

from trainscanner2.video import video_loader_factory
from trainscanner2.analyze import analyze_iter
from trainscanner2.detect import MotionDetector
from trainscanner2.antishake import AntiShaker2
from trainscanner2.render import Render
from trainscanner2.droparea import DropAreaWidget


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
            video_path=os.path.abspath(videofile),  # 確実に絶対パスで記録
            video_base=video_base,
            scaling_factor=scale,
            use_pyqt=False,
            use_multiview=False,
        )
        renderer.multiview_manager = multiview_manager
        renderer.window_manager = None
    else:
        renderer = Render(
            video_path=os.path.abspath(videofile),  # 確実に絶対パスで記録
            video_base=video_base,
            scaling_factor=scale,
            use_multiview=True,
        )

    antishaker = AntiShaker2(velocity=1)

    def iterator():
        for (
            frame_index,
            absolute_position,
            matchscore,
            scaled_frame,
            diff,
            mask,
        ) in analyze_iter(vl, scaling_ratio=scale, antishaker=antishaker):
            yield frame_index, absolute_position, matchscore, scaled_frame, diff, mask

    logger.info("Starting video processing...")
    motiondetector = MotionDetector()

    for frame_index, absolute_position, matchscore, frame, diff, mask in iterator():
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
            if logger.getEffectiveLevel() <= INFO:
                multiview_manager.update_diff(diff)
                multiview_manager.update_mask(mask)
            else:
                # 非表示にする
                multiview_manager.hide_verbose_previews()

        paths, dropped_paths, active_path_ids = motiondetector._detect(
            matchscore, frame_index=frame_index
        )

        if len(paths) == 0:
            antishaker.abs_loc = (0, 0)

        for id, path in paths.items():
            renderer.put(
                id, frame, path.history[-1], absolute_position=absolute_position
            )

        if hasattr(renderer, "multiview_manager") and renderer.multiview_manager:
            renderer.multiview_manager.set_active_paths(active_path_ids)
            renderer.multiview_manager.app.processEvents()

        for path_id in dropped_paths:
            renderer.mark_inactive(id=path_id)

    if multiview_manager:
        multiview_manager.set_done()

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
        # 音声は不要なので、ビデオのみの最高画質を指定
        "format": "bestvideo/best",
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
            # マージなどで拡張子が変更された可能性を考慮して再検索
            base, _ = os.path.splitext(downloaded_file)
            for ext in [".mkv", ".mp4", ".webm"]:
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


def main():
    if "-d" in sys.argv or "--debug" in sys.argv:
        basicConfig(level=DEBUG)
    elif "-v" in sys.argv or "--verbose" in sys.argv:
        basicConfig(level=INFO)
    else:
        # デフォルトではWARNINGに設定して、並列スレッド内からのGUI操作(imshow等)によるクラッシュを防ぐ
        basicConfig(level=WARNING)

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

        # メインウィンドウでもペースト（Ctrl+V / Cmd+V）をいつでも受け付けるように設定
        # これにより DropArea が非表示の時（処理中など）でもペーストでファイルを変更できる
        paste_shortcut = QShortcut(QKeySequence.StandardKey.Paste, manager.window)
        paste_shortcut.activated.connect(drop_area.handle_paste)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
