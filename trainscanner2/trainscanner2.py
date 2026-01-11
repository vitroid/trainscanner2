import sys
import os
import json
import numpy as np
import tempfile
import yt_dlp
from logging import getLogger, INFO, DEBUG, WARNING, basicConfig
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtCore import QTimer

from trainscanner2.video import video_loader_factory
from trainscanner2.analyze import analyze_iter
from trainscanner2.detect import MotionDetector
from trainscanner2.antishake import AntiShaker2
from trainscanner2.render import Render


def process_video(videofile: str, wait=False, video_base=None):
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

    # PyQt6を使用（WindowManagerのタイマーでメインスレッドからWindow更新を実行）
    renderer = Render(
        use_pyqt=True,  # PyQt6を使用（Window更新はメインスレッドで実行される）
        video_path=os.path.abspath(videofile),  # 確実に絶対パスで記録
        video_base=video_base,
        scaling_factor=scale,
    )

    antishaker = AntiShaker2(velocity=1)

    def iterator():
        for (
            frame_index,
            absolute_position,
            matchscore,
            unblurred_scaled_frame,  # 手ぶれ補正後のフレーム
        ) in analyze_iter(vl, scaling_ratio=scale, antishaker=antishaker):
            yield frame_index, absolute_position, matchscore, unblurred_scaled_frame

    logger.info("Starting video processing...")
    motiondetector = MotionDetector()

    # 各Pathを独立したworkerで処理するための管理
    path_workers = {}  # {path_id: Render_one}

    # 並列処理用のThreadPoolExecutor
    max_workers = (
        min(len(path_workers) + 4, os.cpu_count() or 4)
        if path_workers
        else os.cpu_count() or 4
    )
    executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_frame(frame_index, absolute_position, matchscore, frame):
        """1フレームの処理"""
        # プレビューウィンドウに手ぶれ補正後のフレームを表示
        if renderer.window_manager:
            try:
                renderer.window_manager.update_preview_window(frame)
            except Exception as e:
                logger.debug(f"Preview window update error: {e}")

        paths, dropped_paths, active_path_ids = motiondetector._detect(
            matchscore, frame_index=frame_index
        )

        if len(paths) == 0:
            antishaker.abs_loc = (0, 0)

        # 各Pathの処理を並列実行（処理部分のみ、Window更新は除く）
        # OpenCVのcv2.imshow()はスレッドセーフではないため、Window更新はメインスレッドで実行
        futures = {}
        for id, path in paths.items():
            # 既存のworkerがある場合は使用、なければ新規作成
            if id not in path_workers:
                # Render_oneを作成（各Pathごとに独立）
                render_one = renderer._create_render_one(id)
                path_workers[id] = render_one

            # 各Pathのworkerで処理（並列実行、Window更新は除く）
            render_one = path_workers[id]
            future = executor.submit(
                render_one.put_process_only,
                frame=frame.copy() if hasattr(frame, "copy") else frame,
                pathitem=path.history[-1],
                absolute_position=absolute_position,
            )
            futures[id] = future

        # すべての処理が完了するのを待つ
        for id, future in futures.items():
            try:
                future.result()  # 完了を待つ（エラーがあれば例外を発生）
            except Exception as e:
                logger.error(f"Error processing path {id}: {e}")

        # Window更新はメインスレッドで実行（PyQt6の場合、WindowManagerが管理）
        # PyQt6の場合はupdate_window()を呼ぶ必要がある
        for id, path in paths.items():
            if id in path_workers:
                try:
                    path_workers[id].update_window_display()
                except Exception as e:
                    logger.error(f"Error updating window for path {id}: {e}")

        # WindowManagerのタイマーが動作するようにprocessEvents()を呼ぶ
        if renderer.app:
            renderer.app.processEvents()

        for path_id in dropped_paths:
            renderer.mark_inactive(id=path_id)
            # workerから削除
            if path_id in path_workers:
                del path_workers[path_id]

    try:
        for frame_index, absolute_position, matchscore, frame in iterator():
            process_frame(frame_index, absolute_position, matchscore, frame)
    finally:
        # すべての並列処理が完了するのを待つ
        executor.shutdown(wait=True)

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
    logger = getLogger(__name__)
    # 引数の解析（簡易版）
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if "-d" in sys.argv or "--debug" in sys.argv:
        basicConfig(level=DEBUG)
    elif "-v" in sys.argv or "--verbose" in sys.argv:
        basicConfig(level=INFO)
    else:
        # デフォルトではWARNINGに設定
        basicConfig(level=WARNING)

    if not args:
        print(f"Usage: python {sys.argv[0]} [options] <video_file_or_url>")
        sys.exit(1)

    videofile = args[0]
    app = QApplication.instance() or QApplication(sys.argv)

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
            sys.exit(1)

    try:
        process_video(
            actual_video_file,
            wait=True,  # コマンドライン実行時は全ウィンドウが閉じるまで待機
            video_base=video_base,
        )
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
