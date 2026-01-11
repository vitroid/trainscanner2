import numpy as np
import cv2
from logging import getLogger, INFO
import os
import json

from trainscanner2 import FIFO, PathItem
from trainscanner2.imagestrips import ImageStrips
from trainscanner2.window import WindowManager


# ImageWindowとWindowManagerはwindow.pyに移動しました


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
        video_base: str = None,
        cache: bool = False,
    ):
        self.num_leading_frames = num_leading_frames
        self.leading_frames = FIFO(num_leading_frames)
        self.pathitem_history = []  # PathItemのリスト
        self.abs_positions = []  # 各フレームのabsolute_position（手ぶれ補正）
        self.train_positions = []  # 各フレームでのtrain_position（再計算不要に）
        self.id = id
        self.canvas = None  # 遅延初期化
        self.cache = cache
        self.first = False
        self.train_position = 0
        self.alive = True
        self.window_manager = window_manager
        self.window = None
        self.scaling_factor = scaling_factor  # 低解像度→高解像度への変換係数
        self.video_path = video_path  # 動画ファイルパス（高解像度再スキャン用）
        self.video_base = video_base  # 保存時のベース名

    def done(self):
        self.alive = False

        # ユーザー要求: スコアが0.1未満、かつフレーム数が50以下の場合はウィンドウを閉じる
        history_len = len(self.pathitem_history)
        score = self.score
        should_close = score < 0.1 and history_len <= 50

        if self.window_manager and self.window:
            if should_close:
                self.logger.info(
                    f"Closing window {self.id} due to low score ({score:.3f}) "
                    f"and short history ({history_len})"
                )
                self.window_manager.close_window(self.id)
            else:
                # ウィンドウのタイトルを「処理完了」に更新
                self.window_manager.set_window_finished(self.id, score)
        else:
            # OpenCVウィンドウのみ自動で閉じる（存在しない場合はエラーを無視）
            try:
                cv2.destroyWindow(f"{self.id}")
                cv2.waitKey(1)
            except cv2.error:
                # ウィンドウが存在しない場合は無視
                pass

    def _render_one(
        self,
        frame: np.ndarray,
        h: PathItem,
    ):
        # ImageStripsを遅延初期化
        if self.canvas is None:
            self.canvas = ImageStrips(cache=self.cache)

        velocity = h.xy
        self.logger.debug(f"{id=} {velocity=} ")
        dx, dy = velocity
        # hopはdropped frameによる飛び
        dx *= h.hop
        dy *= h.hop
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
        absolute_position=None,
    ):
        # renderを開始するまでに「溜める」時間
        delay_frames = 10
        if not self.alive:
            return
        self.pathitem_history.append(pathitem)
        self.abs_positions.append(
            absolute_position if absolute_position is not None else (0, 0)
        )
        self.leading_frames.append(frame)
        if len(self.pathitem_history) > self.num_leading_frames:
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
                        f"Score: {self.score:.3f}",
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
                    else:
                        # PyQt6が使用されていない場合はOpenCVを使用
                        cv2.imshow(f"{self.id}", img)
        elif len(self.pathitem_history) == delay_frames:
            for f, pi in zip(self.leading_frames.queue, self.pathitem_history):
                self._render_one(f, pi)

    @property
    def score(self):
        """Path全体の平均スコアを返す（ソートと表示用）"""
        if not self.pathitem_history:
            return 0.0
        return np.mean([h.value for h in self.pathitem_history])

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
                - score: 平均品質スコア
                - scaling_factor: 低解像度→高解像度への変換係数
                  （実際の変位 = delta * (1/scaling_factor)）
                - history: フレームごとの詳細情報のリスト
                    * frame_index: ビデオのフレーム番号
                    * match_score: マッチングスコア
                    * delta_x, delta_y: 直前のコマからの変位ベクトル（スケール済み）
                    * train_position: そのフレームでのキャンバス位置
                    * abs_pos_x, abs_pos_y: 手ぶれによる背景移動量
        """
        # ImageStripsを遅延初期化
        if self.canvas is None:
            self.canvas = ImageStrips(cache=self.cache)

        # 【重要】train_positionは再計算せず、_render_one()で記録した値を使う
        # これにより、コードの重複を避け、計算ミスを防ぐ
        history_data = []

        for idx, h in enumerate(self.pathitem_history):
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
                    "frame_index": int(h.frame_index),
                    "match_score": float(h.value),
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
            "score": float(self.score),
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
        - {base_path}.png: stitchされた画像
        - {base_path}.tspos2: stitching履歴（JSON形式）

        Args:
            base_path: ファイルのベースパス（拡張子なし）
                      Noneの場合、video_base または video_path + "_" + id を使用
        """
        # ベースパスを決定
        if base_path is None:
            if self.video_base:
                base_path = f"{self.video_base}_{self.id}"
            elif self.video_path:
                video_base = os.path.splitext(self.video_path)[0]
                base_path = f"{video_base}_{self.id}"
            else:
                base_path = f"train_scan_{self.id}"

        image_path = f"{base_path}.png"
        history_path = f"{base_path}.tspos2"

        # 画像を保存（メモリ効率的）
        if self.canvas is None:
            self.logger.warning("No canvas to save")
            return None, None

        if isinstance(self.canvas, ImageStrips):
            self.logger.debug(
                f"Saving image to {image_path} (memory-efficient mode)..."
            )
            self.canvas.save_to_file(image_path)
        else:
            # 従来のSimpleImage形式
            img = self.canvas.get_image()
            if img is not None:
                cv2.imwrite(image_path, img)
                self.logger.debug(f"Saved image to {image_path}")

        # stitching履歴を保存
        try:
            history_data = self.export_history()
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved history to {history_path}")
        except Exception as e:
            import traceback

            self.logger.error(f"エラー: stitching履歴の保存に失敗しました: {e}")
            traceback.print_exc()
            # sys.exit(1)を削除し、例外を再発生させる
            raise

        return image_path, history_path


# WindowManagerもwindow.pyに移動しました


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
    3. 処理完了後に低品質Pathを一括削除 (close_low_score_windows())
    """

    logger = getLogger(__name__)

    def __init__(
        self,
        use_pyqt=True,
        video_path: str = None,
        scaling_factor: float = 1.0,
        show_gaps=False,
        show_buttons=True,  # ウィンドウに保存ボタンを表示するか
        video_base: str = None,
    ):
        self.renderers = {}  # {id: Render_one} 全Path（active/finished両方）
        self.max_score = 0.0
        self.window_manager = None
        self.scaling_factor = scaling_factor  # 低解像度→高解像度への変換係数
        self.video_path = video_path  # 動画ファイルパス
        self.video_base = video_base  # 保存時のベース名

        if use_pyqt:
            # 動画ファイルパスからベース名を取得
            vb = video_base
            if vb is None and video_path:
                # 拡張子を除いたベース名を取得
                vb = os.path.splitext(video_path)[0]

            try:
                # 常に個別ウィンドウを使用
                self.window_manager = WindowManager(
                    video_base=vb,
                    renderer_callback=self.remove_renderer,
                    show_gaps=show_gaps,  # デバッグ用
                    show_buttons=show_buttons,  # ボタン表示設定
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize window manager: {e}")
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
                num_leading_frames=10,
                window_manager=self.window_manager,
                scaling_factor=self.scaling_factor,
                video_path=self.video_path,
                video_base=self.video_base,
            )
            self.renderers[id] = r

        r.put(
            frame=frame,
            pathitem=historyitem,
            absolute_position=absolute_position,
        )

        q = r.score
        # 最高品質が更新されたら、低品質ウィンドウをチェックして閉じる
        if self.max_score < q:
            self.max_score = q
            self.logger.debug(f"Max score updated: {self.max_score}")

        # 閾値を下げる。あるいはこれを使わないほうがいいかも
        self.max_score *= 0.995

    def _create_render_one(self, id: int) -> Render_one:
        """
        新しいRender_oneインスタンスを作成する

        Args:
            id: PathのID

        Returns:
            Render_one: 新しく作成されたRender_oneインスタンス
        """
        r = Render_one(
            id,
            num_leading_frames=10,
            window_manager=self.window_manager,
            scaling_factor=self.scaling_factor,
            video_path=self.video_path,
            video_base=self.video_base,
        )
        self.renderers[id] = r
        return r

    @property
    def app(self):
        if self.window_manager:
            return self.window_manager.app
        return None

    def mark_inactive(self, id: int):
        """
        MotionDetector.pathsから除外されたPathを「非アクティブ」として扱う。
        """
        if id not in self.renderers:
            return

        # パスが途切れた時点で処理完了（done）とする。
        # ここでスコア判定が行われ、低品質なものはウィンドウが閉じる。
        self.done(id)

    def done(self, id):
        """
        Pathの処理を終了

        【呼ばれるタイミング】
        - 動画処理完了時（全Pathに対して）
        - 品質が閾値以下の場合（将来的に）

        【動作】
        1. ウィンドウを閉じる（または閉じない）
        2. Pathを削除（メモリ解放、以降の処理をスキップ）

        Args:
            id: PathのID
        """
        if id not in self.renderers:
            return

        r = self.renderers[id]
        # r.done() の中でウィンドウが閉じられると、
        # コールバック経由で self.renderers[id] が削除される可能性があるため、
        # 実行前に一旦参照を保持し、実行後に存在確認を行う。
        r.done()

        # Pathを削除（まだ残っている場合のみ）
        if id in self.renderers:
            del self.renderers[id]

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
            self.logger.debug(f"Removed renderer {id}")

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
                - max_score: 最高品質
        """
        active = self.get_active_paths()
        finished = self.get_finished_paths()
        return {
            "total": len(self.renderers),
            "active": len(active),
            "finished": len(finished),
            "max_score": self.max_score,
        }

    def _check_and_close_low_score_windows(self, score_ratio: float = 0.5):
        """
        低品質ウィンドウをチェックして閉じる（処理中に随時実行）

        【目的】
        - 最高品質が更新されるたびに、閾値以下のウィンドウを閉じる
        - ウィンドウが無限に増え続けないようにする

        【呼ばれるタイミング】
        - Render.put()でmax_scoreが更新されたとき

        Args:
            score_ratio: 最高品質に対する比率（デフォルト: 0.5 = 50%）
        """
        if not self.renderers or self.max_score == 0:
            return

        threshold = self.max_score * score_ratio
        self.logger.debug(f"Threshold: {threshold=} {self.max_score=} {score_ratio=}")
        to_close = []

        # 閉じるウィンドウをリストアップ（辞書のコピーを作成してからイテレート）
        for id, renderer in list(self.renderers.items()):
            # 品質が計算されていて（score > 0）、かつ閾値以下
            if renderer.score > 0 and renderer.score < threshold:
                to_close.append(id)

        # ウィンドウを閉じてPathを削除
        for id in to_close:
            # closeEventで既に削除されている可能性があるのでチェック
            if id not in self.renderers:
                continue

            renderer = self.renderers[id]
            self.logger.info(
                f"Removed renderer {id}: 品質が閾値以下 "
                f"(score={renderer.score:.3f} < threshold={threshold:.3f}, max_score={self.max_score:.3f})"
            )
            if self.window_manager:
                # PyQt6の場合: close()を呼ぶとcloseEventが発火して自動的に削除される
                self.window_manager.close_window(id)
                # closeEventでremove_renderer()が呼ばれるので、ここでは削除しない
            else:
                # OpenCVの場合: 手動でウィンドウを閉じて削除（存在しない場合はエラーを無視）
                try:
                    cv2.destroyWindow(f"{id}")
                except cv2.error:
                    # ウィンドウが存在しない場合は無視
                    pass

                # Pathを削除
                if id in self.renderers:
                    del self.renderers[id]

        if to_close:
            self.logger.debug(
                f"Closed {len(to_close)} low-score windows during processing"
            )

    def close_low_score_windows(self, score_ratio: float = 0.5):
        """
        処理完了後に品質が閾値以下のウィンドウを自動で閉じる（互換性のため）

        【注意】
        - このメソッドは処理完了後に明示的に呼ぶ用
        - 実際には、処理中も随時_check_and_close_low_score_windows()が呼ばれている
        - このメソッドは最終確認として呼ばれる

        Args:
            score_ratio: 最高品質に対する比率（デフォルト: 0.5 = 50%）
        """
        self._check_and_close_low_score_windows(score_ratio)

    def close_all(self):
        """すべてのウィンドウを閉じる"""
        if self.window_manager:
            self.window_manager.close_all()

    def wait_for_windows_close(self):
        """PyQt6ウィンドウが全て閉じられるまで待機（PyQt6使用時のみ）"""
        if self.window_manager:
            self.window_manager.wait_for_close()
