import cv2
import base64
import numpy as np
from IPython.display import display, HTML
import ipywidgets as widgets
from logging import getLogger


class JupyterManager:
    """
    標準的な ipywidgets 版 JupyterManager (安定版への差し戻し)
    """

    logger = getLogger(__name__)

    def __init__(self, video_base=None):
        self.video_base = video_base
        self.panels_created = {}  # path_id -> widget
        self.panel_scores = {}  # path_id -> score

        self.status_label = widgets.Label(value="Status: Ready")
        self.preview_widget = widgets.Image(
            format="jpeg", layout=widgets.Layout(width="200px")
        )

        self.panel_vbox = widgets.VBox([], layout=widgets.Layout(width="100%"))

        self.main_display = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.Label(
                            value=(
                                f"Video: {video_base}" if video_base else "TrainScanner"
                            )
                        ),
                        self.status_label,
                        self.preview_widget,
                    ]
                ),
                self.panel_vbox,
            ]
        )

        display(self.main_display)

    def add_path(self, path_id, renderer_one):
        if path_id not in self.panels_created:
            score = getattr(renderer_one, "score", 0.0)
            self.panel_scores[path_id] = score

            # 標準的なHTML表示
            panel_html = f"<b>ID: {path_id}</b> | Score: <span id='score-{path_id}'>{score:.3f}</span>"
            widget = widgets.VBox(
                [
                    widgets.HTML(value=panel_html),
                    widgets.Image(format="png", layout=widgets.Layout(width="100%")),
                ],
                layout=widgets.Layout(
                    border="1px solid #ccc", margin="5px", padding="5px"
                ),
            )

            self.panels_created[path_id] = widget
            # 型エラー回避のため文字列で指定
            widget.layout.order = "1000"
            self.panel_vbox.children = list(self.panel_vbox.children) + [widget]

    def append_path_strip(self, path_id, cv_strip, x, y=0, score=None):
        # 差し戻し版では ImageStrips 側の全体画像表示に依存するか、
        # あるいは簡易的に最新の画像を表示します。
        if path_id in self.panels_created:
            if score is not None:
                self.panel_scores[path_id] = score
                self.panels_created[path_id].children[
                    0
                ].value = f"<b>ID: {path_id}</b> | Score: {score:.3f}"

            # 注意: ここで全画像を再生成すると重いため、本来は renderer_one.canvas.get_image() 等を使用します。
            # ここでは何もしないか、簡易表示に留めます。
            pass

    def update_path_image(self, path_id, cv_img, score=None):
        if path_id in self.panels_created:
            if cv_img is not None:
                _, buffer = cv2.imencode(".png", cv_img)
                self.panels_created[path_id].children[1].value = buffer.tobytes()
            if score is not None:
                self.panel_scores[path_id] = score
                self.panels_created[path_id].children[
                    0
                ].value = f"<b>ID: {path_id}</b> | Score: {score:.3f}"
            self._sort_panels()

    def _sort_panels(self):
        sorted_ids = sorted(
            self.panel_scores.keys(), key=lambda x: self.panel_scores[x], reverse=True
        )
        for i, pid in enumerate(sorted_ids):
            if pid in self.panels_created:
                self.panels_created[pid].layout.order = str(i)

    def update_preview(self, cv_img):
        if cv_img is not None:
            _, buffer = cv2.imencode(".jpg", cv_img)
            self.preview_widget.value = buffer.tobytes()

    def set_active_paths(self, active_path_ids):
        pass

    def mark_path_inactive(self, path_id):
        pass

    def clear_all_paths(self):
        self.panel_vbox.children = []
        self.panels_created.clear()
        self.panel_scores.clear()

    def set_video_base(self, video_base):
        self.video_base = video_base
