import numpy as np
from dataclasses import dataclass
from tiledimage.simpleimage import SimpleImage
from trainscanner.image import linear_alpha
import cv2
from logging import getLogger


@dataclass
class PathItem:
    xy: tuple[int, int]
    value: list


class FIFO:
    def __init__(self, maxlen: int):
        self.queue = []
        self.maxlen = maxlen

    def append(self, item):
        self.queue.append(item)
        if len(self.queue) > self.maxlen:
            self.queue.pop(0)

    def fluctuation(self):
        return max(self.queue) - min(self.queue)

    @property
    def length(self):
        return len(self.queue)

    @property
    def filled(self):
        return len(self.queue) == self.maxlen


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
    alpha = linear_alpha(img_width=rw, mixing_width=20, slit_pos=0, head_right=True)
    rotated = cv2.warpAffine(frame, R, (rw, rh))
    # cv2.imshow("rotated", rotated)
    # cv2.waitKey(0)
    # 画像中心をそろえる
    if first:
        canvas.put_image(
            (int(train_position) - rw // 2, -rh // 2),
            rotated,
        )
    else:
        canvas.put_image(
            (int(train_position) - rw // 2, -rh // 2),
            rotated,
            linear_alpha=alpha,
        )


class Render_one:
    """
    1つのPathの描画を担当する。
    """

    logger = getLogger(__name__)

    def __init__(self, id: int, num_leading_frames: int):
        self.leading_frames = FIFO(num_leading_frames)
        self.history = []
        self.id = id
        self.canvas = SimpleImage()
        self.first = False
        self.train_position = 0
        self.alive = True

    def _render_one(
        self,
        frame: np.ndarray,
        h: PathItem,
    ):
        delta = h.xy
        frame_index, value = h.value
        self.logger.debug(f"{id=} {frame_index=} {delta=} ")
        dx, dy = delta
        dd = -((dx**2 + dy**2) ** 0.5)
        if dd != 0:
            self.train_position += dd
            cosine = dx / dd
            sine = dy / dd
            rotated_placement(
                self.canvas, frame, sine, cosine, self.train_position, self.first
            )
            self.first = False

    def put(self, frame: np.ndarray, pathitem: PathItem, quality_threshold=0.0):
        if not self.alive:
            return
        if self.leading_frames.filled:
            if 0 < self.quality < quality_threshold:
                # close the window
                cv2.destroyWindow(f"{self.id}")
                self.alive = False
            self._render_one(frame, pathitem)
            img = self.canvas.get_image()
            if img is not None:
                cv2.imshow(f"{self.id}", img)
                cv2.waitKey(1)
        else:
            self.history.append(pathitem)
            self.leading_frames.append(frame)
            if self.leading_frames.filled:
                for f, pi in zip(self.leading_frames.queue, self.history):
                    self._render_one(f, pi)

    @property
    def quality(self):
        # 最初の20フレームで判別する。
        if len(self.history) > 20:
            return np.mean([h.value for h in self.history])
        return 0.0


class Render:
    """
    極値とフレームをうけとり、個別のレンダラーに差配する。
    生きているrendererの最高品質を調査し、低品質rendererの打ち切り指示をする
    """

    def __init__(self):
        self.renderers = {}
        self.max_quality = 0.0

    def put(
        self,
        id: int,
        frame: np.ndarray,
        historyitem: PathItem,
    ):
        if id in self.renderers:
            r = self.renderers[id]
        else:
            r = Render_one(id, num_leading_frames=20)
            self.renderers[id] = r
        r.put(frame, historyitem, quality_threshold=self.max_quality * 0.8)
        q = r.quality
        if self.max_quality < q:
            self.max_quality = q
