from dataclasses import dataclass
from collections import deque
import cv2
import numpy as np

from trainscanner2.image import standardize


@dataclass
class PathItem:
    frame_index: int
    xy: tuple[float, float]
    value: list
    hop: int = 1


class MovingAverageFIFO:
    """
    FIFO（First-In-First-Out）キューで移動平均を効率的に計算するクラス。
    collections.dequeを内部で使用し、差分更新により移動平均をO(1)で計算する。
    """

    def __init__(self, maxlen: int):
        """
        Args:
            maxlen: キューに保持できる最大要素数
        """
        self._queue = deque(maxlen=maxlen)
        self._sum = None  # 合計値（差分更新用）
        self.maxlen = maxlen

    def append(self, item):
        """
        アイテムをキューに追加する。
        maxlenを超える場合、自動的に先頭要素が削除され、移動平均も更新される。

        Args:
            item: 追加するアイテム（numpy配列など、数値演算可能な型）
        """
        # 満杯の場合、削除される古い値を取得
        if len(self._queue) >= self.maxlen:
            old_item = self._queue[0]
            # 古い値を合計から減算
            if self._sum is not None:
                self._sum -= old_item.astype(np.float32)

        # 新しい値を追加
        self._queue.append(item)

        # 合計を更新
        item_float = item.astype(np.float32)
        if self._sum is None:
            self._sum = item_float.copy()
        else:
            self._sum += item_float

    @property
    def mean(self):
        """
        現在の移動平均を返す。

        Returns:
            移動平均（numpy配列など、元のアイテムと同じ形状）
        """
        if len(self._queue) == 0:
            raise ValueError("Queue is empty")
        if self._sum is None:
            raise ValueError("Sum is not initialized")
        return self._sum / len(self._queue)

    @property
    def count(self):
        """
        現在の要素数を返す。

        Returns:
            要素数
        """
        return len(self._queue)

    def __len__(self):
        """len()で要素数を取得できるようにする"""
        return len(self._queue)

    def __getitem__(self, index):
        """インデックスアクセスを可能にする"""
        return self._queue[index]

    def __iter__(self):
        """イテレータとして使用できるようにする"""
        return iter(self._queue)


def std_hdr(image):
    # 既にグレースケールの場合は変換をスキップ
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Invalid image shape: {image.shape}")
    return standardize(np.log(gray.astype(np.float32) + 1))
