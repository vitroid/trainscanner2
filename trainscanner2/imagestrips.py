import numpy as np
import cv2
import tempfile
import os
import shutil
from logging import getLogger
from trainscanner2.image import ImageRect


class ImageStrips:
    """縦に長い画像の集合体で巨大画像を表現するクラス。
    最新部分は buffer に保持し、確定した左側の部分は strips (images) として保存する。
    """

    logger = getLogger(__name__)

    def __init__(self, cache=False):
        self.buffer = ImageRect()
        self.cache_dir = tempfile.mkdtemp() if cache else None
        self.shapes = []
        self.images = []  # cache=True時はファイルパス、False時はnp.ndarrayが入る

    def __del__(self):
        if self.cache_dir:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _get_strip(self, index):
        """指定したインデックスの画像を読み込む（キャッシュ対応）"""
        img = self.images[index]
        if isinstance(img, str):
            return cv2.imread(img)
        return img

    def put_image(self, lefttop: tuple[int, int], image: np.ndarray):
        self.put_imagerect(ImageRect(lefttop=lefttop, image=image))

    def put_imagerect(self, imagerect: ImageRect):
        """画像を追加し、確定した左側を切り出して保存する。"""
        center = imagerect.left + imagerect.width // 2

        if self.buffer.image is None:
            # 初回追加
            left, right = imagerect.split_vertically(imagerect.width // 2)
            self.buffer.put_imagerect(right)
            self._add_strip(left)
            return

        if self.buffer.left < center:
            # 右方向に進んだ場合、重なる部分を合成して左端を切り出す
            displacement = center - self.buffer.left
            alpha = np.concatenate(
                [np.linspace(0, 1.0, displacement), np.ones(imagerect.width // 2)]
            )

            elim = imagerect.width - len(alpha)
            if elim < 0:
                return

            _, overlay = imagerect.split_vertically(elim)

            # bufferを拡張して合成
            new_buffer = ImageRect()
            new_buffer.put_imagerect(self.buffer)
            new_buffer.put_image(
                lefttop=(overlay.left, overlay.top),
                image=overlay.image,
                linear_alpha=alpha,
            )

            # 左側の確定部分を切り出し
            strip, self.buffer = new_buffer.split_vertically(displacement)
            self._add_strip(strip)

    def _add_strip(self, strip: ImageRect):
        """確定した短冊を保存する"""
        self.shapes.append(strip.shape)
        if self.cache_dir:
            path = os.path.join(self.cache_dir, f"{len(self.images):05d}.png")
            cv2.imwrite(path, strip.image)
            self.images.append(path)
        else:
            self.images.append(strip.image)

    def _get_padded_images(self, start=0, width=None):
        """表示や保存のために、高さを揃えた画像のリストを生成するイテレータ"""
        if not self.images and (self.buffer is None or self.buffer.image is None):
            return

        # 必要な範囲の画像
        target_indices = (
            range(start, len(self.images)) if start < len(self.images) else []
        )

        # 最大の高さを計算
        relevant_shapes = self.shapes[start:]
        max_h = max(s[0] for s in relevant_shapes) if relevant_shapes else 0
        if self.buffer.image is not None:
            max_h = max(max_h, self.buffer.image.shape[0])

        accum_w = 0
        for idx in target_indices:
            img = self._get_strip(idx)
            accum_w += img.shape[1]
            yield self._pad_image(img, max_h)
            if width and accum_w >= width:
                return

        if self.buffer.image is not None:
            yield self._pad_image(self.buffer.image, max_h)

    def _pad_image(self, img, target_h):
        """画像を中央寄せでパディングする"""
        h, w = img.shape[:2]
        if h >= target_h:
            return img

        top = (target_h - h) // 2
        bottom = target_h - h - top
        padding = (
            ((top, bottom), (0, 0), (0, 0))
            if img.ndim == 3
            else ((top, bottom), (0, 0))
        )
        return np.pad(img, padding, mode="constant")

    def get_image(self, start=0, width=None):
        """現在の画像を1枚に結合して返す（表示用）"""
        imgs = list(self._get_padded_images(start, width))
        return np.hstack(imgs) if imgs else None

    def save_to_file(self, filename):
        """巨大な画像をメモリ効率を考慮して保存する"""
        if not self.images and (self.buffer is None or self.buffer.image is None):
            return

        total_w = sum(s[1] for s in self.shapes) + (
            self.buffer.image.shape[1] if self.buffer.image is not None else 0
        )
        max_h = max(
            [s[0] for s in self.shapes]
            + ([self.buffer.image.shape[0]] if self.buffer.image is not None else [0])
        )
        channels = (
            3
            if len(self.shapes[0]) == 3
            or (self.buffer.image is not None and self.buffer.image.ndim == 3)
            else 1
        )

        shape = (max_h, total_w, 3) if channels == 3 else (max_h, total_w)

        # memmapを使用して巨大画像を作成
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mmap = np.memmap(tmp_path, dtype="uint8", mode="w+", shape=shape)
            x = 0
            for i, img in enumerate(self._get_padded_images()):
                h, w = img.shape[:2]
                if channels == 3:
                    mmap[:, x : x + w, :] = img
                else:
                    mmap[:, x : x + w] = img
                x += w
                if i % 20 == 0:
                    mmap.flush()

            self.logger.info(f"Writing to {filename}...")
            cv2.imwrite(filename, mmap)
            del mmap
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
