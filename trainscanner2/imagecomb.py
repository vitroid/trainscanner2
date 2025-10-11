# 縦に長い画像の集合体で巨大画像を表現するClass
from pyperbox import Rect
import numpy as np
from tiledimage.simpleimage import SimpleImage


class ImageComb:
    def __init__(self):
        # 辞書のキーはtrain_position
        self.images = []
        self.buffer = SimpleImage()

    def put_image(
        self,
        rect: Rect,
        image,
    ):
        """
        画像を追加する

        """
        height, width = image.shape[:2]
        lefthalf = image[:, : width // 2]
        righthalf = image[:, width // 2 :]
        rect_righthalf = Rect.from_bounds(
            left=rect.left + rect.width // 2,
            right=rect.right,
            top=rect.top,
            bottom=rect.bottom,
        )
        if self.buffer.rect is None:
            self.buffer.put_image((rect_righthalf.left, rect_righthalf.top), righthalf)
            self.images.append(lefthalf)
            return
        if self.buffer.rect.left < rect_righthalf.left:
            # 移動があった。
            # 移動した部分は保存する。
            new_rect = self.buffer.rect | rect_righthalf
            new_buffer = SimpleImage()
            new_buffer.put_image(
                (self.buffer.rect.left, self.buffer.rect.top), self.buffer.get_image()
            )
            new_buffer.put_image((rect_righthalf.left, rect_righthalf.top), righthalf)
            assert new_buffer.rect.width == new_buffer.image.shape[1]
            assert new_buffer.rect.height == new_buffer.image.shape[0]
            # いまはスムーズにつながなくていい
            displacement = rect_righthalf.left - self.buffer.rect.left
            self.buffer = SimpleImage()
            rect = Rect.from_bounds(
                left=new_rect.left + displacement,
                right=new_rect.right,
                top=new_rect.top,
                bottom=new_rect.bottom,
            )
            self.buffer.put_image(
                (rect.left, rect.top), new_buffer.get_image()[:, displacement:]
            )

            assert rect.width == self.buffer.image.shape[1]
            assert rect.height == self.buffer.image.shape[0]

            self.images.append(self.buffer.get_image()[:, :displacement])
            return

            # self.right_endの上に、imageを重ねる。この時、画像の高さも変わる可能性があることに注意。

    def get_image(self):
        """
        現在の画像を1枚に結合して返す（表示用）

        【データ構造】
        - self.images: 縦長画像のリスト（左から右へ）
        - self.buffer: 最後の部分（まだimagesに追加されていない）

        【高さの処理】
        - 画像の高さが異なる場合、最大の高さに合わせてパディング
        - 上下中央に配置

        Returns:
            np.ndarray: 結合された画像、またはNone（画像がない場合）
        """
        if not self.images and self.buffer is None:
            return None

        # bufferだけの場合
        if not self.images and self.buffer is not None:
            return self.buffer

        # imagesを横に連結
        if self.images:
            # すべての画像（images + buffer）を集める
            all_images = list(self.images)
            if self.buffer is not None:
                all_images.append(self.buffer.image)

            # 最大の高さを取得
            max_height = max(img.shape[0] for img in all_images)

            # 高さを揃える（上下中央にパディング）
            padded_images = []
            for img in all_images:
                h, w = img.shape[:2]
                if h < max_height:
                    # パディングが必要
                    pad_top = (max_height - h) // 2
                    pad_bottom = max_height - h - pad_top
                    if len(img.shape) == 3:  # カラー画像
                        padded = np.pad(
                            img.image,
                            ((pad_top, pad_bottom), (0, 0), (0, 0)),
                            mode="constant",
                        )
                    else:  # グレースケール
                        padded = np.pad(
                            img, ((pad_top, pad_bottom), (0, 0)), mode="constant"
                        )
                    padded_images.append(padded)
                else:
                    padded_images.append(img)

            # 横に連結
            combined = np.hstack(padded_images)
            return combined

        return None
