# 縦に長い画像の集合体で巨大画像を表現するClass
from pyperbox import Rect
import numpy as np
from tiledimage.simpleimage import SimpleImage
import cv2
import tempfile
import os
from logging import getLogger


class ImageComb:
    logger = getLogger(__name__)

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

            self.images.append(new_buffer.get_image()[:, :displacement])
            return

            # self.right_endの上に、imageを重ねる。この時、画像の高さも変わる可能性があることに注意。

    def get_image(self, start=0, width=None):
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
            return self.buffer.image if self.buffer.image is not None else None

        if len(self.images) <= start:
            return None

        # imagesを横に連結
        if self.images:
            # すべての画像（images + buffer）を集める
            all_images = list(self.images)[start:]
            total_width = sum(img.shape[1] for img in all_images)
            if (
                self.buffer is not None
                and self.buffer.image is not None
                and width is not None
                and total_width < width
            ):
                all_images.append(self.buffer.image)
                total_width += self.buffer.image.shape[1]

            if not all_images:
                return None
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
                            img,
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

    def save_to_file(self, filename):
        """
        画像をファイルに保存する（メモリ効率的な方法）

        【メモリ効率化の工夫】
        - numpy.memmapを使って仮想メモリ上で画像を構築
        - 短冊ごとに書き込むため、巨大画像全体をメモリに読み込む必要がない
        - ディスクスワップを使うため、物理メモリが足りなくても動作する

        【制限事項】
        - 一時ファイルとして.datファイルを作成（処理後に削除）
        - ディスク容量は必要（画像サイズの3倍程度）

        Args:
            filename: 保存先ファイル名（.jpg, .png, .tiffなど）
        """
        if not self.images:
            return

        # 全体のサイズを計算
        all_images = list(self.images)
        if self.buffer is not None and self.buffer.image is not None:
            all_images.append(self.buffer.image)

        if not all_images:
            return

        total_width = sum(img.shape[1] for img in all_images)
        max_height = max(img.shape[0] for img in all_images)

        # カラーかグレースケールかを判定
        channels = 3 if len(all_images[0].shape) == 3 else 1
        shape = (
            (max_height, total_width, channels)
            if channels == 3
            else (max_height, total_width)
        )

        # 一時ファイルを作成（メモリマップド配列用）
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
        temp_filename = temp_file.name
        temp_file.close()

        try:
            # メモリマップド配列を作成（仮想メモリを使用）
            combined = np.memmap(temp_filename, dtype="uint8", mode="w+", shape=shape)

            # 短冊ごとにコピー（メモリ効率的）
            x_offset = 0
            for img in all_images:
                h, w = img.shape[:2]
                # 高さを中央に配置
                y_offset = (max_height - h) // 2

                if channels == 3:
                    combined[y_offset : y_offset + h, x_offset : x_offset + w, :] = img
                else:
                    combined[y_offset : y_offset + h, x_offset : x_offset + w] = img

                x_offset += w

                # メモリマップをフラッシュ（ディスクに書き込む）
                combined.flush()

            # 最終的な保存（この時点でメモリが必要になるが、OSのキャッシュを活用）
            # cv2.imwriteはnumpy配列を受け取るため、memmapも使える
            self.logger.info(f"Writing image to {filename}...")
            success = cv2.imwrite(filename, combined)

            # メモリマップを閉じる
            del combined

            if success:
                self.logger.info(f"Successfully saved image: {filename}")
            else:
                self.logger.error(f"Failed to save image: {filename}")

        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
