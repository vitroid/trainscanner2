# 縦に長い画像の集合体で巨大画像を表現するClass
from re import S
import numpy as np
import cv2
import tempfile
import os
from logging import getLogger
import shutil

from trainscanner.image import ImageRect


class ImageStrips:
    logger = getLogger(__name__)

    def __init__(self, cache=False):
        # 辞書のキーはtrain_position
        self.buffer = ImageRect()
        if cache:
            self.cache_dir = tempfile.mkdtemp()
        else:
            self.cache_dir = None
        self.shapes = []
        self.images = []

    def __del__(self):
        if self.cache_dir is not None:
            shutil.rmtree(self.cache_dir)

    def put_image(self, lefttop: tuple[int, int], image: np.ndarray):
        self.put_imagerect(ImageRect(lefttop=lefttop, image=image))

    def put_imagerect(
        self,
        imagerect: ImageRect,
    ):
        """
        画像を追加する

        imagerectの右半分を、self.bufferに重ねる。
        常に右にずれた場所に画像を重ねるので、imagerectの中央よりも左側の領域はこのあと変更される可能性がないのでメモリーからpurgeする。
        右部分はself.bufferに残す。
        """
        center = imagerect.left + imagerect.width // 2
        if self.buffer.image is None:
            lefthalf, righthalf = imagerect.split_vertically(imagerect.width // 2)
            self.buffer.put_imagerect(righthalf)
            if self.cache_dir is not None:
                filename = self.cache_dir + f"/{len(self.images):05d}.png"
                cv2.imwrite(filename, lefthalf.image)
                self.images.append(filename)
            else:
                self.images.append(lefthalf.image)
            self.shapes.append(lefthalf.shape)
            return
        if self.buffer.left < center:
            # 移動があった。
            displacement = center - self.buffer.left
            alpha = np.concatenate(
                (np.linspace(0, 1.0, displacement), np.ones(imagerect.width // 2))
            )
            elim = imagerect.width - alpha.shape[0]
            _, overlay = imagerect.split_vertically(elim)

            new_buffer = ImageRect()
            new_buffer.put_imagerect(self.buffer)
            new_buffer.put_image(
                lefttop=(overlay.left, overlay.top),
                image=overlay.image,
                linear_alpha=alpha,
            )
            # いまはスムーズにつながなくていい
            strip, self.buffer = new_buffer.split_vertically(displacement)
            if self.cache_dir is not None:
                filename = self.cache_dir + f"/{len(self.images):05d}.png"
                cv2.imwrite(filename, strip.image)
                self.images.append(filename)
            else:
                self.images.append(strip.image)
            self.shapes.append(strip.shape)
            return

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
        # self.logger.debug(f"get_image(start={start}, width={width})")
        if len(self.images) == 0 and self.buffer is None:
            return None

        # bufferだけの場合
        if len(self.images) == 0 and self.buffer is not None:
            return self.buffer.image if self.buffer.image is not None else None

        if len(self.images) <= start:
            return None

        # imagesを横に連結
        if len(self.images) > 0:
            # すべての画像（images + buffer）を集める
            # all_images = [self.images[i] for i in range(start, len(self.images))]
            total_width = sum([x[1] for x in self.shapes[start:]])
            if (
                self.buffer is not None
                and self.buffer.image is not None
                and width is not None
                and total_width < width
            ):
                # all_images.append(self.buffer.image)
                total_width += self.buffer.image.shape[1]

            # if not all_images:
            #     return None
            # 最大の高さを取得
            max_height = max([x[0] for x in self.shapes[start:]])

            # 高さを揃える（上下中央にパディング）
            padded_images = []
            accum = 0
            for img in self.images[start:]:
                if self.cache_dir is not None:
                    img = cv2.imread(img)
                h, w = img.shape[:2]
                accum += w
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
                if width is not None and accum >= width:
                    break
            else:
                padded_images.append(self.buffer.image)

            if len(padded_images) > 0:
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
            filename: 保存先ファイル名（.png, .tiffなど）
        """
        if not self.images:
            return

        # 全体のサイズを計算
        # ここで全画像を読みこむのでメモリーがたくさん必要になる。
        # all_images = list(self.images.values())
        # if self.buffer is not None and self.buffer.image is not None:
        #     all_images.append(self.buffer.image)

        # if not all_images:
        #     return

        total_width = sum([x[1] for x in self.shapes])
        max_height = max([x[0] for x in self.shapes])
        if self.buffer is not None and self.buffer.image is not None:
            total_width += self.buffer.image.shape[1]
            max_height = max(max_height, self.buffer.image.shape[0])

        # カラーかグレースケールかを判定
        channels = 3 if len(self.shapes[0]) == 3 else 1
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
            for i, img in enumerate(self.images):
                if self.cache_dir is not None:
                    img = cv2.imread(img)
                h, w = img.shape[:2]
                # 高さを中央に配置
                y_offset = (max_height - h) // 2

                if channels == 3:
                    combined[y_offset : y_offset + h, x_offset : x_offset + w, :] = img
                else:
                    combined[y_offset : y_offset + h, x_offset : x_offset + w] = img

                x_offset += w

                # メモリマップをフラッシュ（ディスクに書き込む）
                if i % 20 == 0:
                    combined.flush()

            if self.buffer is not None and self.buffer.image is not None:
                h, w = self.buffer.image.shape[:2]
                y_offset = (max_height - h) // 2
                if channels == 3:
                    combined[y_offset : y_offset + h, x_offset : x_offset + w, :] = (
                        self.buffer.image
                    )
                else:
                    combined[y_offset : y_offset + h, x_offset : x_offset + w] = (
                        self.buffer.image
                    )
                x_offset += w

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
