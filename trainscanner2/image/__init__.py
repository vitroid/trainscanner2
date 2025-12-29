import cv2
import numpy as np
from pyperbox import Rect, Range
import math
from dataclasses import dataclass
import logging

# trainscanner2/image/__init__.py


def _find_peaks(arr: np.ndarray):
    """
    周囲8点のいずれよりも値が大きい点を極値とし、その位置と値を返す。
    """
    centers = arr[1:-1, 1:-1]
    non_max = np.zeros_like(centers, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                cmp = (
                    arr[
                        1 + dy : centers.shape[0] + 1 + dy,
                        1 + dx : centers.shape[1] + 1 + dx,
                    ]
                    > centers
                )
                non_max |= cmp
    is_max = ~non_max
    return [(x + 1, y + 1) for y, x in np.argwhere(is_max)]


def find_paraboloid_extremum(values):
    """
    3x3の格子点(x,y in [-1, 0, 1])の値から、
    最小二乗法で放物面をフィッティングし、その極値を計算する。
    """
    try:
        z_vec = np.array(values).flatten()
        if z_vec.shape[0] != 9:
            raise ValueError("入力は3x3の配列である必要があります。")
    except Exception as e:
        return {"status": "error", "message": f"入力値エラー: {e}"}

    x = np.array([-1, 0, 1] * 3)
    y = np.repeat([-1, 0, 1], 3)

    M = np.stack([x**2, y**2, x * y, x, y, np.ones(9)], axis=-1)

    try:
        p, _, _, _ = np.linalg.lstsq(M, z_vec, rcond=None)
        a, b, c, d, e, f = p
    except np.linalg.LinAlgError as e:
        return {"status": "error", "message": f"最小二乗法エラー: {e}"}

    Hessian = np.array([[2 * a, c], [c, 2 * b]])
    B = np.array([-d, -e])
    det = np.linalg.det(Hessian)

    if np.isclose(det, 0):
        return {
            "status": "no_unique_extremum",
            "message": "det=0",
        }

    try:
        coords = np.linalg.solve(Hessian, B)
        x0, y0 = coords
    except np.linalg.LinAlgError:
        return {
            "status": "error",
            "message": "座標の連立方程式を解けませんでした。",
        }

    z0 = p @ [x0**2, y0**2, x0 * y0, x0, y0, 1]

    if det > 0:
        ext_type = "極小値 (Local Minimum)" if a > 0 else "極大値 (Local Maximum)"
    else:
        ext_type = "鞍点 (Saddle Point)"

    return {
        "status": "success",
        "x": x0,
        "y": y0,
        "z": z0,
        "type": ext_type,
    }


def find_parabola_extremum(values):
    """
    格子点(x in [-1, 0, 1])の値から、
    放物線をフィッティングし、その極値を計算する。
    """
    z = np.array(values).flatten()
    a = (z[0] + z[2]) / 2 - z[1]
    b = (z[2] - z[0]) / 2
    c = z[1]
    if a == 0:
        return {"status": "error", "message": "a=0"}
    x0 = -b / (2 * a)
    z0 = a * x0**2 + b * x0 + c
    return {
        "status": "success",
        "x": x0,
        "z": z0,
    }


@dataclass
class MatchRect:
    """
    目盛りつきのmatch score行列
    """

    value: np.ndarray
    rect: Rect

    def _peak_subpixel(self):
        if self.value.shape[0] == 1:
            x = np.argmax(self.value[0])
            if 0 < x < self.rect.width - 1:
                values = self.value[0, x - 1 : x + 2]
                result = find_parabola_extremum(values)
                if result["status"] == "success":
                    return self.coord(x + result["x"], 0), result["z"]
            return self.coord(x, 0), self.value[0, x]
        else:
            _, maxval, _, (x, y) = cv2.minMaxLoc(self.value)
            if 0 < x < self.rect.width - 1 and 0 < y < self.rect.height - 1:
                values = self.value[y - 1 : y + 2, x - 1 : x + 2]
                result = find_paraboloid_extremum(values)
                if result["status"] == "success":
                    return self.coord(x + result["x"], y + result["y"]), result["z"]
            return self.coord(x, y), maxval

    def validate(self):
        if (self.rect.height, self.rect.width) != self.value.shape:
            raise ValueError(
                f"value shape {self.value.shape} does not match rect shape {self.rect.height}x{self.rect.width}"
            )

    def peak(self, subpixel=False):
        if subpixel:
            return self._peak_subpixel()
        _, maxval, _, (x, y) = cv2.minMaxLoc(self.value)
        return self.coord(x, y), maxval

    def peaks(self, height: float = 0.5, subpixel=False):
        for x, y in _find_peaks(self.value):
            if self.value[y, x] > height:
                if subpixel:
                    if 0 < x < self.rect.width - 1 and 0 < y < self.rect.height - 1:
                        values = self.value[y - 1 : y + 2, x - 1 : x + 2]
                        result = find_paraboloid_extremum(values)
                        if result["status"] == "success":
                            yield self.coord(x + result["x"], y + result["y"]), result[
                                "z"
                            ]
                        else:
                            yield self.coord(x, y), self.value[y, x]
                    else:
                        yield self.coord(x, y), self.value[y, x]
                else:
                    yield self.coord(x, y), self.value[y, x]

    def coord(self, x: float, y: float):
        return x + self.rect.left, y + self.rect.top

    def plot_image(self):
        """
        スコア行列を表示用の画像に変換する。
        """
        # 値を0-1に正規化（表示用）
        v_min = np.min(self.value)
        v_max = np.max(self.value)
        if v_max > v_min:
            normalized = (self.value - v_min) / (v_max - v_min)
        else:
            normalized = np.zeros_like(self.value)

        display_img = (normalized * 255).astype(np.uint8)
        # 必要に応じて拡大表示
        h, w = display_img.shape[:2]
        if w < 200 or h < 200:
            scale = max(200 // w, 200 // h)
            display_img = cv2.resize(
                display_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )
        return display_img

    def plot(self, label: str = ""):
        """
        スコア行列をデバッグ表示する。
        """
        display_img = self.plot_image()
        cv2.imshow(f"MatchRect: {label}", display_img)
        cv2.waitKey(1)


class ImageRect:
    """絶対座標付きの画像。"""

    def __init__(
        self,
        lefttop: tuple[int, int] = (0, 0),
        image: np.ndarray = None,
        bgcolor=(100, 100, 100),
    ):
        self.left, self.top = lefttop
        self.bgcolor = np.array(bgcolor, dtype=np.uint8)
        self.image = image

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def rect(self):
        if self.image is not None:
            return Rect.from_bounds(
                self.left,
                self.right,
                self.top,
                self.bottom,
            )

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def right(self):
        return self.left + self.image.shape[1]

    @property
    def bottom(self):
        return self.top + self.image.shape[0]

    @property
    def shape(self):
        if self.image is not None:
            return self.image.shape
        return (0, 0, 3)

    def get_region(self, rect: Rect | None = None):
        if rect is None:
            rect = self.rect
        if rect is None:
            return

        dst_width = rect.width
        dst_height = rect.height

        imagerect = ImageRect(
            lefttop=(rect.left, rect.top),
            image=np.zeros((dst_height, dst_width, 3), dtype=np.uint8) + self.bgcolor,
        )
        crop = self.rect & rect
        imagerect.put_image(
            lefttop=(crop.left, crop.top),
            image=self.image[
                crop.top - self.rect.top : crop.bottom - self.rect.top,
                crop.left - self.rect.left : crop.right - self.rect.left,
            ],
        )
        return imagerect

    def put_imagerect(self, imagerect):
        self.put_image(lefttop=(imagerect.left, imagerect.top), image=imagerect.image)

    def put_image(
        self,
        lefttop: tuple[int, int],
        image: np.ndarray,
        linear_alpha=None,
        full_alpha=None,
    ):
        h, w = image.shape[:2]
        rect = Rect.from_bounds(lefttop[0], lefttop[0] + w, lefttop[1], lefttop[1] + h)
        if self.image is None:
            self.left, self.top = lefttop
            self.image = image.copy()
        else:
            newrect = self.rect | rect
            new_image = (
                np.zeros([newrect.height, newrect.width, 3], dtype=np.uint8)
                + self.bgcolor
            )
            new_image[
                self.top - newrect.top : self.bottom - newrect.top,
                self.left - newrect.left : self.right - newrect.left,
            ] = self.image
            self.image = new_image
            self.left = newrect.left
            self.top = newrect.top

            if linear_alpha is None and full_alpha is None:
                self.image[
                    rect.top - self.top : rect.bottom - self.top,
                    rect.left - self.left : rect.right - self.left,
                ] = image
            else:
                if full_alpha is not None:
                    alpha = full_alpha[:, :, np.newaxis]
                else:
                    alpha = linear_alpha[np.newaxis, :, np.newaxis]
                self.image[
                    rect.top - self.top : rect.bottom - self.top,
                    rect.left - self.left : rect.right - self.left,
                    :,
                ] = (
                    alpha * image
                    + (1 - alpha)
                    * self.image[
                        rect.top - self.top : rect.bottom - self.top,
                        rect.left - self.left : rect.right - self.left,
                        :,
                    ]
                )

    def split_vertically(self, left_width: int):
        left = ImageRect(
            lefttop=(self.left, self.top), image=self.image[:, :left_width]
        )
        right = ImageRect(
            lefttop=(self.left + left_width, self.top), image=self.image[:, left_width:]
        )
        return left, right


def match_rect(target: ImageRect, focus: ImageRect) -> MatchRect:
    scores = cv2.matchTemplate(target.image, focus.image, cv2.TM_CCOEFF_NORMED)
    rect = Rect.from_bounds(
        target.left - focus.left,
        target.right - focus.right + 1,
        target.top - focus.top,
        target.bottom - focus.bottom + 1,
    )
    return MatchRect(value=scores, rect=rect)


def match_rect_expanded(target: ImageRect, focus: ImageRect, margin: int) -> MatchRect:
    expanded_target = np.zeros(
        (target.height + 2 * margin, target.width + 2 * margin), dtype=np.float32
    )
    expanded_target[margin:-margin, margin:-margin] = target.image
    scores = cv2.matchTemplate(expanded_target, focus.image, cv2.TM_CCOEFF_NORMED)
    return MatchRect(
        value=scores,
        rect=Rect.from_bounds(
            target.left - focus.left - margin,
            target.right - focus.right + 1 + margin,
            target.top - focus.top - margin,
            target.bottom - focus.bottom + 1 + margin,
        ),
    )


def standardize(frame):
    # 既にグレースケールの場合は変換をスキップ
    if len(frame.shape) == 2:
        gray = frame
    elif len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Invalid image shape: {frame.shape}")
    return ((gray - np.mean(gray)) / (np.std(gray) + 1e-6)).astype(np.float32)


def diffImage(frame1, frame2, dx, dy, mode="stack"):
    affine = np.array(((1.0, 0.0, dx), (0.0, 1.0, dy)), dtype=np.float32)
    h, w = frame1.shape[0:2]
    if mode == "diff":
        std2 = standardize(frame2)
        frame1_warped = cv2.warpAffine(frame1, affine, (w, h))
        std1 = standardize(frame1_warped)
        return (255 * cv2.absdiff(std1, std2)).astype(np.uint8)
    elif mode == "stack":
        flags = np.arange(h) * 16 % h > h // 2
        frame1_warped = cv2.warpAffine(frame1, affine, (w, h))
        frame1_warped[flags] = frame2[flags]
        return frame1_warped
    return frame1
