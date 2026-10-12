"""Blend only touched native YUV samples, avoiding a whole-frame RGB round trip."""

import math
from collections import OrderedDict
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .color import hlg_graphics
from .domain import Face
from .errors import UserError

SUPPORTED = {f"yuv{s}p{bits}" for s in ("420", "422", "444") for bits in ("", "10le")}


def cloud() -> np.ndarray:
    image = Image.new("L", (512, 512))
    draw = ImageDraw.Draw(image)
    # An opaque centre hides facial details; only the perimeter is feathered.
    for x, y, radius in [
        (145, 177, 93),
        (281, 137, 104),
        (371, 225, 95),
        (358, 343, 100),
        (228, 379, 100),
        (123, 307, 98),
    ]:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    draw.rectangle((122, 122, 390, 390), fill=255)
    image = image.filter(ImageFilter.GaussianBlur(14))
    # Restore the exact opaque central face area after feathering.
    ImageDraw.Draw(image).rectangle((128, 128, 384, 384), fill=255)
    rgba = np.empty((512, 512, 4), dtype=np.uint8)
    rgba[:, :, :3] = (250, 251, 255)
    rgba[:, :, 3] = np.asarray(image)
    return rgba


def load_png(path: Path) -> np.ndarray:
    if path.stat().st_size > 20 * 1024 * 1024:
        raise UserError("PNG は 20 MB 以下にしてください。")
    with Image.open(path) as image:
        if image.format != "PNG" or max(image.size) > 4096 or min(image.size) < 32:
            raise UserError("32〜4096 ピクセルの PNG を選んでください。")
        rgba = np.asarray(image.convert("RGBA")).copy()
    h, w = rgba.shape[:2]
    centre = rgba[h // 4 : math.ceil(3 * h / 4), w // 4 : math.ceil(3 * w / 4), 3]
    if centre.min() < 128:
        raise UserError("PNG 中央の縦横 50% を、不透明または十分な濃さの画像で覆ってください。")
    # Hide the source face completely, retaining the uploaded perimeter alpha.
    centre[:] = 255
    return rgba


def placement(
    box: tuple, width: int, height: int, rgba: np.ndarray, scale: float = 2.2
) -> tuple[int, int, np.ndarray] | None:
    return PreparedOverlay(rgba, max_cache_bytes=0).placement(box, width, height, scale)


def _geometry(box: tuple, width: int, height: int, shape: tuple, scale: float):
    x, y, w, h = box
    aspect = shape[1] / shape[0]
    target_w = max(w * scale, h * scale * aspect)
    target_h = target_w / aspect
    # Align to all supported chroma grids; the opaque core still covers the face.
    left = math.floor((x + w / 2 - target_w / 2) / 2) * 2
    top = math.floor((y + h / 2 - target_h / 2) / 2) * 2
    right = math.ceil((x + w / 2 + target_w / 2) / 2) * 2
    bottom = math.ceil((y + h / 2 + target_h / 2) / 2) * 2
    if right <= 0 or bottom <= 0 or left >= width or top >= height:
        return None
    x1, y1, x2, y2 = max(0, left), max(0, top), min(width, right), min(height, bottom)
    return x1, y1, (right - left, bottom - top, left - x1, top - y1, x2 - x1, y2 - y1)


def blend_bgr(
    bgr: np.ndarray,
    faces: list[Face],
    rgba: np.ndarray,
    *,
    prepared: "PreparedOverlay | None" = None,
) -> np.ndarray:
    output = bgr.copy()
    if prepared is None:
        prepared = PreparedOverlay(rgba)
    for face in faces:
        region = prepared.placement(face.box, bgr.shape[1], bgr.shape[0])
        if region is None:
            continue
        x, y, patch = region
        h, w = patch.shape[:2]
        alpha = patch[:, :, 3:4].astype(np.float32) / 255
        original = output[y : y + h, x : x + w]
        original[:] = np.rint(original * (1 - alpha) + patch[:, :, 2::-1] * alpha).astype(np.uint8)
    return output


def native_plane(frame: av.VideoFrame, index: int) -> np.ndarray:
    plane = frame.planes[index]
    dtype = np.dtype("<u2") if "10le" in frame.format.name else np.dtype("u1")
    return np.ndarray(
        (plane.height, plane.width),
        dtype=dtype,
        buffer=plane,
        strides=(plane.line_size, dtype.itemsize),
    )


class PreparedOverlay:
    """Per-export (or per-worker) overlay with a byte-bounded, exact-geometry LRU.

    Colour conversion stays after interpolation and uint8 rounding, matching the
    original renderer, including HLG. Position changes can reuse coefficients
    when size and clipping are identical. Instances must not cross worker threads.
    """

    def __init__(self, rgba: np.ndarray, max_cache_bytes: int = 64 * 1024 * 1024):
        self._source = rgba.astype(np.float32)
        self._source[:, :, :3] *= self._source[:, :, 3:4] / 255
        self._source.flags.writeable = False
        self.max_cache_bytes = max(0, max_cache_bytes)
        self.cache_bytes = 0
        self._cache = OrderedDict()

    def _remember(self, key: tuple, arrays: tuple[np.ndarray, ...]):
        size = sum(array.nbytes for array in arrays)
        if size <= self.max_cache_bytes:
            while self._cache and self.cache_bytes + size > self.max_cache_bytes:
                _, (_, evicted_size) = self._cache.popitem(last=False)
                self.cache_bytes -= evicted_size
            for array in arrays:
                array.flags.writeable = False
            self._cache[key] = arrays, size
            self.cache_bytes += size
        return arrays

    def _cached(self, key: tuple):
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
            return value[0]
        return None

    def _patch(self, geometry: tuple) -> np.ndarray:
        key = ("patch", *geometry)
        cached = self._cached(key)
        if cached is not None:
            return cached[0]
        width, height, dx, dy, clipped_width, clipped_height = geometry
        transform = np.float32(
            [[width / self._source.shape[1], 0, dx], [0, height / self._source.shape[0], dy]]
        )
        patch = cv2.warpAffine(
            self._source, transform, (clipped_width, clipped_height), flags=cv2.INTER_LINEAR
        )
        alpha = patch[:, :, 3:4]
        patch[:, :, :3] = np.divide(
            patch[:, :, :3] * 255,
            alpha,
            out=np.zeros_like(patch[:, :, :3]),
            where=alpha > 0,
        )
        patch = np.clip(np.rint(patch), 0, 255).astype(np.uint8)
        return self._remember(key, (patch,))[0]

    def placement(self, box: tuple, width: int, height: int, scale: float = 2.2):
        region = _geometry(box, width, height, self._source.shape, scale)
        if region is None:
            return None
        x, y, geometry = region
        return x, y, self._patch(geometry)

    def blend_native(self, frame: av.VideoFrame, faces: list[Face], hlg: bool = False) -> None:
        if frame.format.name not in SUPPORTED:
            raise UserError(
                "このピクセル形式は未対応です。SDR の YUV 420/422/444（8/10 bit）を使用してください。"
            )
        bits = 10 if "10le" in frame.format.name else 8
        maximum = (1 << bits) - 1
        planes = [native_plane(frame, index) for index in range(3)]
        colour_key = (frame.format.name, int(frame.color_range), int(frame.colorspace), hlg)
        for face in faces:
            region = _geometry(face.box, frame.width, frame.height, self._source.shape, 2.2)
            if region is None:
                continue
            x, y, geometry = region
            positions = []
            sizes = []
            for plane in planes:
                sx, sy = frame.width / plane.shape[1], frame.height / plane.shape[0]
                px, py = round(x / sx), round(y / sy)
                pw, ph = math.ceil(geometry[4] / sx), math.ceil(geometry[5] / sy)
                positions.append((px, py))
                sizes.append((min(pw, plane.shape[1] - px), min(ph, plane.shape[0] - py)))
            key = ("native", geometry, colour_key, tuple(sizes))
            coefficients = self._cached(key)
            if coefficients is None:
                patch = self._patch(geometry)
                coefficients = self._prepare_native(patch, sizes, bits, colour_key)
                coefficients = self._remember(key, coefficients)
            for index, plane in enumerate(planes):
                px, py = positions[index]
                width, height = sizes[index]
                alpha, inverse_alpha, premultiplied = coefficients[index * 3 : index * 3 + 3]
                target = plane[py : py + height, px : px + width]
                blended = np.clip(np.rint(target * inverse_alpha + premultiplied), 0, maximum)
                # Exact bit preservation wherever alpha is zero.
                np.copyto(target, blended.astype(target.dtype), where=alpha > 0)

    @staticmethod
    def _prepare_native(patch, sizes, bits, colour_key):
        _, color_range, colorspace, hlg = colour_key
        factor = 2 ** (bits - 8)
        full_range = color_range == 2
        kr, kb = (0.2126, 0.0722) if colorspace == 1 else (0.299, 0.114)
        if colorspace == 9:
            kr, kb = 0.2627, 0.0593
        rgb = patch[:, :, :3].astype(np.float32) / 255
        if hlg:
            rgb = hlg_graphics(rgb)
        alpha = patch[:, :, 3].astype(np.float32) / 255
        luma = rgb[:, :, 0] * kr + rgb[:, :, 1] * (1 - kr - kb) + rgb[:, :, 2] * kb
        cb = (rgb[:, :, 2] - luma) / (2 * (1 - kb))
        cr = (rgb[:, :, 0] - luma) / (2 * (1 - kr))
        maximum = (1 << bits) - 1
        channels = (
            [luma * maximum, (cb + 0.5) * maximum, (cr + 0.5) * maximum]
            if full_range
            else [(16 + 219 * luma) * factor, (128 + 224 * cb) * factor, (128 + 224 * cr) * factor]
        )
        coefficients = []
        for channel, size in zip(channels, sizes, strict=True):
            a = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
            premultiplied = cv2.resize(channel * alpha, size, interpolation=cv2.INTER_AREA)
            coefficients.extend((a, 1 - a, premultiplied))
        return tuple(coefficients)


def blend_native(
    frame: av.VideoFrame, faces: list[Face], rgba: np.ndarray, hlg: bool = False
) -> None:
    PreparedOverlay(rgba).blend_native(frame, faces, hlg)
