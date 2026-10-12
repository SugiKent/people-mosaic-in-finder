"""HLG graphics with explicit reference/peak white; UI-only tone mapping."""

import av
import numpy as np

SRGB_TO_2020 = np.float32(
    [[0.627404, 0.329283, 0.043313], [0.069097, 0.919540, 0.011362], [0.016391, 0.088013, 0.895595]]
)
TO_SRGB = np.linalg.inv(SRGB_TO_2020)
A, B, C = 0.17883277, 0.28466892, 0.55991073
HLG_WHITE = (np.exp((0.75 - C) / A) + B) / 12


def hlg_graphics(rgb: np.ndarray, *, peak_white: bool = False) -> np.ndarray:
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    # Delivery overlays keep PNG white at the signal's maximum white. Archival
    # composition retains the original 75% HLG reference-white convention.
    linear = np.clip(linear @ SRGB_TO_2020.T, 0, 1) * (1 if peak_white else HLG_WHITE)
    return np.where(
        linear <= 1 / 12, np.sqrt(3 * linear), A * np.log(np.maximum(12 * linear - B, 1e-8)) + C
    )


def display_bgr(frame: av.VideoFrame, hlg: bool = False) -> np.ndarray:
    bgr = frame.to_ndarray(format="bgr24")
    if not hlg:
        return bgr
    # HLG -> display linear light on a nominal 1000-nit display, then SDR preview.
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255
    scene = np.where(rgb <= 0.5, rgb**2 / 3, (np.exp((rgb - C) / A) + B) / 12)
    luminance = scene @ np.float32([0.2627, 0.6780, 0.0593])
    light = scene * np.maximum(luminance[:, :, None], 1e-8) ** 0.2 * 10
    linear709 = light @ TO_SRGB.T
    y = np.maximum(linear709 @ np.float32([0.2126, 0.7152, 0.0722]), 0)
    linear709 /= 1 + y[:, :, None]
    linear709 = np.clip(linear709, 0, 1)
    srgb = np.where(
        linear709 <= 0.0031308, linear709 * 12.92, 1.055 * linear709 ** (1 / 2.4) - 0.055
    )
    return np.ascontiguousarray(np.rint(srgb[:, :, ::-1] * 255).astype(np.uint8))
