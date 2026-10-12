from fractions import Fraction
from threading import Event

import av
import numpy as np
import pytest

from people_mosaic.color import hlg_graphics
from people_mosaic.domain import Face
from people_mosaic.engine import FrameCompositor
from people_mosaic.overlay import native_plane


def test_hlg_delivery_white_black_and_neutral_ramp():
    ramp = np.repeat(np.linspace(0, 1, 256, dtype=np.float32)[:, None], 3, axis=1)
    converted = hlg_graphics(ramp, peak_white=True)
    assert np.allclose(converted[0], 0)
    assert np.allclose(converted[-1], 1, atol=1e-6)
    assert np.all(np.diff(converted, axis=0) >= 0)
    assert np.max(np.ptp(converted, axis=1)) < 1e-5
    assert np.allclose(hlg_graphics(ramp)[-1], 0.75, atol=1e-5)


@pytest.mark.parametrize("full_range", [False, True])
@pytest.mark.parametrize("rotation", [0, 90])
@pytest.mark.parametrize("hlg,mp4", [(False, False), (False, True), (True, False), (True, True)])
def test_compositor_delivery_white_transparency_and_colour_metadata(full_range, rotation, hlg, mp4):
    bits = 10 if hlg else 8
    frame = av.VideoFrame(160, 120, "yuv420p10le" if hlg else "yuv420p")
    frame.pts, frame.time_base = 0, Fraction(1, 30)
    frame.color_range, frame.colorspace = (2 if full_range else 1), (9 if hlg else 1)
    factor = 2 ** (bits - 8)
    base = 40 * factor
    for index in range(3):
        native_plane(frame, index)[:] = base if index == 0 else 128 * factor
    rgba = np.full((64, 64, 4), 255, np.uint8)
    rgba[:16, :, 3] = 0
    rgba[16:24, :, 3] = 128
    original_rgba = rgba.copy()
    pix_fmt = ("p010le" if hlg else "yuv420p") if mp4 else None
    compositor = FrameCompositor(rgba, rotation, hlg, pix_fmt, Event())
    result = compositor(frame, [Face((60, 40, 20, 20))])
    assert result.color_range == (2 if full_range else 1)
    assert result.colorspace == (9 if hlg else 1)
    planar = result.reformat(format="yuv420p10le" if hlg else "yuv420p")
    white = 0.75 if hlg and not mp4 else 1
    maximum = (1 << bits) - 1
    expected = round(white * maximum if full_range else (16 + 219 * white) * factor)
    y = native_plane(planar, 0)
    assert abs(int(y[50, 70]) - expected) <= 1
    # The patch begins at (48, 28); clear and half-alpha bands are preserved.
    assert np.all(y[:8, :8] == base)
    assert y[30, 70] == base
    assert abs(int(y[41, 70]) - round((base * 127 + expected * 128) / 255)) <= 1
    for index in (1, 2):
        assert abs(int(native_plane(planar, index)[25, 35]) - 128 * factor) <= 1
    assert np.array_equal(rgba, original_rgba)
