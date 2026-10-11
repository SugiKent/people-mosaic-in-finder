"""Frozen pre-cache renderer outputs protect native rounding and blend ordering."""

import hashlib
import itertools

import av
import numpy as np
import pytest

from people_mosaic.domain import Face
from people_mosaic.overlay import PreparedOverlay, native_plane

# Produced by the renderer before optimization, not by PreparedOverlay. Each
# digest includes all planes for both ranges, SDR/HLG, three matrices and moves.
LEGACY_NATIVE = {
    "yuv420p": "53c303f9c759f928913d8439334b86fb14c93c313215dc52a0d26bbbeaeb45be",
    "yuv420p10le": "a8c2c5744b35ef6e34ae61a9b503877ae4091967b2607852a30a7045b6f1ee40",
    "yuv422p": "fc448827d31ea62bb55a04d45e99349a6bd3fbcab55bdb171ad4647aeb89d585",
    "yuv422p10le": "248d202c0c2faeb01177569407fa3c4e3825c10029481b89841725e6c6de5f96",
    "yuv444p": "e81ab39fb101a7b21e990da3058e283b7b04a9bee7a0c6112663c424651ea948",
    "yuv444p10le": "7a26fef1b1fba77d776e491d8ca13c9d1d2796269feae9c4d4860dde3555b952",
}
BOXES = [
    (-9.2, -8.1, 27.2, 31.4),
    (31.5, 21.25, 21.8, 19.2),
    (38, 27, 26.5, 22.1),
    (154, 115, 31, 28),
    (200, 200, 10, 10),
]


def png():
    rgba = np.random.default_rng(912).integers(0, 256, size=(72, 96, 4), dtype=np.uint8)
    rgba[:7, :, 3] = 0
    rgba[18:54, 24:72, 3] = 255
    return rgba


@pytest.mark.parametrize("format", sorted(LEGACY_NATIVE))
def test_cached_native_matches_legacy_every_sample(format):
    prepared = PreparedOverlay(png())
    digest = hashlib.sha256()
    for full, hlg, matrix, offset in itertools.product((1, 2), (False, True), (1, 5, 9), (0, 2)):
        frame = av.VideoFrame(190, 138, format)
        frame.color_range, frame.colorspace = full, matrix
        rng = np.random.default_rng(731)
        for index in range(3):
            plane = native_plane(frame, index)
            plane[:] = rng.integers(
                0, 1024 if "10le" in format else 256, size=plane.shape, dtype=plane.dtype
            )
        faces = [Face((x + offset, y + offset, w, h)) for x, y, w, h in BOXES]
        prepared.blend_native(frame, faces, hlg)
        for index in range(3):
            digest.update(native_plane(frame, index).tobytes())
    assert digest.hexdigest() == LEGACY_NATIVE[format]
    assert 0 < prepared.cache_bytes <= prepared.max_cache_bytes


def test_cached_placement_matches_legacy_fractional_boxes_clipping_and_odd_dimensions():
    prepared = PreparedOverlay(png())
    digest = hashlib.sha256()
    for offset in (0, 2, 15, -20):
        for x, y, w, h in BOXES:
            region = prepared.placement((x + offset, y + offset, w, h), 191, 139)
            if region is not None:
                x, y, patch = region
                digest.update(np.asarray((x, y, *patch.shape), dtype=np.int64).tobytes())
                digest.update(patch.tobytes())
    assert digest.hexdigest() == "b2be61f2b6e065daa118af09e4417fca64ab2d3feeef8dbf9aa20524d6c72b7d"


def test_cache_stays_bounded_and_zero_alpha_is_exact_after_eviction():
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[:, :, :3] = 255
    cached = PreparedOverlay(rgba, max_cache_bytes=18_000)
    uncached = PreparedOverlay(rgba, max_cache_bytes=0)
    frame = av.VideoFrame(160, 120, "yuv420p10le")
    for index in range(3):
        native_plane(frame, index)[:] = 231 + index * 127
    before = [native_plane(frame, i).copy() for i in range(3)]
    for width in range(8, 30):
        faces = [Face((40, 30, width, width))]
        cached.blend_native(frame, faces, True)
        uncached.blend_native(frame, faces, True)
        assert cached.cache_bytes <= cached.max_cache_bytes
        assert uncached.cache_bytes == 0
        assert all(np.array_equal(native_plane(frame, i), before[i]) for i in range(3))
    assert cached._cache


def test_source_snapshot_and_position_reuse_do_not_share_pixel_storage():
    rgba = png()
    prepared = PreparedOverlay(rgba)
    first = prepared.placement((40, 40, 20, 20), 200, 200)[2]
    rgba[:] = 0
    moved = prepared.placement((42, 42, 20, 20), 200, 200)[2]
    assert moved is first
    assert not np.shares_memory(first, rgba)
    assert not first.flags.writeable
