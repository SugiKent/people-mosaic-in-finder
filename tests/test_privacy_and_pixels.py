from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

from people_mosaic.color import hlg_graphics
from people_mosaic.domain import Face, Person, hidden_faces, identify
from people_mosaic.errors import UserError
from people_mosaic.overlay import SUPPORTED, blend_native, cloud, load_png, native_plane


def test_only_confident_explicit_selection_can_reveal():
    faces = [
        Face((0, 0, 20, 20), 0, True),
        Face((0, 0, 20, 20), 0, False),
        Face((0, 0, 20, 20)),
        Face((0, 0, 20, 20), 1, True),
    ]
    assert hidden_faces(faces, set()) == faces
    assert hidden_faces(faces, {0}) == faces[1:]


def test_ambiguous_identity_does_not_reveal():
    a = np.array([1.0, 0.0], np.float32)
    b = np.array([0.999, 0.0447], np.float32)
    people = [Person(a, np.zeros((1, 1, 3)), 0), Person(b, np.zeros((1, 1, 3)), 0)]
    assert identify(a, people) == (None, False)
    assert identify(np.array([0.65, 0.76]), [people[0]]) == (0, False)


def test_transparent_face_centre_is_rejected(tmp_path):
    path = tmp_path / "unsafe.png"
    Image.new("RGBA", (64, 64), (255, 255, 255, 0)).save(path)
    with pytest.raises(UserError, match="不透明"):
        load_png(path)
    safe = np.full((64, 64, 4), 255, np.uint8)
    safe[:16, :, 3] = 0
    Image.fromarray(safe).save(path)
    assert np.array_equal(load_png(path), safe)


def test_default_cloud_preserves_every_bundled_pixel_and_original_transparency():
    source = Path(__file__).parents[1] / "src/people_mosaic/assets/default-fluffy.png"
    with Image.open(source) as image:
        expected = np.asarray(image.convert("RGBA"))
    rgba = cloud()
    assert rgba.shape == expected.shape
    assert np.array_equal(rgba, expected)
    assert np.any(rgba[:, :, 3] == 0)
    assert np.any((rgba[:, :, 3] > 0) & (rgba[:, :, 3] < 255))
    assert np.array_equal(rgba, cloud())


@pytest.mark.parametrize("format", sorted(SUPPORTED))
def test_native_overlay_preserves_every_untouched_sample(format):
    frame = av.VideoFrame(160, 120, format)
    frame.colorspace = 1
    frame.color_range = 1
    rng = np.random.default_rng(8)
    maximum = 1024 if "10le" in format else 256
    originals = []
    for index in range(3):
        plane = native_plane(frame, index)
        plane[:] = rng.integers(0, maximum, size=plane.shape, dtype=plane.dtype)
        originals.append(plane.copy())
    blend_native(frame, [Face((60, 40, 20, 20))], cloud())
    for index, before in enumerate(originals):
        after = native_plane(frame, index)
        # This far-away area is outside every feathered edge and chroma block.
        assert np.array_equal(after[:10, :10], before[:10, :10])
        assert np.any(after != before)
    # No masks must be an exact no-op, even at 10 bit.
    modified = [native_plane(frame, index).copy() for index in range(3)]
    blend_native(frame, [], cloud())
    assert all(np.array_equal(native_plane(frame, i), modified[i]) for i in range(3))


def test_full_resolution_outside_alpha_is_exact():
    frame = av.VideoFrame(160, 120, "yuv444p")
    for i in range(3):
        native_plane(frame, i)[:] = 23 + i * 50
    before = [native_plane(frame, i).copy() for i in range(3)]
    png = np.zeros((64, 64, 4), np.uint8)
    png[16:48, 16:48] = (255, 255, 255, 255)
    blend_native(frame, [Face((60, 40, 20, 20))], png)
    # Transparent corners within the overlay bounding box are unchanged too.
    for i in range(3):
        assert native_plane(frame, i)[29, 49] == before[i][29, 49]
        assert native_plane(frame, i)[50, 70] != before[i][50, 70]


def test_hlg_graphics_reference_white_and_untouched_10bit_samples():
    assert np.allclose(hlg_graphics(np.ones((1, 1, 3), np.float32)), 0.75, atol=1e-5)
    frame = av.VideoFrame(160, 120, "yuv420p10le")
    frame.colorspace = 9
    frame.color_range = 1
    for i in range(3):
        native_plane(frame, i)[:] = 300 + i * 40
    before = [native_plane(frame, i).copy() for i in range(3)]
    blend_native(frame, [Face((60, 40, 20, 20))], cloud(), hlg=True)
    for i in range(3):
        assert np.array_equal(native_plane(frame, i)[:10, :10], before[i][:10, :10])
    assert 705 < native_plane(frame, 0)[50, 70] < 725


def test_cloud_core_becomes_opaque_without_changing_perimeter_or_uploaded_file(tmp_path):
    path = tmp_path / "generated-cloud.png"
    rgba = np.full((64, 64, 4), 240, np.uint8)
    rgba[:, :, 3] = 60
    rgba[16:48, 16:48, 3] = 147
    Image.fromarray(rgba).save(path)
    before = path.read_bytes()
    loaded = load_png(path)
    assert np.all(loaded[16:48, 16:48, 3] == 255)
    assert np.array_equal(loaded[:, :, :3], rgba[:, :, :3])
    assert np.array_equal(loaded[:16], rgba[:16])
    assert np.array_equal(loaded[48:], rgba[48:])
    assert path.read_bytes() == before
    rgba[30, 30, 3] = 127
    Image.fromarray(rgba).save(path)
    with pytest.raises(UserError):
        load_png(path)
