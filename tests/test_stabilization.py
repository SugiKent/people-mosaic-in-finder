import numpy as np
import pytest

from people_mosaic.domain import Face, hidden_faces
from people_mosaic.stabilize import stabilize_faces


def centre(face):
    x, y, w, h = face.box
    return np.array([x + w / 2, y + h / 2])


def assert_contains(stable, raw):
    x, y, w, h = stable.box
    rx, ry, rw, rh = raw.box
    assert x <= rx + 1e-9 and y <= ry + 1e-9
    assert x + w >= rx + rw - 1e-9 and y + h >= ry + rh - 1e-9


def test_stationary_position_and_size_jitter_is_reduced_with_raw_coverage():
    frames = [
        [Face((100 + [-3, 3, -2, 2, -1, 1][i % 6], 120 + (-1) ** i * 2, 80 + i % 3 * 2, 100))]
        for i in range(120)
    ]
    result = stabilize_faces(frames, [i / 60 for i in range(len(frames))])
    raw_centres = np.array([centre(f[0]) for f in frames])
    stable_centres = np.array([centre(f[0]) for f in result])
    raw_jitter = np.linalg.norm(np.diff(raw_centres, axis=0), axis=1).mean()
    stable_jitter = np.linalg.norm(np.diff(stable_centres, axis=0), axis=1).mean()
    assert stable_jitter < raw_jitter * 0.25
    assert np.mean(np.abs(np.diff([f[0].box[2] for f in result]))) < np.mean(
        np.abs(np.diff([f[0].box[2] for f in frames]))
    )
    for raw, stable in zip(frames, result, strict=True):
        assert_contains(stable[0], raw[0])


def test_deliberate_movement_follows_quickly_and_never_leaves_face_uncovered():
    frames = [[Face((100 + i * 10, 50, 60, 80))] for i in range(60)]
    result = stabilize_faces(frames, [i / 60 for i in range(60)])
    for raw, stable in zip(frames, result, strict=True):
        assert_contains(stable[0], raw[0])
        assert np.linalg.norm(centre(stable[0]) - centre(raw[0])) <= 15.01
        assert stable[0].box[2] <= raw[0].box[2] * 1.5 + 1e-8
    assert centre(result[-1][0])[0] - centre(result[0][0])[0] > 570


@pytest.mark.parametrize("person", [None, 7])
def test_two_faces_with_same_or_unknown_identity_are_matched_one_to_one(person):
    frames = [
        [Face((10, 20, 40, 60), person), Face((200, 20, 40, 60), person)],
        [Face((202, 20, 40, 60), person), Face((12, 20, 40, 60), person)],
    ]
    result = stabilize_faces(frames, [0, 1 / 60])
    assert np.allclose(centre(result[1][0]), centre(frames[0][1]))
    assert np.allclose(centre(result[1][1]), centre(frames[0][0]))
    # When two new faces are both close to just one old face, only one can
    # inherit its geometry; the other starts at its own raw box.
    result = stabilize_faces(
        [
            [Face((10, 20, 40, 60), person)],
            [Face((11, 20, 40, 60), person), Face((12, 20, 40, 60), person)],
        ],
        [0, 1 / 60],
    )
    assert np.allclose(centre(result[1][0]), [30, 50])
    assert np.allclose(centre(result[1][1]), [32, 50])
    assert_contains(result[1][1], Face((12, 20, 40, 60)))


def test_identity_confidence_and_features_are_never_inherited_from_a_track():
    feature = np.array([1.0, 0.0])
    faces = [
        Face((100, 80, 80, 80), 1, True, feature),
        Face((102, 80, 80, 80), None, False),
        Face((101, 80, 80, 80), 2, False),
        Face((103, 80, 80, 80), 1, False),
        Face((102, 80, 80, 80), 1, True),
    ]
    frames = [[face] for face in faces]
    result = stabilize_faces(frames, [i / 60 for i in range(len(faces))])
    for raw, stable in zip(faces, [f[0] for f in result], strict=True):
        assert stable.person == raw.person and stable.confident == raw.confident
        assert stable.feature is raw.feature
        assert bool(hidden_faces([stable], {1})) == bool(hidden_faces([raw], {1}))
        assert_contains(stable, raw)


@pytest.mark.parametrize("reset", ["cut", "jump", "gap", "missing", "scale"])
def test_cuts_missing_faces_large_jumps_and_gaps_reset_without_trails(reset):
    first = Face((100, 80, 80, 80))
    next_face = Face((103, 82, 80, 80))
    frames, times, cuts = [[first], [next_face]], [0, 1 / 60], set()
    if reset == "cut":
        cuts = {1}
    elif reset == "jump":
        next_face.box = (500, 80, 80, 80)
    elif reset == "gap":
        times[1] = 0.4
    elif reset == "missing":
        frames, times = [[first], [], [next_face]], [0, 1 / 60, 2 / 60]
    else:
        next_face.box = (100, 80, 200, 200)
    result = stabilize_faces(frames, times, cuts)
    assert np.allclose(centre(result[-1][0]), centre(next_face))
    assert_contains(result[-1][0], next_face)
    assert len(result[-1]) == 1
    if reset == "missing":
        assert result[1] == []


def test_smoothing_uses_elapsed_time_for_vfr_not_frame_count():
    frames = [[Face((100, 80, 100, 100))], [Face((120, 80, 100, 100))]]
    short = stabilize_faces(frames, [0, 1 / 120])
    long = stabilize_faces(frames, [0, 0.1])
    assert centre(long[1][0])[0] > centre(short[1][0])[0]
    assert_contains(short[1][0], frames[1][0])
    assert_contains(long[1][0], frames[1][0])


def test_size_shrinks_gradually_after_actual_zoom_change():
    frames = [[Face((100, 100, 100, 100))]] + [[Face((110, 110, 80, 80))] for _ in range(90)]
    result = stabilize_faces(frames, [i / 60 for i in range(len(frames))])
    sizes = [frame[0].box[2] for frame in result]
    assert sizes[1] == sizes[0]
    assert sizes[30] == sizes[0]
    assert sizes[50] < sizes[0]
    assert all(a >= b for a, b in zip(sizes, sizes[1:]))
    assert sizes[-1] < 105
    for raw, stable in zip(frames, result, strict=True):
        assert_contains(stable[0], raw[0])


def test_real_cloud_placement_footprint_stops_breathing_for_small_face_jitter():
    from people_mosaic.overlay import PreparedOverlay, cloud

    frames = [[Face((100 + (-1) ** i * 2, 100 + (-1) ** i * 2, 50, 50))] for i in range(60)]
    stable = stabilize_faces(frames, [i / 60 for i in range(len(frames))])
    overlay = PreparedOverlay(cloud())

    def footprint(face):
        x, y, patch = overlay.placement(face.box, 640, 480)
        return [x, y, patch.shape[1], patch.shape[0]]

    raw_edges = np.array([footprint(frame[0]) for frame in frames])
    stable_edges = np.array([footprint(frame[0]) for frame in stable])
    assert np.abs(np.diff(stable_edges[8:], axis=0)).sum() == 0
    assert np.abs(np.diff(raw_edges[8:], axis=0)).sum() > 0
    assert (
        np.abs(np.diff(stable_edges, axis=0)).sum() < np.abs(np.diff(raw_edges, axis=0)).sum() * 0.1
    )


def test_tiny_faces_and_a_stale_large_detection_keep_bounded_padding():
    for size in (8, 20, 40, 100):
        frames = [[Face((100, 100, size, size))]] + [
            [Face((100 + i % 4, 100 + i % 3, size * 0.65, size * 0.65))] for i in range(80)
        ]
        result = stabilize_faces(frames, [i / 60 for i in range(len(frames))])
        for raw, stable in zip(frames, result, strict=True):
            assert_contains(stable[0], raw[0])
            assert stable[0].box[2] <= size * 1.5 + 1e-8
            assert stable[0].box[3] <= size * 1.5 + 1e-8
        assert result[-1][0].box[2] <= frames[-1][0].box[2] * 1.5 + 1e-8
        assert result[-1][0].box[3] <= frames[-1][0].box[3] * 1.5 + 1e-8


def test_twelve_pixel_detection_change_on_eighty_pixel_face_is_now_held():
    # This motion crossed the previous 8% / 3px deadband. A cloud can now stay
    # still while a detector shifts moderately, still fully covering its face.
    frames = [[Face((100, 100, 80, 80))], [Face((112, 100, 80, 80))]]
    result = stabilize_faces(frames, [0, 1 / 60])
    assert np.allclose(centre(result[1][0]), centre(result[0][0]))
    assert_contains(result[1][0], frames[1][0])


@pytest.mark.parametrize("size,jitter", [(50, 4), (80, 6), (100, 6)])
def test_moderate_position_and_dimensions_jitter_keep_cloud_footprint_steady(size, jitter):
    from people_mosaic.overlay import PreparedOverlay, cloud

    frames = []
    for i in range(180):
        delta = jitter if i % 2 else -jitter
        width, height = size + (i % 3 - 1) * 3, size + (i % 4 - 2) * 2
        frames.append([Face((200 + delta - width / 2, 200 - delta - height / 2, width, height))])
    result = stabilize_faces(frames, [i / 60 for i in range(len(frames))])
    overlay = PreparedOverlay(cloud())

    def footprint(face):
        x, y, patch = overlay.placement(face.box, 640, 480)
        return [x, y, patch.shape[1], patch.shape[0]]

    raw = np.array([footprint(frame[0]) for frame in frames])
    stable = np.array([footprint(frame[0]) for frame in result])
    assert (
        np.abs(np.diff(stable[60:], axis=0)).sum() < np.abs(np.diff(raw[60:], axis=0)).sum() * 0.1
    )
    for raw_frame, stable_frame in zip(frames, result, strict=True):
        assert_contains(stable_frame[0], raw_frame[0])


@pytest.mark.parametrize("size,jitter", [(50, 4), (80, 6), (100, 6)])
def test_short_period_dimensions_alone_do_not_make_cloud_breathe(size, jitter):
    from people_mosaic.overlay import PreparedOverlay, cloud

    frames = []
    for i in range(180):
        dimension = size + (jitter if i % 2 else -jitter)
        frames.append([Face((200 - dimension / 2, 200 - dimension / 2, dimension, dimension))])
    result = stabilize_faces(frames, [i / 60 for i in range(len(frames))])
    overlay = PreparedOverlay(cloud())
    shapes = [overlay.placement(frame[0].box, 640, 480)[2].shape for frame in result]
    assert len(set(shapes[30:])) == 1
    for raw, stable in zip(frames, result, strict=True):
        assert_contains(stable[0], raw[0])


def test_continuous_shrink_wait_is_based_on_elapsed_time_in_vfr():
    frames = [[Face((100, 100, 100, 100))]] + [Face((110, 110, 80, 80)) for _ in range(6)]
    frames = [frame if isinstance(frame, list) else [frame] for frame in frames]
    times = [0, 0.1, 0.25, 0.4, 0.55, 0.69, 0.72]
    result = stabilize_faces(frames, times)
    sizes = [frame[0].box[2] for frame in result]
    # The rolling bound may tighten after 0.6 seconds, but shrink must not
    # commence just because a small number of VFR frames were observed.
    assert sizes[:6] == [sizes[0]] * 6
    assert sizes[-1] < sizes[0]
    assert sizes[-1] >= frames[-1][0].box[2]
    for raw, stable in zip(frames, result, strict=True):
        assert_contains(stable[0], raw[0])


def test_short_dimension_jitter_near_the_size_cap_does_not_force_shrink_each_frame():
    from people_mosaic.overlay import PreparedOverlay, cloud

    frames = []
    for i in range(120):
        size = 100 + (6 if i % 2 else -6)
        # Sustained movement brings the covering box near its size cap. The
        # fluctuating current detection must not repeatedly lower that cap.
        cx = 400 + i * 12
        frames.append([Face((cx - size / 2, 200 - size / 2, size, size))])
    result = stabilize_faces(frames, [i / 60 for i in range(len(frames))])
    overlay = PreparedOverlay(cloud())
    shapes = [overlay.placement(frame[0].box, 2400, 480)[2].shape for frame in result]
    assert len(set(shapes[40:])) == 1
    for raw, stable in zip(frames, result, strict=True):
        assert_contains(stable[0], raw[0])
        assert stable[0].box[2] <= 106 * 1.5 + 1e-8
    assert centre(result[-1][0])[0] - centre(result[0][0])[0] > 1390
