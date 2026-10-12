import pytest

from people_mosaic.domain import Face, hidden_faces
from people_mosaic.stabilize import stabilize_faces


def test_logo_track_preserves_score_and_override_without_person_identity():
    raw = [[Face((100 + i, 200, 30, 35), score=0.7 + i / 100)] for i in range(8)]
    result = stabilize_faces(raw, [i / 30 for i in range(8)])
    ids = {faces[0].track_id for faces in result}
    assert len(ids) == 1 and None not in ids
    for i, faces in enumerate(result):
        assert faces[0].score == raw[i][0].score
        assert faces[0].person is None and not faces[0].confident
        assert not hidden_faces(faces, set(), excluded_tracks=ids)


@pytest.mark.parametrize("change", ["cut", "gap", "empty", "backwards", "identity", "recognized"])
def test_detection_exclusion_does_not_cross_discontinuities(change):
    first = Face((100, 200, 30, 35), person=0 if change == "identity" else None)
    last = Face(
        (101, 200, 30, 35),
        person=1 if change in {"identity", "recognized"} else None,
        confident=change == "recognized",
    )
    frames, times, cuts = [[first], [last]], [0, 1 / 30], set()
    if change == "cut":
        cuts = {1}
    elif change == "gap":
        times = [0, 0.5]
    elif change == "empty":
        frames, times = [[first], [], [last]], [0, 1 / 30, 2 / 30]
    elif change == "backwards":
        times = [0, 0]
    result = stabilize_faces(frames, times, cuts)
    old_id, new_id = result[0][0].track_id, result[-1][0].track_id
    assert old_id != new_id
    assert hidden_faces(result[-1], set(), excluded_tracks={old_id})


def test_unknown_bridge_cannot_transfer_exclusion_to_another_candidate():
    result = stabilize_faces(
        [[Face((100, 200, 30, 35), person=person)] for person in [0, None, 1]],
        [0, 1 / 30, 2 / 30],
    )
    assert len({faces[0].track_id for faces in result}) == 3


def test_new_confident_identification_restarts_exclusion_even_with_same_candidate_id():
    result = stabilize_faces(
        [[Face((100, 200, 30, 35), person=2, confident=confident)] for confident in [False, True]],
        [0, 1 / 30],
    )
    old_id = result[0][0].track_id
    assert result[1][0].track_id != old_id
    assert hidden_faces(result[1], set(), excluded_tracks={old_id})


@pytest.mark.parametrize("direction", ["split", "merge"])
def test_competing_geometry_starts_new_exclusion_ids(direction):
    pair = [Face((80, 200, 30, 35)), Face((120, 200, 30, 35))]
    singleton = [Face((100, 200, 30, 35))]
    frames = [singleton, pair] if direction == "split" else [pair, singleton]
    result = stabilize_faces(frames, [0, 1 / 30])
    assert {face.track_id for face in result[0]}.isdisjoint({face.track_id for face in result[1]})


def test_cut_track_ids_are_never_reused_for_later_objects():
    result = stabilize_faces(
        [[Face((100, 200, 30, 35))] for _ in range(6)],
        [i / 30 for i in range(6)],
        {2, 4},
    )
    assert [faces[0].track_id for faces in result] == [0, 0, 1, 1, 2, 2]
