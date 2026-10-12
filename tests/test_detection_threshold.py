from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import av
import numpy as np
import pytest

from people_mosaic import engine
from people_mosaic.domain import Analysis, Face, hidden_faces
from people_mosaic.errors import UserError
from people_mosaic.models import digest
from people_mosaic.overlay import blend_native, cloud, native_plane


@pytest.mark.parametrize("threshold", [None, 0.81])
def test_detector_uses_floor_or_explicit_threshold_and_preserves_restored_score(
    monkeypatch, threshold
):
    created = []
    detection = np.array([[2, 3, 8, 9, *([1] * 10), 0.74]], np.float32)

    class Detector:
        def setInputSize(self, size):
            assert size == (80, 40)

        def detect(self, image):
            return None, detection.copy()

    def create(*args):
        created.append(args)
        return Detector()

    monkeypatch.setattr(engine.cv2, "FaceDetectorYN", SimpleNamespace(create=create))
    monkeypatch.setattr(
        engine.cv2, "FaceRecognizerSF", SimpleNamespace(create=lambda *args: object())
    )
    frame = av.VideoFrame.from_ndarray(np.zeros((160, 320, 3), np.uint8), format="bgr24")
    frame.pts, frame.time_base = 1, Fraction(1, 10)
    worker = engine.FrameAnalyzer(
        Path("detector"),
        Path("recognizer"),
        0,
        Event(),
        80,
        **({"score_threshold": threshold} if threshold is not None else {}),
    )

    _, found, _ = worker(frame)

    assert created[0][3] == (threshold if threshold is not None else 0.30)
    assert found[0][0].box == (8, 12, 32, 36)
    assert found[0][0].score == pytest.approx(0.74)
    assert found[0][0].feature is None


def analyze_mock_detections(monkeypatch, results):
    stream = SimpleNamespace(average_rate=10, duration=3, time_base=Fraction(1, 10))

    @contextmanager
    def decoder(*args):
        yield stream, iter(())

    monkeypatch.setattr(
        engine,
        "probe",
        lambda path: {"width": 64, "height": 48, "rotation": 0},
    )
    monkeypatch.setattr(engine, "digest", lambda path: "fingerprint")
    monkeypatch.setattr(engine, "model_paths", lambda: (Path("detector"), Path("recognizer")))
    monkeypatch.setattr(engine, "analysis_decoder", decoder)
    monkeypatch.setattr(engine, "analyzed_frames", lambda *args: (result for result in results))
    monkeypatch.setattr(
        engine,
        "follow_all",
        lambda previous, current, faces, *args: [
            (face.box[0] + 1, *face.box[1:]) for face in faces
        ],
    )
    monkeypatch.setattr(engine, "stabilize_faces", lambda frames, *args: frames)

    return engine.analyze(Path("input.mkv"), lambda *args: None, Event())


def test_optical_flow_hold_retains_original_detection_score_without_identity(monkeypatch):
    source_face = Face((10, 10, 20, 20), person=0, confident=True, score=0.41)
    gray = np.zeros((48, 64), np.uint8)
    results = [(0.0, [(source_face, None)], gray), (0.1, [], gray), (0.2, [], gray)]
    analysis = analyze_mock_detections(monkeypatch, results)

    assert [frame[0].score for frame in analysis.frames] == [0.41, 0.41, 0.41]
    assert analysis.frames[0][0].confident
    assert all(not frame[0].confident and frame[0].person is None for frame in analysis.frames[1:])
    assert all(not hidden_faces(frame, set(), score_threshold=0.65) for frame in analysis.frames)


def test_weak_reacquisition_preserves_existing_protection_then_expires(monkeypatch):
    gray = np.zeros((48, 64), np.uint8)
    raw_scores = [(0.0, 0.90), (0.1, 0.40), (0.4, 0.40), (0.5, 0.40), (0.6, 0.40)]
    results = [
        (timestamp, [(Face((10, 10, 20, 20), score=score), None)], gray)
        for timestamp, score in raw_scores
    ]

    analysis = analyze_mock_detections(monkeypatch, results)

    assert all(len(faces) == 1 for faces in analysis.frames)
    assert [faces[0].score for faces in analysis.frames] == [0.90, 0.90, 0.90, 0.90, 0.40]
    assert all(hidden_faces(faces, set()) for faces in analysis.frames[:-1])
    assert hidden_faces(analysis.frames[-1], set()) == []
    assert all(not faces[0].confident and faces[0].person is None for faces in analysis.frames)


def test_intermediate_actual_score_keeps_own_horizon_through_missing_and_weak_frames(monkeypatch):
    gray = np.zeros((48, 64), np.uint8)
    results = [
        (
            timestamp,
            [(Face((10, 10, 20, 20), score=score), None)] if score is not None else [],
            gray,
        )
        for timestamp, score in [(0.0, 0.90), (0.4, 0.80), (0.6, None), (0.8, 0.40), (0.91, None)]
    ]

    analysis = analyze_mock_detections(monkeypatch, results)

    assert all(len(faces) == 1 for faces in analysis.frames)
    assert [faces[0].score for faces in analysis.frames] == [0.90, 0.90, 0.80, 0.80, 0.40]
    assert all(hidden_faces(faces, set()) for faces in analysis.frames[:-1])
    assert hidden_faces(analysis.frames[-1], set()) == []


def test_scene_cut_drops_previous_detection_score_history(monkeypatch):
    results = [
        (0.0, [(Face((10, 10, 20, 20), score=0.90), None)], np.zeros((48, 64), np.uint8)),
        (0.1, [(Face((10, 10, 20, 20), score=0.40), None)], np.full((48, 64), 100, np.uint8)),
        (0.2, [], np.full((48, 64), 100, np.uint8)),
    ]

    analysis = analyze_mock_detections(monkeypatch, results)

    assert [faces[0].score for faces in analysis.frames] == [0.90, 0.40, 0.40]
    assert all(not hidden_faces(faces, set()) for faces in analysis.frames[1:])


def test_threshold_and_track_exclusions_have_identical_preview_and_export_masks():
    faces = [
        Face((0, 0, 20, 20), score=0.74, track_id=1),
        Face((30, 0, 20, 20), score=0.75, track_id=2),
        Face((60, 0, 20, 20), score=0.99, track_id=3),
        Face((90, 0, 20, 20), person=4, confident=True, score=0.99, track_id=4),
        Face((120, 0, 20, 20), person=4, confident=False, score=0.99, track_id=5),
    ]
    manual = [Face((150, 0, 20, 20), score=0.1, track_id=3)]
    revealed, excluded = {4}, {3}
    preview = hidden_faces(faces, revealed, score_threshold=0.75, excluded_tracks=excluded) + manual
    seen = []
    video = SimpleNamespace(index=0)

    def packets():
        for index in range(2):
            if index:
                # Caller edits during a worker run cannot change later masks.
                revealed.clear()
                excluded.clear()
                manual.clear()
            frame = SimpleNamespace(
                pts=index, time_base=Fraction(1, 10), make_writable=lambda: None
            )
            yield SimpleNamespace(stream=video, decode=lambda frame=frame: [frame])

    def composite(frame, masks):
        seen.append((frame.pts, masks))
        return frame

    list(
        engine.composited_packets(
            SimpleNamespace(demux=packets),
            video,
            SimpleNamespace(frames=[faces, faces], times=[0, 0.1]),
            revealed,
            manual,
            composite,
            Event(),
            score_threshold=0.75,
            excluded_tracks=excluded,
        )
    )

    assert preview == [faces[1], faces[4], Face((150, 0, 20, 20), score=0.1, track_id=3)]
    assert [masks for _, masks in sorted(seen)] == [preview, preview]


@pytest.mark.parametrize("threshold", [0.29, 1, float("nan"), float("inf"), None, "invalid"])
def test_invalid_export_threshold_is_rejected_before_input_io(tmp_path, threshold, monkeypatch):
    def forbidden(*args):
        pytest.fail("Invalid settings must not read or export video")

    monkeypatch.setattr(engine, "digest", forbidden)
    with pytest.raises(UserError, match="0.30"):
        engine.export(
            SimpleNamespace(path=tmp_path / "input.mkv"),
            tmp_path / "output.mkv",
            set(),
            cloud(),
            [],
            False,
            lambda *args: None,
            Event(),
            score_threshold=threshold,
        )
    assert not (tmp_path / "output.mkv").exists()


@pytest.mark.parametrize("threshold", [None, 0.85])
def test_lossless_export_renders_only_selected_scores_and_tracks_with_manual_masks(
    tmp_path, threshold
):
    path = tmp_path / "source.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 240, 120, "yuv420p"
        for index in range(3):
            image = np.full((120, 240, 3), (10 + index, 60, 130), np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = index, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        originals = list(container.decode(video=0))
    faces = [
        Face((20, 40, 20, 20), score=0.70, track_id=1),
        Face((70, 40, 20, 20), score=0.90, track_id=2),
        Face((120, 40, 20, 20), score=0.95, track_id=3),
    ]
    manual = Face((180, 40, 20, 20), score=0.1)
    analysis = Analysis(
        path,
        digest(path),
        240,
        120,
        10,
        [faces for _ in originals],
        [float(frame.pts * frame.time_base) for frame in originals],
    )
    output = engine.export(
        analysis,
        tmp_path / "result.mkv",
        set(),
        cloud(),
        [manual],
        False,
        lambda *args: None,
        Event(),
        excluded_tracks={3},
        **({"score_threshold": threshold} if threshold is not None else {}),
    )
    with av.open(str(output)) as container:
        rendered = list(container.decode(video=0))

    assert len(rendered) == len(originals)
    for original, result in zip(originals, rendered, strict=True):
        blend_native(original, [faces[1], manual], cloud())
        assert all(
            np.array_equal(native_plane(original, index), native_plane(result, index))
            for index in range(3)
        )
