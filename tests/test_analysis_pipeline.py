from threading import Barrier, Event, Lock
from threading import enumerate as active_threads
from types import SimpleNamespace

import numpy as np
import pytest

from people_mosaic.domain import Face
from people_mosaic.engine import (
    ANALYSIS_WORKERS,
    Cancelled,
    analyzed_frames,
    follow_all,
    tracking_gray,
)


def frames(count=12):
    return [SimpleNamespace(pts=index, time_base=0.1) for index in range(count)]


def test_parallel_results_keep_order_and_bound_pending_frames():
    rendezvous = Barrier(2)
    finished = Event()
    decoded = []
    completed = []
    lock = Lock()

    def decode(stream):
        for frame in frames():
            decoded.append(frame.pts)
            yield frame

    def work(frame):
        if frame.pts < 2:
            rendezvous.wait(timeout=5)
            if frame.pts == 0:
                assert finished.wait(timeout=5)
        with lock:
            completed.append(frame.pts)
        if frame.pts == 1:
            finished.set()
        return frame.pts

    results = []
    for result in analyzed_frames(iter(decode(None)), work, Event()):
        results.append(result)
        assert len(decoded) - len(results) <= ANALYSIS_WORKERS * 2 - 1
    assert results == list(range(12))
    assert completed.index(1) < completed.index(0)


@pytest.mark.parametrize("failure", ["cancel", "error"])
def test_pipeline_closes_workers_after_cancellation_or_failure(failure):
    cancel = Event()

    def work(frame):
        if failure == "error" and frame.pts == 3:
            raise ValueError("inference failed")
        return frame.pts

    container = SimpleNamespace(decode=lambda stream: iter(frames()))
    iterator = analyzed_frames(iter(container.decode(None)), work, cancel)
    assert next(iterator) == 0
    if failure == "cancel":
        cancel.set()
    with pytest.raises(Cancelled if failure == "cancel" else ValueError):
        list(iterator)
    assert not any(thread.name.startswith("face") for thread in active_threads())


def test_batched_tracking_restores_source_coordinates_and_holds_lost_faces(monkeypatch):
    gray = np.zeros((540, 960), np.uint8)
    faces = [Face((400, 200, 160, 120)), Face((800, 300, 80, 80))]
    calls = []

    def flow(previous, current, points, initial):
        calls.append(points)
        status = np.ones((18, 1), np.uint8)
        status[9:] = 0
        return points + np.array([[[3, -2]]], np.float32), status, None

    monkeypatch.setattr("people_mosaic.engine.cv2.calcOpticalFlowPyrLK", flow)
    boxes = follow_all(gray, gray, faces, 3840, 2160)
    assert len(calls) == 1
    assert boxes == [(412.0, 192.0, 160, 120), faces[1].box]
    assert follow_all(gray, gray, [], 3840, 2160) == []
    assert len(calls) == 1
    assert tracking_gray(np.zeros((2160, 3840, 3), np.uint8)).shape == (540, 960)


def test_inference_lanes_own_models_and_keep_rotated_full_resolution(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from fractions import Fraction
    from pathlib import Path

    import av

    from people_mosaic.engine import FrameAnalyzer

    rendezvous = Barrier(2)
    detectors = []
    recognizers = []
    bgr = np.zeros((80, 160, 3), np.uint8)
    bgr[:, :80] = (12, 34, 56)

    class Detector:
        def __init__(self, *args):
            detectors.append(self)

        def setInputSize(self, size):
            self.size = size

        def detect(self, image):
            rendezvous.wait(timeout=5)
            assert self.size == (80, 160)
            assert np.array_equal(image, np.rot90(bgr))
            return None, np.array([[10, 20, 40, 40, *([0] * 10), 0.95]], np.float32)

    class Recognizer:
        def __init__(self, *args):
            recognizers.append(self)

        def alignCrop(self, image, detection):
            return image[:40, :40].copy()

        def feature(self, aligned):
            return np.array([[3, 4]], np.float32)

    monkeypatch.setattr("people_mosaic.engine.cv2.FaceDetectorYN", SimpleNamespace(create=Detector))
    monkeypatch.setattr(
        "people_mosaic.engine.cv2.FaceRecognizerSF", SimpleNamespace(create=Recognizer)
    )
    worker = FrameAnalyzer(Path("detector"), Path("recognizer"), 1, Event())
    originals = []
    for index in range(2):
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts, frame.time_base = index, Fraction(1, 10)
        originals.append(frame)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, originals))
    assert len(detectors) == len(recognizers) == 2
    assert [result[0] for result in results] == [0, 0.1]
    for _, found, _ in results:
        face, _ = found[0]
        assert face.box == (10, 20, 40, 40)
        assert np.allclose(face.feature, [0.6, 0.8])
        assert face.person is None and not face.confident


@pytest.mark.parametrize("rotation", [0, 1])
def test_fast_detection_restores_boxes_and_landmarks_before_native_identity_crop(
    monkeypatch, rotation
):
    from fractions import Fraction
    from pathlib import Path

    import av

    from people_mosaic.engine import FrameAnalyzer

    detection = np.array([[8, 4, 10, 12, 9, 5, 15, 5, 12, 9, 9, 12, 15, 12, 0.95]], np.float32)
    aligned_inputs = []

    class Detector:
        def setInputSize(self, size):
            assert size == ((80, 45) if rotation == 0 else (45, 80))

        def detect(self, image):
            assert max(image.shape[:2]) == 80
            return None, detection.copy()

    class Recognizer:
        def alignCrop(self, image, restored):
            aligned_inputs.append((image.shape, restored.copy()))
            return image[:40, :40]

        def feature(self, aligned):
            return np.array([[1.0, 0.0]], np.float32)

    monkeypatch.setattr(
        "people_mosaic.engine.cv2.FaceDetectorYN", SimpleNamespace(create=lambda *a: Detector())
    )
    monkeypatch.setattr(
        "people_mosaic.engine.cv2.FaceRecognizerSF", SimpleNamespace(create=lambda *a: Recognizer())
    )
    original = av.VideoFrame.from_ndarray(np.zeros((180, 320, 3), np.uint8), format="bgr24")
    original.pts, original.time_base = 2, Fraction(1, 10)
    timestamp, found, _ = FrameAnalyzer(
        Path("detector"), Path("recognizer"), rotation, Event(), 80
    )(original)
    assert timestamp == 0.2
    assert found[0][0].box == (32, 16, 40, 48)
    shape, restored = aligned_inputs[0]
    assert shape == ((180, 320, 3) if rotation == 0 else (320, 180, 3))
    assert np.allclose(restored[:14], detection[0, :14] * 4)
    assert restored[14] == detection[0, 14]


@pytest.mark.parametrize("failure", ["open", "first_decode"])
def test_apple_decoder_falls_back_before_first_result_and_closes_containers(monkeypatch, failure):
    import av

    from people_mosaic.engine import analysis_decoder

    attempts = []
    closed = []
    frame = SimpleNamespace(pts=0)
    stream = SimpleNamespace(codec_context=SimpleNamespace())

    def open_source(path, **options):
        accelerated = bool(options)
        attempts.append(accelerated)
        if accelerated and failure == "open":
            raise av.error.FFmpegError(5, "hardware unavailable")

        def decode(video):
            if accelerated:
                raise av.error.FFmpegError(5, "hardware unavailable")
            yield frame

        return SimpleNamespace(
            streams=SimpleNamespace(video=[stream]),
            decode=decode,
            close=lambda: closed.append(accelerated),
        )

    monkeypatch.setattr("people_mosaic.engine.sys.platform", "darwin")
    monkeypatch.setattr("people_mosaic.engine.hwdevices_available", lambda: ["videotoolbox"])
    monkeypatch.setattr("people_mosaic.engine.HWAccel", lambda *a, **k: object())
    monkeypatch.setattr("people_mosaic.engine.av.open", open_source)
    with analysis_decoder("video.mp4", True) as (_, decoded):
        assert list(decoded) == [frame]
    assert attempts == [True, False]
    assert closed == ([False] if failure == "open" else [True, False])


def test_decoder_does_not_restart_after_partial_results(monkeypatch):
    import av

    from people_mosaic.engine import analysis_decoder

    attempts = []
    closed = []
    stream = SimpleNamespace(codec_context=SimpleNamespace())

    def decode(video):
        yield SimpleNamespace(pts=0)
        raise av.error.FFmpegError(5, "late hardware failure")

    def open_source(path, **options):
        attempts.append(options)
        return SimpleNamespace(
            streams=SimpleNamespace(video=[stream]),
            decode=decode,
            close=lambda: closed.append(True),
        )

    monkeypatch.setattr("people_mosaic.engine.sys.platform", "darwin")
    monkeypatch.setattr("people_mosaic.engine.hwdevices_available", lambda: ["videotoolbox"])
    monkeypatch.setattr("people_mosaic.engine.HWAccel", lambda *a, **k: object())
    monkeypatch.setattr("people_mosaic.engine.av.open", open_source)
    with pytest.raises(av.error.FFmpegError):
        with analysis_decoder("video.mp4", True) as (_, decoded):
            list(decoded)
    assert len(attempts) == 1 and closed == [True]
