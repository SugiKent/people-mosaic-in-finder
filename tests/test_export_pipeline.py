from fractions import Fraction
from threading import Barrier, Event
from threading import enumerate as active_threads
from types import SimpleNamespace

import av
import numpy as np
import pytest

from people_mosaic import engine
from people_mosaic.domain import Analysis, Face
from people_mosaic.models import digest
from people_mosaic.overlay import blend_native, cloud, native_plane


def test_parallel_composition_preserves_order_and_bounds_native_frames():
    barrier, second_done = Barrier(2), Event()
    submitted, consumed, writable = [], [], []
    frames = [
        SimpleNamespace(
            pts=i,
            time_base=Fraction(1, 10),
            make_writable=lambda i=i: writable.append(i),
        )
        for i in range(24)
    ]
    analysis = SimpleNamespace(frames=[[]] * len(frames), times=[i / 10 for i in range(24)])
    video = SimpleNamespace(index=0)

    def packets():
        for frame in frames:
            submitted.append(frame.pts)
            yield SimpleNamespace(stream=video, decode=lambda frame=frame: [frame])

    def compose(frame, masks):
        assert frame.pts in writable
        if frame.pts < 2:
            barrier.wait(timeout=5)
            if frame.pts == 0:
                assert second_done.wait(timeout=5)
            else:
                second_done.set()
        return frame.pts

    source = SimpleNamespace(demux=packets)
    for is_video, value in engine.composited_packets(
        source, video, analysis, set(), [], compose, Event()
    ):
        assert is_video
        consumed.append(value)
        assert len(submitted) - len(consumed) < engine.EXPORT_PENDING
    assert consumed == list(range(24))
    assert writable == list(range(24))
    assert not any(t.name.startswith("composite") for t in active_threads())


@pytest.mark.parametrize("failure", ["cancel", "worker"])
def test_export_workers_close_on_cancellation_or_compositor_error(failure):
    cancel = Event()
    video = SimpleNamespace(index=0)
    frame = SimpleNamespace(pts=0, time_base=Fraction(1, 10), make_writable=lambda: None)
    packet = SimpleNamespace(stream=video, decode=lambda: [frame])
    source = SimpleNamespace(demux=lambda: [packet])
    analysis = SimpleNamespace(frames=[[]], times=[0])

    def compose(original, masks):
        if failure == "worker":
            raise ValueError("compositor failed")
        cancel.set()
        return original

    with pytest.raises(engine.Cancelled if failure == "cancel" else ValueError):
        list(engine.composited_packets(source, video, analysis, set(), [], compose, cancel))
    assert not any(t.name.startswith("composite") for t in active_threads())


def test_hardware_h264_initialization_failure_uses_software_before_export(monkeypatch):
    class HardwareContext:
        def open(self):
            raise av.error.FFmpegError(5, "hardware unavailable")

    monkeypatch.setattr(engine.sys, "platform", "darwin")
    monkeypatch.setattr(engine.av, "codecs_available", {"h264_videotoolbox", "libx264"})
    monkeypatch.setattr(
        engine.av, "CodecContext", SimpleNamespace(create=lambda *a: HardwareContext())
    )
    analysis = SimpleNamespace(width=1920, height=1080, fps=30, hlg=False)
    assert engine.mp4_codec(analysis, {}) == "libx264"


@pytest.mark.parametrize("mp4", [False, True])
@pytest.mark.parametrize("start", [0, 1200])
def test_interframe_vfr_export_pixels_audio_offsets_and_duration(tmp_path, mp4, start, monkeypatch):
    # H264 reference buffers are shared. Mutating decoded pixels without
    # make_writable corrupts later frames, unlike the old FFV1-only fixture.
    source = tmp_path / "interframe.mkv"
    times = [start + value for value in [0, 70, 120, 210, 270, 400, 490, 550, 690, 750, 850, 990]]
    with av.open(str(source), "w", options={"avoid_negative_ts": "disabled"}) as container:
        video = container.add_stream("libx264", rate=20)
        video.width, video.height, video.pix_fmt = 320, 240, "yuv420p"
        video.time_base = video.codec_context.time_base = Fraction(1, 1000)
        video.options = {"crf": "10", "preset": "medium"}
        audio = container.add_stream("pcm_s16le", rate=48000)
        audio.layout = "mono"
        for i, pts in enumerate(times):
            image = np.zeros((240, 320, 3), np.uint8)
            image[:] = (i * 15, 90, 170)
            image[20 + i * 2 : 100 + i * 2, 20:170] = (80, i * 13, 100)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base, frame.duration = pts, Fraction(1, 1000), 50
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        samples = (
            np.rint(15000 * np.sin(np.arange(48000) * (2 * np.pi * 440 / 48000)))
            .astype(np.int16)
            .reshape(1, -1)
        )
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = 48000
        frame.pts, frame.time_base = round((start / 1000 + 0.1) * 48000), Fraction(1, 48000)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    with av.open(str(source)) as container:
        originals = list(container.decode(video=0))
    analysis = Analysis(
        source,
        digest(source),
        320,
        240,
        20,
        [[Face((30, 40, 40, 40))] for _ in originals],
        [float(frame.pts * frame.time_base) for frame in originals],
    )
    output = tmp_path / ("result.mp4" if mp4 else "result.mkv")
    monkeypatch.setattr(engine, "mp4_codec", lambda *args: "libx264")
    remux = engine.remux_public_mp4
    remux_calls = []

    def packet_signature(path):
        with av.open(str(path)) as container:
            return [
                (
                    packet.pts * packet.time_base,
                    packet.dts * packet.time_base,
                    packet.duration * packet.time_base,
                    bytes(packet),
                )
                for packet in container.demux(video=0)
                if packet.dts is not None
            ]

    def checked_remux(video_path, audio_path, result_path, cancel):
        signature = packet_signature(video_path)
        remux(video_path, audio_path, result_path, cancel)
        # Old FFmpeg CLI versions infer a different final VFR duration during
        # stream copy. Our mux must preserve every compressed video packet,
        # including its exact PTS/DTS/duration, without another video encoding.
        assert packet_signature(result_path) == signature
        remux_calls.append(True)

    monkeypatch.setattr(engine, "remux_public_mp4", checked_remux)
    engine.export(analysis, output, set(), cloud(), [], mp4, lambda *a: None, Event())
    assert bool(remux_calls) == mp4
    with av.open(str(output)) as container:
        results = list(container.decode(video=0))
    assert len(results) == len(originals)
    for original, result in zip(originals, results, strict=True):
        assert (
            abs(float(original.pts * original.time_base) - float(result.pts * result.time_base))
            < 0.001
        )
        if not mp4:
            original.make_writable()
            blend_native(original, [Face((30, 40, 40, 40))], cloud())
            assert all(
                np.array_equal(native_plane(original, i), native_plane(result, i)) for i in range(3)
            )
    with av.open(str(source)) as container:
        source_audio = list(container.decode(audio=0))
    with av.open(str(output)) as container:
        result_audio = list(container.decode(audio=0))
    assert (
        abs(
            float(source_audio[0].pts * source_audio[0].time_base)
            - float(result_audio[0].pts * result_audio[0].time_base)
        )
        <= 0.023
    )

    # AAC has encoder priming frames; verify the signal onset, not just the
    # first decoded (potentially silent) packet, retains the +100ms offset.
    def onset(frames):
        samples = np.concatenate([f.to_ndarray().ravel() for f in frames]).astype(float)
        audible = np.flatnonzero(np.abs(samples) > np.max(np.abs(samples)) * 0.2)
        return float(frames[0].pts * frames[0].time_base) + audible[0] / 48000

    assert abs(onset(source_audio) - onset(result_audio)) < 0.005
    assert (
        abs(
            float(results[-1].duration * results[-1].time_base)
            - float(originals[-1].duration * originals[-1].time_base)
        )
        < 0.002
    )
    if not mp4:
        assert np.array_equal(
            np.concatenate([f.to_ndarray().ravel() for f in source_audio]),
            np.concatenate([f.to_ndarray().ravel() for f in result_audio]),
        )
    assert digest(source) == analysis.fingerprint
    assert not list(tmp_path.glob(".people-mosaic-*"))
