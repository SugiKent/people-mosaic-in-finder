import subprocess
from threading import Event

import av
import numpy as np
import pytest

from people_mosaic.domain import Analysis, Face
from people_mosaic.engine import Cancelled, export
from people_mosaic.errors import UserError
from people_mosaic.models import digest
from people_mosaic.overlay import blend_native, cloud, native_plane


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
    )
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    analysis = Analysis(
        path,
        digest(path),
        160,
        120,
        10,
        [[Face((60, 40, 20, 20))] for _ in frames],
        [float(frame.pts * frame.time_base) for frame in frames],
    )
    return analysis


def audio_samples(path):
    with av.open(str(path)) as container:
        return np.concatenate([frame.to_ndarray().ravel() for frame in container.decode(audio=0)])


def test_lossless_export_pixels_timing_audio_and_input(source, tmp_path):
    output = tmp_path / "result.mkv"
    export(source, output, set(), cloud(), [], False, lambda *args: None, Event())
    with av.open(str(source.path)) as original, av.open(str(output)) as result:
        a, b = list(original.decode(video=0)), list(result.decode(video=0))
    assert len(a) == len(b) == 10
    for original, result in zip(a, b, strict=True):
        assert (
            abs(float(original.pts * original.time_base) - float(result.pts * result.time_base))
            < 0.001
        )
        blend_native(original, [Face((60, 40, 20, 20))], cloud())
        assert all(
            np.array_equal(native_plane(original, i), native_plane(result, i)) for i in range(3)
        )
    assert np.array_equal(audio_samples(source.path), audio_samples(output))
    assert digest(source.path) == source.fingerprint
    assert not list(tmp_path.glob(".people-mosaic-*"))


def test_mp4_output(source, tmp_path):
    output = tmp_path / "result.mp4"
    export(source, output, set(), cloud(), [], True, lambda *args: None, Event())
    with av.open(str(output)) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        assert container.streams.audio[0].codec_context.name == "aac"
        assert len(list(container.decode(video=0))) == 10


def test_cancel_never_publishes_partial_output(source, tmp_path):
    event = Event()

    def progress(*args):
        event.set()

    output = tmp_path / "result.mkv"
    with pytest.raises(Cancelled):
        export(source, output, set(), cloud(), [], False, progress, event)
    assert not output.exists()
    assert not list(tmp_path.glob(".people-mosaic-*"))


def test_source_changes_and_overwrites_are_refused(source, tmp_path):
    with pytest.raises(UserError, match="上書き"):
        export(source, source.path, set(), cloud(), [], False, lambda *args: None, Event())
    source.path.write_bytes(source.path.read_bytes() + b"changed")
    with pytest.raises(UserError, match="変更"):
        export(
            source, tmp_path / "result.mkv", set(), cloud(), [], False, lambda *args: None, Event()
        )
