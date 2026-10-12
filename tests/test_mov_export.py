import subprocess
from fractions import Fraction
from threading import Event
from types import SimpleNamespace

import av
import numpy as np
import pytest

from people_mosaic import engine
from people_mosaic.domain import Analysis, Face
from people_mosaic.errors import UserError
from people_mosaic.models import digest
from people_mosaic.overlay import native_plane

MOV_CODEC = engine.mov_codec


@pytest.fixture(autouse=True)
def software_prores(monkeypatch):
    monkeypatch.setattr(engine, "mov_codec", lambda analysis: "prores_ks")


def make_source(
    tmp_path,
    *,
    hlg=False,
    audio_codec="pcm_s16le",
    rotation=False,
    full_range=False,
    bits=None,
    chroma422=False,
):
    bits = bits or (10 if hlg else 8)
    path = tmp_path / ("original.mov" if rotation else "original.mkv")
    times = [1200 + value for value in (0, 70, 120, 210, 270, 400, 490, 550, 690, 750, 850, 990)]
    with av.open(str(path), "w", options={"avoid_negative_ts": "disabled"}) as dest:
        video = dest.add_stream("libx264" if rotation else "ffv1", rate=20)
        video.width, video.height = 160, 120
        video.pix_fmt = "yuv422p10le" if chroma422 else "yuv420p10le" if bits == 10 else "yuv420p"
        video.time_base = video.codec_context.time_base = Fraction(1, 1000)
        video.codec_context.color_range = 2 if full_range else 1
        video.codec_context.colorspace = 9 if hlg else 1
        video.codec_context.color_primaries = 9 if hlg else 1
        video.codec_context.color_trc = 18 if hlg else 1
        video.options = {"crf": "10"} if rotation else {"level": "3"}
        audio = dest.add_stream(audio_codec, rate=48000)
        audio.layout = "mono"
        for index, pts in enumerate(times):
            frame = av.VideoFrame(160, 120, video.pix_fmt)
            frame.pts, frame.time_base, frame.duration = pts, Fraction(1, 1000), 50
            frame.color_range, frame.colorspace = (2 if full_range else 1), 9 if hlg else 1
            scale = 4 if bits == 10 else 1
            native_plane(frame, 0)[:] = (50 + index * 7) * scale
            native_plane(frame, 0)[15:30, 10:40] = (80 + index * 3) * scale
            native_plane(frame, 1)[:] = 128 * scale
            native_plane(frame, 2)[:] = 128 * scale
            if full_range:
                native_plane(frame, 0)[:40] = 0
                native_plane(frame, 0)[40:80] = 1 << (bits - 1)
                native_plane(frame, 0)[80:] = (1 << bits) - 1
            for packet in video.encode(frame):
                dest.mux(packet)
        for packet in video.encode():
            dest.mux(packet)
        samples = np.rint(15000 * np.sin(np.arange(48000) * (2 * np.pi * 440 / 48000)))
        # AAC accepts floating samples; PCM/FLAC use integer samples.
        data = (
            (samples / 32768).astype(np.float32)
            if audio_codec == "aac"
            else samples.astype(np.int16)
        )
        aframe = av.AudioFrame.from_ndarray(
            data.reshape(1, -1), format="fltp" if audio_codec == "aac" else "s16", layout="mono"
        )
        aframe.sample_rate = 48000
        aframe.pts, aframe.time_base = 62400, Fraction(1, 48000)
        for packet in audio.encode(aframe):
            dest.mux(packet)
        for packet in audio.encode():
            dest.mux(packet)
    if rotation:
        rotated = tmp_path / "rotated.mov"
        subprocess.run(
            [
                engine.tool_path("ffmpeg"),
                "-nostdin",
                "-v",
                "error",
                "-display_rotation:v:0",
                "90",
                "-i",
                str(path),
                "-map",
                "0",
                "-c",
                "copy",
                str(rotated),
            ],
            check=True,
        )
        rotated.replace(path)
    info = engine.probe(path)
    with av.open(str(path)) as source:
        originals = list(source.decode(video=0))
    width, height = (120, 160) if info["rotation"] % 2 else (160, 120)
    analysis = Analysis(
        path,
        digest(path),
        width,
        height,
        20,
        [[Face((60, 40, 20, 20))] for _ in originals],
        [float(frame.pts * frame.time_base) for frame in originals],
        hlg=hlg,
    )
    return analysis, originals


def write_mov(analysis, output, progress=lambda *args: None, cancel=None):
    rgba = np.full((64, 64, 4), 255, np.uint8)
    return engine.export(
        analysis, output, set(), rgba, [], False, progress, cancel or Event(), mov=True
    )


def audio_samples(path):
    with av.open(str(path)) as source:
        frames = list(source.decode(audio=0))
    return frames, np.concatenate([frame.to_ndarray().ravel() for frame in frames])


@pytest.mark.parametrize("hlg", [False, True])
def test_mov_prores_hq_white_colour_pixels_timing_and_pcm(tmp_path, hlg):
    analysis, originals = make_source(tmp_path, hlg=hlg)
    output = write_mov(analysis, tmp_path / "original_mosaic.mov")
    info = engine.probe(output)
    assert info["codec_name"] == "prores"
    assert info["profile"] == "HQ"
    assert info["codec_tag_string"] == "apch"
    assert info["pix_fmt"] == "yuv422p10le"
    assert info["width"] == 160 and info["height"] == 120
    assert info["color_range"] == "tv"
    assert info["color_space"] == ("bt2020nc" if hlg else "bt709")
    assert info["color_transfer"] == ("arib-std-b67" if hlg else "bt709")
    assert info["color_primaries"] == ("bt2020" if hlg else "bt709")
    with av.open(str(output)) as source:
        results = list(source.decode(video=0))
    assert len(results) == len(originals) == 12
    for original, result in zip(originals, results, strict=True):
        assert (
            abs(float(result.pts * result.time_base) - float(original.pts * original.time_base))
            < 0.001
        )
        # Opaque #fff stays white; intact regions have only small ProRes quantization error.
        assert np.max(np.abs(native_plane(result, 0)[48:52, 68:72].astype(int) - 940)) <= 4
        expected = original.reformat(format="yuv422p10le")
        for index in range(3):
            error = np.abs(
                native_plane(result, index)[:12, :12].astype(int)
                - native_plane(expected, index)[:12, :12].astype(int)
            )
            assert np.max(error) <= 4
    assert (
        abs(
            float(results[-1].duration * results[-1].time_base)
            - float(originals[-1].duration * originals[-1].time_base)
        )
        < 0.002
    )
    source_frames, source_audio = audio_samples(analysis.path)
    result_frames, result_audio = audio_samples(output)
    assert np.array_equal(source_audio, result_audio)
    assert source_frames[0].sample_rate == result_frames[0].sample_rate == 48000
    assert (
        abs(
            float(source_frames[0].pts * source_frames[0].time_base)
            - float(result_frames[0].pts * result_frames[0].time_base)
        )
        < 0.001
    )
    assert digest(analysis.path) == analysis.fingerprint
    assert not list(tmp_path.glob(".people-mosaic-*"))


def test_mov_aac_packets_are_copied_without_reencoding(tmp_path):
    analysis, _ = make_source(tmp_path, audio_codec="aac")
    output = write_mov(analysis, tmp_path / "result.mov")

    def packets(path):
        with av.open(str(path)) as source:
            return [bytes(packet) for packet in source.demux(audio=0) if packet.dts is not None]

    assert packets(output) == packets(analysis.path)
    with av.open(str(output)) as source:
        assert source.streams.audio[0].codec_context.name == "aac"


@pytest.mark.parametrize("hlg", [False, True])
def test_mov_unsupported_audio_transcodes_to_pcm_without_reencoding_video(
    tmp_path, monkeypatch, hlg
):
    analysis, originals = make_source(tmp_path, audio_codec="flac", hlg=hlg)
    original_remux = engine.remux_public_mp4
    remuxed = []

    def packets(path):
        with av.open(str(path)) as source:
            return [
                (p.pts * p.time_base, p.dts * p.time_base, p.duration * p.time_base, bytes(p))
                for p in source.demux(video=0)
                if p.dts is not None
            ]

    def checked_remux(video_path, audio_path, output, cancel):
        signature = packets(video_path)
        original_remux(video_path, audio_path, output, cancel)
        assert packets(output) == signature
        remuxed.append(True)

    monkeypatch.setattr(engine, "remux_public_mp4", checked_remux)
    output = write_mov(analysis, tmp_path / "result.mov")
    assert remuxed
    assert engine.probe(output)["color_transfer"] == ("arib-std-b67" if hlg else "bt709")
    with av.open(str(output)) as source:
        assert source.streams.audio[0].codec_context.name == "pcm_s24le"
        results = list(source.decode(video=0))
    assert len(results) == len(originals)
    a, source_audio = audio_samples(analysis.path)
    b, output_audio = audio_samples(output)
    # PyAV stores 24-bit PCM in a 32-bit sample container.
    assert np.array_equal(source_audio.astype(np.int32) << 16, output_audio)
    assert abs(float(a[0].pts * a[0].time_base) - float(b[0].pts * b[0].time_base)) < 0.001


def test_mov_bakes_input_rotation_without_stale_display_metadata(tmp_path):
    analysis, originals = make_source(tmp_path, rotation=True)
    assert engine.probe(analysis.path)["rotation"] % 2 == 1
    output = write_mov(analysis, tmp_path / "portrait.mov")
    info = engine.probe(output)
    assert (info["width"], info["height"]) == (120, 160)
    assert info["rotation"] == 0
    with av.open(str(output)) as source:
        results = list(source.decode(video=0))
    assert len(results) == len(originals)
    assert np.min(results[0].to_ndarray(format="bgr24")[48:52, 68:72]) >= 250


def test_mov_changed_source_and_overwrite_are_rejected(tmp_path):
    analysis, _ = make_source(tmp_path)
    with pytest.raises(UserError, match="上書き"):
        write_mov(analysis, analysis.path)
    analysis.path.write_bytes(analysis.path.read_bytes() + b"changed")
    with pytest.raises(UserError, match="変更"):
        write_mov(analysis, tmp_path / "result.mov")
    assert not (tmp_path / "result.mov").exists()


def test_mov_numbering_and_cancellation_never_overwrite_or_publish_partial(tmp_path):
    analysis, _ = make_source(tmp_path)
    output = tmp_path / "original_mosaic.mov"
    output.write_bytes(b"existing")
    output.with_name("original_mosaic_1.mov").write_bytes(b"also existing")
    assert write_mov(analysis, output) == output.with_name("original_mosaic_2.mov")
    assert output.read_bytes() == b"existing"
    cancel = Event()
    with pytest.raises(engine.Cancelled):
        write_mov(analysis, output, lambda *args: cancel.set(), cancel)
    assert not output.with_name("original_mosaic_3.mov").exists()
    assert not list(tmp_path.glob(".people-mosaic-*"))


def test_mov_and_mp4_cannot_be_requested_together_before_io(tmp_path):
    with pytest.raises(UserError, match="どちらか"):
        engine.export(
            None,
            tmp_path / "result.mov",
            set(),
            None,
            [],
            True,
            lambda *args: None,
            Event(),
            mov=True,
        )


def test_prores_hardware_probe_failure_falls_back_before_creating_output(monkeypatch):
    contexts = []

    class Context:
        def open(self):
            contexts.append(self)
            raise av.error.FFmpegError(5, "hardware unavailable")

    monkeypatch.setattr(engine.sys, "platform", "darwin")
    monkeypatch.setattr(engine.av, "codecs_available", {"prores_videotoolbox", "prores_ks"})
    monkeypatch.setattr(engine.av, "CodecContext", SimpleNamespace(create=lambda *args: Context()))
    assert MOV_CODEC(SimpleNamespace(width=1920, height=1080, fps=30)) == "prores_ks"
    assert contexts[0].pix_fmt == "p210le"
    assert contexts[0].options == {"profile": "3", "allow_sw": "0"}


def test_mov_source_mutated_during_export_is_discarded(tmp_path):
    analysis, _ = make_source(tmp_path)
    output = tmp_path / "result.mov"

    def mutate(*args):
        analysis.path.write_bytes(analysis.path.read_bytes() + b"changed")

    with pytest.raises(UserError, match="変更"):
        write_mov(analysis, output, mutate)
    assert not output.exists()
    assert not list(tmp_path.glob(".people-mosaic-*"))


def test_prores_hardware_probe_success_selects_native_encoder(monkeypatch):
    class Context:
        def open(self):
            assert self.pix_fmt == "p210le"
            assert self.options == {"profile": "3", "allow_sw": "0"}

    monkeypatch.setattr(engine.sys, "platform", "darwin")
    monkeypatch.setattr(engine.av, "codecs_available", {"prores_videotoolbox", "prores_ks"})
    monkeypatch.setattr(engine.av, "CodecContext", SimpleNamespace(create=lambda *args: Context()))
    assert MOV_CODEC(SimpleNamespace(width=1920, height=1080, fps=30)) == "prores_videotoolbox"


def test_missing_prores_encoder_is_a_clear_error_before_io(monkeypatch):
    monkeypatch.setattr(engine.av, "codecs_available", set())
    with pytest.raises(UserError, match="MOV 用"):
        MOV_CODEC(SimpleNamespace(width=1920, height=1080, fps=30))


@pytest.mark.parametrize(
    "bits,hlg,chroma422",
    [(8, False, False), (10, False, False), (10, True, False), (10, False, True), (10, True, True)],
)
def test_mov_converts_full_range_black_midtone_white_and_opaque_cloud(
    tmp_path, bits, hlg, chroma422
):
    analysis, originals = make_source(
        tmp_path, full_range=True, bits=bits, hlg=hlg, chroma422=chroma422
    )
    assert all(frame.color_range == 2 for frame in originals)
    output = write_mov(analysis, tmp_path / "full_range.mov")
    info = engine.probe(output)
    assert info["color_range"] == "tv"
    assert info["color_space"] == ("bt2020nc" if hlg else "bt709")
    assert info["color_transfer"] == ("arib-std-b67" if hlg else "bt709")
    with av.open(str(output)) as source:
        results = list(source.decode(video=0))
    assert len(results) == len(originals)
    expected_mid = 64 + 876 * (1 << (bits - 1)) / ((1 << bits) - 1)
    for frame in results:
        assert frame.color_range == 1
        y = native_plane(frame, 0)
        assert np.max(np.abs(y[8:16, 8:16].astype(int) - 64)) <= 4
        assert np.max(np.abs(y[48:56, 8:16].astype(int) - expected_mid)) <= 4
        assert np.max(np.abs(y[88:96, 8:16].astype(int) - 940)) <= 4
        assert np.max(np.abs(y[48:52, 68:72].astype(int) - 940)) <= 4
        for index in (1, 2):
            assert np.max(np.abs(native_plane(frame, index)[8:16, 8:16].astype(int) - 512)) <= 4
    # PCM and source bytes retain their existing preservation guarantees.
    assert np.array_equal(audio_samples(analysis.path)[1], audio_samples(output)[1])
    assert digest(analysis.path) == analysis.fingerprint


@pytest.mark.parametrize("pix_fmt", ["yuv420p", "yuv420p10le", "yuv422p10le"])
def test_mov_compositor_converts_range_even_when_pixel_format_already_matches(pix_fmt):
    bits = 8 if pix_fmt == "yuv420p" else 10
    frame = av.VideoFrame(160, 120, pix_fmt)
    frame.color_range, frame.colorspace = 2, 1
    frame.pts, frame.time_base, frame.duration = 1200, Fraction(1, 1000), 50
    maximum = (1 << bits) - 1
    native_plane(frame, 0)[:40] = 0
    native_plane(frame, 0)[40:80] = 1 << (bits - 1)
    native_plane(frame, 0)[80:] = maximum
    for index in (1, 2):
        native_plane(frame, index)[:] = 1 << (bits - 1)
    result = engine.FrameCompositor(
        np.full((64, 64, 4), 255, np.uint8), 0, False, "yuv422p10le", Event(), limited_range=True
    )(frame, [Face((60, 40, 20, 20))])
    assert result.color_range == 1
    assert result.colorspace == 1
    assert (result.pts, result.time_base, result.duration) == (1200, Fraction(1, 1000), 50)
    assert np.max(np.abs(native_plane(result, 0)[8:16, 8:16].astype(int) - 64)) <= 4
    assert np.max(np.abs(native_plane(result, 0)[88:96, 8:16].astype(int) - 940)) <= 4
    assert np.max(np.abs(native_plane(result, 0)[48:52, 68:72].astype(int) - 940)) <= 4
