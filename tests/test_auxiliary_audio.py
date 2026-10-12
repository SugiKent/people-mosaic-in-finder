import json
from types import SimpleNamespace

import pytest

from people_mosaic.engine import probe
from people_mosaic.errors import UserError


def metadata(monkeypatch, tmp_path, audio):
    path = tmp_path / "iphone.mov"
    path.write_bytes(b"fixture")
    streams = [
        {"codec_type": "video", "width": 64, "height": 48, "pix_fmt": "yuv420p"},
        *audio,
    ]
    monkeypatch.setattr("people_mosaic.engine.tool_path", lambda *args: "ffprobe")
    monkeypatch.setattr(
        "people_mosaic.engine.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps({"streams": streams})
        ),
    )
    return path


def test_iphone_aac_selected_without_unsupported_spatial_audio(monkeypatch, tmp_path):
    path = metadata(
        monkeypatch,
        tmp_path,
        [
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "codec_tag_string": "mp4a"},
            {"index": 2, "codec_type": "audio", "codec_tag_string": "apac"},
        ],
    )
    info = probe(path)
    assert info["audio_indices"] == [1]
    assert info["audio_codecs"] == ["aac"]


def test_spatial_only_audio_is_rejected_instead_of_silently_dropped(monkeypatch, tmp_path):
    path = metadata(
        monkeypatch, tmp_path, [{"index": 1, "codec_type": "audio", "codec_tag_string": "apac"}]
    )
    with pytest.raises(UserError, match="APAC"):
        probe(path)


def test_other_audio_tracks_are_preserved(monkeypatch, tmp_path):
    path = metadata(
        monkeypatch,
        tmp_path,
        [
            {"index": 1, "codec_type": "audio", "codec_name": "aac"},
            {"index": 4, "codec_type": "audio", "codec_name": "pcm_s16le"},
        ],
    )
    info = probe(path)
    assert info["audio_indices"] == [1, 4]
    assert info["audio_codecs"] == ["aac", "pcm_s16le"]
