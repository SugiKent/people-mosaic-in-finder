import sys
from pathlib import Path

import pytest

from people_mosaic import engine, models, runtime
from people_mosaic.domain import Analysis
from people_mosaic.errors import UserError


def test_frozen_tools_ignore_system_path_and_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: pytest.fail("PATH fallback"))
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "ffprobe").write_bytes(b"bundled probe")
    assert runtime.tool_path("ffprobe") == str(tools / "ffprobe")
    with pytest.raises(UserError, match="再インストール"):
        runtime.tool_path("ffmpeg")


def test_frozen_model_preparation_never_downloads(monkeypatch):
    monkeypatch.setattr(models, "frozen", lambda: True)
    monkeypatch.setattr(models, "model_paths", lambda: (Path("detector"), Path("recognizer")))
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    models.download()


def test_mp4_software_fallback_uses_bundled_openh264(monkeypatch):
    monkeypatch.setattr(engine.sys, "platform", "linux")
    monkeypatch.setattr(engine.av, "codecs_available", {"libopenh264", "ffv1"})
    analysis = Analysis(Path("source"), "", 320, 240, 30)
    assert engine.mp4_codec(analysis, {}) == "libopenh264"
