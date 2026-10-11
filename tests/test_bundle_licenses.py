"""Source assets preserve notices without extracting untrusted archive paths."""

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "collect_bundle_licenses",
    Path(__file__).resolve().parents[1] / "scripts/collect_bundle_licenses.py",
)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def test_source_notice_collection_ignores_symlinks_and_escaping_paths(tmp_path):
    source = tmp_path / "upstream.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        for name, contents in (
            ("project/LICENSE", b"copyright and license"),
            ("project/thirdparty/NOTICE.txt", b"third party notice"),
            ("../LICENSE", b"must not escape"),
            ("/LICENSE", b"must not escape"),
            ("project/run.py", b"not a notice"),
        ):
            entry = tarfile.TarInfo(name)
            entry.size = len(contents)
            archive.addfile(entry, io.BytesIO(contents))
        entry = tarfile.TarInfo("project/COPYING")
        entry.type, entry.linkname = tarfile.SYMTYPE, "/private/unrelated"
        archive.addfile(entry)
    output = tmp_path / "notices"
    assert collector.copy_source_notices(source, output) == 2
    assert sorted(
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    ) == [
        "upstream.tar.gz/project/LICENSE",
        "upstream.tar.gz/project/thirdparty/NOTICE.txt",
    ]
    assert not (tmp_path / "LICENSE").exists()


def test_download_checks_hash_and_does_not_replace_previous_source_on_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "source.tar.gz"
    path.write_bytes(b"previous archive")
    monkeypatch.setattr(collector, "request", lambda _: io.BytesIO(b"corrupt response"))
    expected = hashlib.sha256(b"correct source").hexdigest()
    with pytest.raises(ValueError, match="checksum mismatch"):
        collector.download("https://official.example/source.tar.gz", tmp_path, expected=expected)
    assert path.read_bytes() == b"previous archive"
    assert not path.with_name("source.tar.gz.download").exists()


def test_verified_download_cache_is_reused_without_network(tmp_path, monkeypatch):
    path = tmp_path / "source.tar.gz"
    path.write_bytes(b"verified source")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    def unexpected(_):
        raise AssertionError("Verified source should not be downloaded again")

    monkeypatch.setattr(collector, "request", unexpected)
    assert (
        collector.download("https://official.example/source.tar.gz", tmp_path, expected=expected)
        == path
    )


def test_installed_notice_files_keep_package_version_and_names(tmp_path, monkeypatch):
    installed = tmp_path / "runtime"
    installed.mkdir()
    (installed / "LICENSE.txt").write_text("runtime license")
    (installed / "private.png").write_bytes(b"not for release notices")

    class Distribution:
        version = "1.2.3"
        files = ["LICENSE.txt", "private.png"]

        def read_text(self, name):
            assert name == "METADATA"
            return "Name: runtime\nVersion: 1.2.3\n"

        def locate_file(self, path):
            return installed / path

    monkeypatch.setattr(collector.importlib.metadata, "distribution", lambda _: Distribution())
    output = tmp_path / "notices"
    assert collector.installed_notices(output, ["runtime"]) == [
        {"name": "runtime", "version": "1.2.3"}
    ]
    assert (output / "installed/runtime/LICENSE.txt").read_text() == "runtime license"
    assert not list(output.rglob("*.png"))


def test_model_notices_record_pinned_weight_and_license_hashes(tmp_path, monkeypatch):
    root = tmp_path / "app"
    (root / "src/people_mosaic").mkdir(parents=True)
    (root / "models").mkdir()
    weight = root / "models/model.onnx"
    weight.write_bytes(b"verified weights")
    model_hash = collector.digest(weight)
    (root / "src/people_mosaic/models.py").write_text(
        f"MODELS = {{'model.onnx': ('face_detection_yunet', '{model_hash}')}}\n"
    )
    notice = tmp_path / "LICENSE.txt"
    notice.write_text("verified notice")
    calls = []

    def download(url, cache, *, filename, expected):
        calls.append((url, filename, expected))
        return notice

    monkeypatch.setattr(collector, "download", download)
    records, sources = collector.model_notices(tmp_path / "notices", tmp_path / "cache", root)
    assert records[0]["sha256"] == model_hash
    assert records[0]["license_sha256"] == collector.digest(notice)
    assert collector.ZOO_COMMIT in records[0]["license_url"]
    assert calls[0][2] == collector.MODEL_LICENSE_HASHES["face_detection_yunet"]
    assert sources[0]["version"] == collector.ZOO_COMMIT
    weight.write_bytes(b"modified weights")
    with pytest.raises(ValueError, match="Bundled model checksum mismatch"):
        collector.model_notices(tmp_path / "notices", tmp_path / "cache", root)
    assert len(calls) == 1
