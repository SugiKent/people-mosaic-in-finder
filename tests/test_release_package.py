"""Release publishing must never include local models, videos, or environments."""

import hashlib
import importlib.util
import json
import stat
from pathlib import Path
from zipfile import ZipFile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "package_release", Path(__file__).resolve().parents[1] / "scripts/package_release.py"
)
assert SPEC is not None and SPEC.loader is not None
package_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package_release)


def source_files():
    files = {
        path: (b"fixture source\n", 0o755 if path.endswith(".command") else 0o644)
        for path in package_release.REQUIRED_FILES
    }
    files["pyproject.toml"] = (b'[project]\nversion = "0.1.0"\n', 0o644)
    files["src/people_mosaic/engine.py"] = (b"# runtime source\n", 0o644)
    return files


def test_release_allowlist_and_command_permissions(tmp_path):
    files = source_files()
    for path in (
        ".git/config",
        ".venv/bin/python",
        "models/face.onnx",
        "input.mp4",
        "build/People Mosaic.app/Contents/Info.plist",
        "src/people_mosaic/private.png",
        "src/people_mosaic/__pycache__/app.py",
        "src/people_mosaic/../../private.py",
        "/src/people_mosaic/secret.py",
        "scripts/private.py",
    ):
        files[path] = (b"PRIVATE CONTENT MUST NOT SHIP", 0o644)
    output, checksum = package_release.package_files(files, tmp_path, "v0.1.0", commit="a" * 40)
    prefix = "People-Mosaic-0.1.0/"
    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {prefix + path for path in source_files()} | {
            prefix + "RELEASE.json"
        }
        assert json.loads(archive.read(prefix + "RELEASE.json")) == {
            "format": 1,
            "version": "0.1.0",
            "commit": "a" * 40,
        }
        for item in archive.infolist():
            mode = item.external_attr >> 16
            assert stat.S_ISREG(mode)
            assert bool(mode & stat.S_IXUSR) == item.filename.endswith(".command")
            assert b"PRIVATE CONTENT" not in archive.read(item)
    assert output.name == "People-Mosaic-0.1.0-macOS-arm64.zip"
    assert (
        checksum.read_text()
        == f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.name}\n"
    )


def test_release_is_reproducible_and_independent_of_input_order(tmp_path):
    files = source_files()
    first, _ = package_release.package_files(files, tmp_path / "first", commit="a" * 40)
    second, _ = package_release.package_files(
        dict(reversed(list(files.items()))), tmp_path / "second", commit="a" * 40
    )
    assert first.read_bytes() == second.read_bytes()


def test_missing_installer_fails_before_creating_archive(tmp_path):
    files = source_files()
    del files["Install.command"]
    with pytest.raises(ValueError, match="Missing release files: Install.command"):
        package_release.package_files(files, tmp_path, commit="a" * 40)
    assert not list(tmp_path.iterdir())


def test_tag_must_match_project_version(tmp_path):
    with pytest.raises(ValueError, match="does not match project version"):
        package_release.package_files(source_files(), tmp_path, "v0.2.0", commit="a" * 40)


@pytest.mark.parametrize("version", ["../private", "0.1.0/escape", "0.1", "not-a-version"])
def test_unsafe_or_invalid_version_is_rejected(tmp_path, version):
    files = source_files()
    files["pyproject.toml"] = (f'[project]\nversion = "{version}"\n'.encode(), 0o644)
    with pytest.raises(ValueError, match="safe semantic version"):
        package_release.package_files(files, tmp_path, commit="a" * 40)


def test_unexecutable_command_fails(tmp_path):
    files = source_files()
    files["Install.command"] = (b"#!/bin/zsh\n", 0o644)
    with pytest.raises(ValueError, match="must be executable in Git"):
        package_release.package_files(files, tmp_path, commit="a" * 40)


def test_git_revision_ignores_uncommitted_changes_and_local_assets(tmp_path):
    import subprocess

    repository = tmp_path / "repo"
    repository.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repository, check=True, capture_output=True, text=True
        )

    git("init", "--quiet")
    for path, (contents, mode) in source_files().items():
        target = repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        target.chmod(mode)
    git("add", ".")
    git(
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "Fixture release source",
    )
    git("tag", "v0.1.0")
    (repository / "src/people_mosaic/app.py").write_bytes(b"uncommitted source")
    (repository / "src/people_mosaic/private.py").write_bytes(b"untracked private data")
    (repository / "private-video.mp4").write_bytes(b"untracked private video")
    entries = package_release.git_files(repository, "v0.1.0")
    assert entries["src/people_mosaic/app.py"][0] == b"fixture source\n"
    assert "src/people_mosaic/private.py" not in entries
    assert "private-video.mp4" not in entries
    revision = git("rev-parse", "v0.1.0^{commit}").stdout.strip()
    output, _ = package_release.package_files(
        entries, tmp_path / "output", "v0.1.0", commit=revision
    )
    with ZipFile(output) as archive:
        assert archive.read("People-Mosaic-0.1.0/src/people_mosaic/app.py") == b"fixture source\n"
        assert json.loads(archive.read("People-Mosaic-0.1.0/RELEASE.json"))["commit"] == revision
