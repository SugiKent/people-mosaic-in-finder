"""Binary release packaging preserves app structure and excludes unrelated files."""

import hashlib
import importlib.util
import json
import os
import platform
import plistlib
import stat
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "package_release", Path(__file__).resolve().parents[1] / "scripts/package_release.py"
)
assert SPEC is not None and SPEC.loader is not None
package_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package_release)
COMMIT = "a" * 40
PREFIX = "People-Mosaic-0.1.0/"


def manifest(commit=COMMIT):
    return {"format": 2, "kind": "app-bundle", "version": "0.1.0", "commit": commit, "dirty": False}


def source_files():
    return {
        "Install.command": (b"#!/bin/zsh\n# offline installer\n", 0o755),
        "README.md": (b"Release installation instructions\n", 0o644),
        "pyproject.toml": (b'[project]\nversion = "0.1.0"\n', 0o644),
    }


@pytest.fixture
def bundle(tmp_path):
    app = tmp_path / "build/bundle/People Mosaic.app"
    executable = app / "Contents/MacOS/PeopleMosaic"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"bundled executable")
    executable.chmod(0o755)
    resources = app / "Contents/Resources"
    resources.mkdir()
    (resources / "RELEASE.json").write_text(json.dumps(manifest()))
    (resources / "model.onnx").write_bytes(b"verified model fixture")
    (app / "Contents/Info.plist").write_bytes(
        plistlib.dumps(
            {
                "CFBundleIdentifier": "local.people-mosaic.app",
                "CFBundleExecutable": "PeopleMosaic",
                "CFBundleShortVersionString": "0.1.0",
            }
        )
    )
    return app


def test_binary_release_structure_permissions_checksum_and_no_local_files(tmp_path, bundle):
    files = source_files()
    for path in ("private.mp4", ".venv/bin/python", "private.png", "src/secret.py"):
        files[path] = (b"PRIVATE MUST NOT SHIP", 0o644)
        (tmp_path / Path(path).name).write_bytes(b"PRIVATE MUST NOT SHIP")
    output, checksum = package_release.package_files(
        files, tmp_path / "output", "v0.1.0", commit=COMMIT, bundle=bundle
    )
    with ZipFile(output) as archive:
        assert json.loads(archive.read(PREFIX + "RELEASE.json")) == manifest()
        assert archive.read(PREFIX + "People Mosaic.app/Contents/Resources/model.onnx") == (
            b"verified model fixture"
        )
        for item in archive.infolist():
            assert item.filename.startswith(PREFIX)
            assert b"PRIVATE MUST NOT SHIP" not in archive.read(item)
            assert not item.filename.endswith("pyproject.toml")
        for name in ("Install.command", "People Mosaic.app/Contents/MacOS/PeopleMosaic"):
            mode = archive.getinfo(PREFIX + name).external_attr >> 16
            assert stat.S_ISREG(mode) and mode & stat.S_IXUSR
    assert checksum.read_text() == (
        f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.name}\n"
    )


def test_internal_relative_file_and_framework_symlinks_are_preserved(tmp_path, bundle):
    resources = bundle / "Contents/Resources"
    (resources / "model-link.onnx").symlink_to("model.onnx")
    framework = bundle / "Contents/Frameworks/Example.framework"
    (framework / "Versions/A").mkdir(parents=True)
    (framework / "Versions/A/Example").write_bytes(b"library")
    (framework / "Versions/Current").symlink_to("A", target_is_directory=True)
    (framework / "Example").symlink_to("Versions/Current/Example")
    output, _ = package_release.package_files(
        source_files(), tmp_path / "output", commit=COMMIT, bundle=bundle
    )
    with ZipFile(output) as archive:
        name = PREFIX + "People Mosaic.app/Contents/Frameworks/Example.framework/Example"
        assert stat.S_ISLNK(archive.getinfo(name).external_attr >> 16)
        assert archive.read(name) == b"Versions/Current/Example"
        current = (
            PREFIX + "People Mosaic.app/Contents/Frameworks/Example.framework/Versions/Current"
        )
        assert stat.S_ISLNK(archive.getinfo(current).external_attr >> 16)
        assert archive.read(current) == b"A"


@pytest.mark.parametrize("kind", ["absolute", "outside", "dangling", "loop"])
def test_unsafe_bundle_symlinks_rejected_before_output(tmp_path, bundle, kind):
    link = bundle / "Contents/Resources/link"
    outside = tmp_path / "private.png"
    outside.write_bytes(b"PRIVATE")
    targets = {
        "absolute": str(outside),
        "outside": "../../../../private.png",
        "dangling": "missing",
        "loop": "link",
    }
    link.symlink_to(targets[kind])
    with pytest.raises(ValueError, match="symlink"):
        package_release.package_files(
            source_files(), tmp_path / "output", commit=COMMIT, bundle=bundle
        )
    assert not (tmp_path / "output").exists()


def test_special_files_rejected(tmp_path, bundle):
    os.mkfifo(bundle / "Contents/Resources/pipe")
    with pytest.raises(ValueError, match="special filesystem entry"):
        package_release.package_files(
            source_files(), tmp_path / "output", commit=COMMIT, bundle=bundle
        )


def test_bundle_provenance_version_and_executable_are_required(tmp_path, bundle):
    manifest_path = bundle / "Contents/Resources/RELEASE.json"
    manifest_path.write_text(json.dumps(manifest("b" * 40)))
    with pytest.raises(ValueError, match="does not match release commit"):
        package_release.package_files(source_files(), tmp_path, commit=COMMIT, bundle=bundle)
    manifest_path.write_text(json.dumps(manifest()))
    executable = bundle / "Contents/MacOS/PeopleMosaic"
    executable.chmod(0o644)
    with pytest.raises(ValueError, match="not executable"):
        package_release.package_files(source_files(), tmp_path, commit=COMMIT, bundle=bundle)


def test_zip_is_reproducible_despite_input_order_and_file_mtime(tmp_path, bundle):
    first, _ = package_release.package_files(
        source_files(), tmp_path / "first", commit=COMMIT, bundle=bundle
    )
    for path in bundle.rglob("*"):
        os.utime(path, (1700000000, 1700000000))
    second, _ = package_release.package_files(
        dict(reversed(list(source_files().items()))),
        tmp_path / "second",
        commit=COMMIT,
        bundle=bundle,
    )
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("version", ["../private", "0.1.0/escape", "0.1", "not-a-version"])
def test_unsafe_version_is_rejected(tmp_path, bundle, version):
    files = source_files()
    files["pyproject.toml"] = (f'[project]\nversion = "{version}"\n'.encode(), 0o644)
    with pytest.raises(ValueError, match="safe semantic version"):
        package_release.package_files(files, tmp_path, commit=COMMIT, bundle=bundle)


def test_git_source_ignores_uncommitted_changes_and_untracked_media(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repository, check=True, capture_output=True, text=True
        )

    git("init", "--quiet")
    for path, (contents, mode) in source_files().items():
        target = repository / path
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
        "Fixture release",
    )
    (repository / "Install.command").write_bytes(b"UNCOMMITTED installer")
    (repository / "private.mp4").write_bytes(b"PRIVATE video")
    assert package_release.git_files(repository, "HEAD") == source_files()


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Installer targets Apple Silicon macOS",
)
def test_install_command_invokes_only_bundled_executable_without_developer_path(tmp_path):
    package = tmp_path / "Download with spaces 日本語"
    executable = package / "People Mosaic.app/Contents/MacOS/PeopleMosaic"
    executable.parent.mkdir(parents=True)
    executable.write_text('#!/bin/zsh\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    install = package / "Install.command"
    install.write_bytes((Path(__file__).resolve().parents[1] / "Install.command").read_bytes())
    install.chmod(0o755)
    result = subprocess.run(
        ["/bin/zsh", str(install)],
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["--install", str(package.resolve())]
